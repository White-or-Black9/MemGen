"""Stdlib M3 controller: prepare, locked queue, integrity checks and aggregation.

The worker imports torch lazily. No changes to the bounded M2 entry point.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
POLICIES = ('full', 'random', 'last_written', 'cosine_only', 'recency_only', 'native_order')
SEEDS = (42, 142, 242, 342, 442)
CONTEXTS = tuple(range(5))
QUESTIONS = 100
MAB = Path('/mnt/18T/baishilong/benchmarks/MemoryAgentBench')
SCORER_PYTHON = '/home/baishilong/miniconda3/envs/MABench/bin/python'
PROTOCOL = dict(top_k=2, max_slots=16, retrieve_threshold=.05, update_threshold=.10,
                decay_alpha=.05, generation_max_length=40, chunk_size=4096,
                requested_contexts=5, question_limit=100, reseed_per_context=True,
                trace_score_decomposition=True, save_frozen_bank=True,
                eventqa_protocol='frozen_context_bank', strict_official_eventqa_prompt=False,
                first_line_official_eventqa_prompt=False)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    """Replace only a complete fsynced JSON; a crash cannot expose a partial row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked(campaign, name='queue'):
    with (Path(campaign) / (name + '.lock')).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def jobs():
    return [dict(id=f's{seed}-c{ctx}', seed=seed, context_index=ctx,
                 construction_seed=seed + ctx) for seed in SEEDS for ctx in CONTEXTS]


def identity(manifest, job, query, policy):
    return dict(campaign=manifest['campaign'], seed=job['seed'],
                context_index=job['context_index'], query_id=query, policy=policy)


def expected(manifest):
    return {digest(identity(manifest, job, q, p)) for job in manifest['jobs']
            for q in range(QUESTIONS) for p in POLICIES}


def source_paths():
    return sorted(set(ROOT.glob('memgen/**/*.py')) | set(ROOT.glob('interactions/**/*.py')) |
                  set(ROOT.glob('scripts/eval/*.py')) | {ROOT / 'main.py',
                  ROOT / 'scripts/eval/run_review_m3_formal.sh'})


def prepare(campaign):
    campaign = Path(campaign).resolve()
    if campaign.exists():
        raise FileExistsError('campaign exists; use worker/resume, never overwrite')
    # Model path identities are frozen Hugging Face revisions, all shard files hashed.
    model = Path('/home/baishilong/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
    checkpoint = Path('/home/baishilong/.cache/huggingface/hub/models--Kana-s--MemGen/snapshots/269d9b1741130b94fffa410cdaa3d4bc74081a7f/Qwen2.5-1.5B-Instruct/triviaqa/weaver-sft/pn=8_pl=8_in=0_il=8/model')
    paths = [ROOT / 'review/nference_Time_Latent_Me.pdf', ROOT / 'configs/latent_memory/triviaqa.yaml',
             MAB / 'configs/data_conf/Accurate_Retrieval/EventQA/Eventqa_64k.yaml',
             MAB / 'utils/eval_other_utils.py',
             Path('/mnt/18T/baishilong/datasets/MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet')]
    for directory in (model, checkpoint):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths.extend(sorted(p for p in directory.rglob('*') if p.is_file()))
    # Official bridge/import dependencies are also part of the input contract.
    paths.extend(sorted((MAB / 'utils').glob('*.py')))
    manifest = dict(schema='m3-formal/v1', campaign=campaign.name, created_at=now(),
                    protocol=PROTOCOL, policies=list(POLICIES), seeds=list(SEEDS), contexts=list(CONTEXTS),
                    jobs=jobs(), expected_records=15000,
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_paths()},
                    input_sha256={str(p): sha(p) for p in set(paths)},
                    model_path=str(model), checkpoint_path=str(checkpoint),
                    reviewer_items=['R03', 'R07'], parent='research_notes/next_version/m2/M2_REPORT.md',
                    paper_role='main_required', section_id='retrieval_ablation', item_id='M3',
                    claim_links=['R03', 'R07'], comparison_baselines=['random', 'last_written'],
                    evidence_role='claim-carrying', metric='strict_candidate_accuracy',
                    comparability='current-source new campaign; not historical dirty-P7 reproduction',
                    resource_contract='all eligible RTX A6000 with >=12GiB free; one worker/card',
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    manifest['contract_sha256'] = digest(manifest)
    if len(expected(manifest)) != 15000:
        raise RuntimeError('manifest identity collision')
    campaign.mkdir(parents=True)
    atomic(campaign / 'manifest.json', manifest)
    atomic(campaign / 'queue.json', dict(status='READY', jobs={j['id']: dict(status='pending') for j in jobs()}))
    (campaign / 'tracked_source.diff').write_text(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT, text=True))
    return manifest


def verify_manifest(manifest, *, check_inputs=False):
    unsigned = dict(manifest)
    contract = unsigned.pop('contract_sha256')
    if digest(unsigned) != contract:
        raise ValueError('manifest contract hash mismatch')
    current = {str(p.relative_to(ROOT)): sha(p) for p in source_paths()}
    if current != manifest['source_sha256']:
        raise ValueError('source drift: cannot resume or mix campaign')
    if check_inputs:
        for path, value in manifest['input_sha256'].items():
            if sha(path) != value:
                raise ValueError(f'input drift: {path}')


def owner(gpu):
    return dict(host=socket.gethostname(), pid=os.getpid(), gpu=str(gpu))


def claim(campaign, worker):
    with locked(campaign):
        queue = read(Path(campaign) / 'queue.json')
        if queue['status'] == 'STOPPED':
            return None
        for job in jobs():
            if queue['jobs'][job['id']]['status'] == 'pending':
                queue['jobs'][job['id']] = dict(status='running', owner=worker, started_at=now())
                queue['status'] = 'RUNNING'
                atomic(Path(campaign) / 'queue.json', queue)
                return job
    return None


def finish(campaign, job, worker, error=None):
    with locked(campaign):
        queue = read(Path(campaign) / 'queue.json')
        state = queue['jobs'][job['id']]
        if state.get('owner') != worker or state['status'] != 'running':
            raise RuntimeError('lost job ownership')
        state.update(status='failed' if error else 'done', finished_at=now())
        if error:
            state['error'] = error
            queue['status'] = 'STOPPED'
        elif all(s['status'] == 'done' for s in queue['jobs'].values()):
            queue['status'] = 'GENERATED'
        atomic(Path(campaign) / 'queue.json', queue)


def resume(campaign):
    """Explicit recovery; never steal live work, or retry a failed protocol automatically."""
    verify_manifest(read(Path(campaign) / 'manifest.json'), check_inputs=True)
    with locked(campaign):
        queue = read(Path(campaign) / 'queue.json')
        if any(s['status'] == 'failed' for s in queue['jobs'].values()):
            raise RuntimeError('failed job requires diagnosis/new campaign, not blind resume')
        for state in queue['jobs'].values():
            if state['status'] != 'running':
                continue
            previous = state['owner']
            if previous['host'] != socket.gethostname():
                raise RuntimeError('cannot establish remote worker liveness')
            try:
                os.kill(previous['pid'], 0)
            except ProcessLookupError:
                state.update(status='pending', recovered_at=now(), previous_owner=previous)
                state.pop('owner')
            else:
                continue
        atomic(Path(campaign) / 'queue.json', queue)
    return queue


def record_path(campaign, job, q, p):
    return Path(campaign) / 'jobs' / job['id'] / 'queries' / f'q{q:03d}_{p}.json'


def load_record(path, ident, provenance):
    """Only verified complete envelopes can be skipped; corrupt bytes are preserved."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        envelope = read(path)
        content = dict(envelope)
        checksum = content.pop('content_sha256')
        if checksum != digest(content) or envelope['complete'] is not True:
            raise ValueError('incomplete/hash mismatch')
        if not isinstance(envelope['record'], dict) or not isinstance(envelope['runtime_result'], dict):
            raise ValueError('invalid record envelope')
        if envelope['identity'] != ident or envelope['provenance'] != provenance:
            raise RuntimeError('record provenance/identity drift')
        return envelope
    except (ValueError, KeyError, TypeError) as exc:
        preserved = path.with_name(path.name + '.corrupt.' + datetime.now().strftime('%Y%m%dT%H%M%S%f'))
        os.replace(path, preserved)
        atomic(path.with_name(path.name + '.error.json'), dict(error=str(exc), preserved=str(preserved)))
        return None


def save_record(path, ident, provenance, record, runtime):
    envelope = dict(complete=True, identity=ident, provenance=provenance,
                    record=record, runtime_result=runtime, finished_at=now())
    envelope['content_sha256'] = digest(envelope)
    for attempt in range(2):
        try:
            atomic(path, envelope)
            return envelope
        except OSError as exc:
            # Only genuine transient filesystem errors, never ENOSPC or permission.
            if attempt or exc.errno not in (4, 11, 16, 116):
                raise


def run_job(campaign, manifest, job, ensure_snapshot, execute_query):
    """Dependency-injected engine allows CPU acceptance without importing torch."""
    snapshot, provenance = ensure_snapshot(job)
    for q in range(QUESTIONS):
        records = []
        for policy in POLICIES:
            if read(Path(campaign) / 'queue.json')['status'] == 'STOPPED':
                raise RuntimeError('campaign stopped by another worker')
            ident = identity(manifest, job, q, policy)
            path = record_path(campaign, job, q, policy)
            envelope = load_record(path, ident, provenance)
            if envelope is None:
                attempt_root = path.parent.parent / 'attempts' / f'q{q:03d}_{policy}'
                attempt_root.mkdir(parents=True, exist_ok=True)
                attempt = len(list(attempt_root.glob('attempt-*.json'))) + 1
                # Persist BEFORE calling the model; interruption cannot hide a prior try.
                atomic(attempt_root / f'attempt-{attempt}.json',
                       dict(identity=ident, provenance=provenance, started_at=now(),
                            status='STARTED', owner=owner(os.environ.get('CUDA_VISIBLE_DEVICES', 'mock'))))
                record, runtime = execute_query(snapshot, job, q, policy)
                record['generation_attempts'] = attempt
                record['attempt_accounting'] = 'started execution invocations; an interrupted invocation may not have completed generation'
                envelope = save_record(path, ident, provenance, record, runtime)
                atomic(attempt_root / f'attempt-{attempt}.json',
                       dict(identity=ident, provenance=provenance, finished_at=now(), status='SAVED',
                            result_content_sha256=envelope['content_sha256']))
            records.append(envelope['record'])
        budgets = {(r['selection'][0]['n'], r['selection'][0]['latent_count']) for r in records}
        if len(budgets) != 1:
            raise RuntimeError(f'cross-policy budget mismatch {job["id"]}/q{q}')
        print(f'QUESTION_COMPLETE job={job["id"]} q={q}', flush=True)


def collect(campaign, manifest):
    """Refuse an incomplete or misidentified campaign; never rank partial results."""
    rows = []
    seen = set()
    for job in manifest['jobs']:
        meta = read(Path(campaign) / 'jobs' / job['id'] / 'snapshot_identity.json')
        snapshot_path = Path(campaign) / 'jobs' / job['id'] / 'frozen_bank.pt'
        if sha(snapshot_path) != meta['snapshot_file_sha256']:
            raise ValueError('snapshot file changed')
        for q in range(QUESTIONS):
            paired = []
            for p in POLICIES:
                ident = identity(manifest, job, q, p)
                envelope = load_record(record_path(campaign, job, q, p), ident, meta)
                if envelope is None:
                    raise ValueError(f'incomplete campaign: {ident}')
                key = digest(ident)
                if key in seen:
                    raise ValueError('duplicate identity')
                seen.add(key)
                row = envelope['record']
                if any(row.get(k) != ident[k] for k in ('seed', 'context_index', 'query_id', 'policy')):
                    raise ValueError('record identity does not match envelope')
                runtime = envelope['runtime_result']
                if (row['query_write_count_delta'] != 0 or not row['snapshot_unchanged'] or
                    runtime['query_write_count_delta'] != 0 or not runtime['query_read_only_enforced'] or
                    runtime['bank_snapshot_changed_after_query'] or runtime['query_retrieval_invocation_count'] != 1):
                    raise ValueError('query frozen/read-only contract failed')
                attempts = list((Path(campaign) / 'jobs' / job['id'] / 'attempts' / f'q{q:03d}_{p}').glob('attempt-*.json'))
                if not attempts or row.get('generation_attempts') != len(attempts):
                    raise ValueError('generation attempt accounting mismatch')
                if len(row['selection']) != 1:
                    raise ValueError('multiple selection traces')
                trace = row['selection'][0]
                if trace['n'] not in (0, 1, 2) or trace['latent_count'] != 8 * trace['n'] or len(trace['selected_indices']) != trace['n']:
                    raise ValueError('selection/latent budget failed')
                paired.append(row)
                rows.append(row)
            if len({digest([r['context_id'], r['question'], r['gold_answers'], r['query_sha256']]) for r in paired}) != 1:
                raise ValueError('paired question/gold/query mismatch')
            if len({(r['selection'][0]['n'], r['selection'][0]['latent_count']) for r in paired}) != 1:
                raise ValueError('paired budget mismatch')
    if seen != expected(manifest):
        raise ValueError('incomplete expected identity set')
    return rows


def aggregate_rows(rows):
    from scripts.eval import review_eventqa_decomposition as m
    safe = [dict(r, official_em=r['official_em'] or 0,
                 official_raw_recall=r['official_raw_recall'] or 0) for r in rows]
    summary = {p: m.summarize([r for r in safe if r['policy'] == p]) for p in POLICIES}
    conditional = {p: m.summarize([r for r in safe if r['policy'] == p and r['selection'][0]['n'] > 0]) for p in POLICIES}
    full = {(r['seed'], r['context_index'], r['query_id']): r for r in rows if r['policy'] == 'full'}
    contrasts = {}
    rng = random.Random(20261002)
    for policy in POLICIES[1:]:
        other = [r for r in rows if r['policy'] == policy]
        deltas = [(r['seed'], r['context_index'],
                   int(full[(r['seed'], r['context_index'], r['query_id'])]['strict_candidate_correct']) - int(r['strict_candidate_correct'])) for r in other]
        seedctx = {f's{s}-c{c}': sum(d for ss, cc, d in deltas if (ss, cc) == (s, c)) / QUESTIONS for s in SEEDS for c in CONTEXTS}
        contexts = [sum(seedctx[f's{s}-c{c}'] for s in SEEDS) / len(SEEDS) for c in CONTEXTS]
        seeds = [sum(seedctx[f's{s}-c{c}'] for c in CONTEXTS) / len(CONTEXTS) for s in SEEDS]
        bootstrap = sorted(sum(rng.choice(contexts) for _ in CONTEXTS) / len(CONTEXTS) for _ in range(10000))
        contrasts[policy] = dict(mean_delta=sum(contexts) / 5, wins=sum(d == 1 for _, _, d in deltas),
                                 losses=sum(d == -1 for _, _, d in deltas), ties=sum(d == 0 for _, _, d in deltas),
                                 seed_context_deltas=seedctx, context_means=contexts, seed_means=seeds,
                                 positive_contexts=sum(d > 0 for d in contexts), positive_seeds=sum(d > 0 for d in seeds),
                                 exploratory_context_block_ci95=[bootstrap[249], bootstrap[9749]])
        both_valid = [r for r in other if r['candidate_valid'] and full[(r['seed'], r['context_index'], r['query_id'])]['candidate_valid']]
        contrasts[policy]['both_candidate_valid'] = dict(
            denominator=len(both_valid), selection_bias='output-dependent subset; not causal full-set accuracy',
            full_correct=sum(full[(r['seed'], r['context_index'], r['query_id'])]['strict_candidate_correct'] for r in both_valid),
            other_correct=sum(r['strict_candidate_correct'] for r in both_valid))
        contrasts[policy]['selection_overlap_by_seed_context'] = {
            f's{s}-c{cc}': dict(denominator=QUESTIONS, same_set=sum(
                set(r['selection'][0]['selected_indices']) == set(full[(s, cc, r['query_id'])]['selection'][0]['selected_indices'])
                for r in other if (r['seed'], r['context_index']) == (s, cc)))
            for s in SEEDS for cc in CONTEXTS}
    stable = all(contrasts[p]['mean_delta'] > 0 and contrasts[p]['positive_contexts'] >= 4
                 and contrasts[p]['positive_seeds'] >= 4 for p in ('random', 'last_written'))
    recency = {(r['seed'], r['context_index'], r['query_id']): r for r in rows if r['policy'] == 'recency_only'}
    overlap = sum(set(r['selection'][0]['selected_indices']) == set(recency[k]['selection'][0]['selected_indices']) for k, r in full.items())
    selection_variation = {}
    for s in SEEDS:
        for c in CONTEXTS:
            traces = [r['selection'][0] for k, r in full.items() if k[:2] == (s, c)]
            selection_variation[f's{s}-c{c}'] = dict(unique_sets=len({tuple(t['selected_indices']) for t in traces}),
                cosine_range=_range([v for t in traces for v in t['cosine']]),
                age_range=_range([v for t in traces for v in t['ages']]),
                final_score_range=_range([v for t in traces for v in t['final_scores']]))
    return dict(summary=summary, n_positive=conditional, paired_full=contrasts,
                full_recency_same_set=dict(numerator=overlap, denominator=len(full)),
                full_selection_variation=selection_variation, direction_stable=stable,
                bootstrap=dict(unit='whole context including all questions and five seeds', iterations=10000, seed=20261002,
                               caveat='exploratory only; five contexts, not IID question-level significance'),
                next_route='plan relevance-label analysis' if stable else 'plan decay/representation/output diagnosis',
                claim_boundary='direction stability is not significance or proof of semantic retrieval')


def _range(values):
    return [min(values), max(values)] if values else None


def eligible_gpus(csv_text):
    cards = list(csv.DictReader(csv_text.splitlines(), skipinitialspace=True))
    return [card['index'].strip() for card in cards
            if card['name'].strip() == 'NVIDIA RTX A6000' and
            float(card['memory.free [MiB]'].split()[0]) >= 12 * 1024]


def launch(campaign):
    """Explicit launch command, after CPU gate; use every currently eligible card."""
    campaign = Path(campaign).resolve()
    manifest = read(campaign / 'manifest.json')
    verify_manifest(manifest, check_inputs=True)
    if campaign.parent != ROOT / 'outputs/mab/review_m3_formal':
        raise ValueError('launcher requires the dedicated M3 output root')
    if read(campaign / 'queue.json')['status'] != 'READY':
        raise ValueError('initial launch only; use resume then explicit worker for recovery')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.free,memory.used,utilization.gpu',
                                     '--format=csv'], text=True)
    cards = eligible_gpus(query)
    audit = dict(time=now(), gpu_csv=query, eligible=cards,
                 load_average=os.getloadavg(), free_disk_bytes=__import__('shutil').disk_usage(campaign).free,
                 tmux='/home/baishilong/miniconda3/bin/tmux')
    atomic(campaign / 'launch_audit.json', audit)
    if not cards:
        raise RuntimeError('no eligible GPU; no workers launched')
    sessions = {gpu: f'memgen-m3-{campaign.name}-gpu{gpu}' for gpu in cards}
    for session in sessions.values():
        if subprocess.run([audit['tmux'], 'has-session', '-t', session], capture_output=True).returncode == 0:
            raise RuntimeError('existing tmux session; no duplicate launch')
    for gpu, session in sessions.items():
        subprocess.run([audit['tmux'], 'new-session', '-d', '-s', session, '-c', str(ROOT),
                        'bash', 'scripts/eval/run_review_m3_formal.sh', gpu, campaign.name], check=True)
    atomic(campaign / 'launch.json', dict(time=now(), sessions=sessions, status='WORKERS_STARTING'))
    return dict(sessions=sessions, expected_records=15000, status='WORKERS_STARTING')


def score(campaign):
    from scripts.eval import review_eventqa_decomposition as m
    campaign = Path(campaign)
    with locked(campaign, 'aggregate'):
        manifest = read(campaign / 'manifest.json')
        verify_manifest(manifest, check_inputs=True)
        queue = read(campaign / 'queue.json')
        if queue['status'] != 'GENERATED' or any(s['status'] != 'done' for s in queue['jobs'].values()):
            raise ValueError('not all 25 jobs generated; formal aggregation refused')
        rows = collect(campaign, manifest)
        official = m.load_official(MAB / 'utils/eval_other_utils.py')
        for row in rows:
            info = m.candidate_info(row['question'], row['gold_answers'], official)
            if info['errors']:
                raise ValueError(info['errors'])
            row.update(m.evaluate(row['prediction'], row['gold_answers'], info, row['format_flags'], official))
        result = aggregate_rows(rows)
        result.update(status='COMPLETE', count=len(rows), contract_sha256=manifest['contract_sha256'], finished_at=now())
        result['execution_attempts_started'] = sum(r['generation_attempts'] for r in rows)
        result['resumed_records_with_prior_attempts'] = sum(r['generation_attempts'] > 1 for r in rows)
        atomic(campaign / 'scored_records.json', rows)
        atomic(campaign / 'results.json', result)
        lines = ['# M3 正式六策略结果', '', '|策略|题数|严格候选正确率|官方 EM|候选有效率|', '|---|---:|---:|---:|---:|']
        for p in POLICIES:
            r = result['summary'][p]
            lines.append(f"|{p}|{r['question_count']}|{r['strict_candidate_accuracy']['value']:.4f}|{r['em']['value']:.4f}|{r['candidate_valid_rate']['value']:.4f}|")
        lines += ['', f"描述性方向稳定：{result['direction_stable']}。", '区间仅为五个 context 的探索性 block bootstrap；不证明统计显著或语义检索。',
                  '这是当前源码的新 campaign，不是历史 dirty P7 的精确复现。', f"下一路线（需另行批准）：{result['next_route']}。", '']
        (campaign / 'RESULTS.md').write_text('\n'.join(lines))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'launch', 'worker', 'resume', 'status', 'score'])
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.campaign)
    elif args.command == 'launch':
        result = launch(args.campaign)
    elif args.command == 'worker':
        from scripts.eval.review_m3_worker import worker
        return worker(Path(args.campaign), args.gpu)
    elif args.command == 'resume':
        result = resume(args.campaign)
    elif args.command == 'score':
        result = score(args.campaign)
    else:
        result = read(Path(args.campaign) / 'queue.json')
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
