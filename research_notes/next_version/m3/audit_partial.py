"""Read-only, model-free audit of an atomic prefix of the active M3 campaign.

Kept outside scripts/eval: this validator does NOT change frozen inference sources.
No partial scoring, no queue edits, no repairs, no torch/pickle snapshot loading.
Only the explicitly requested audit report is written, via the controller's atomic API.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.eval import review_m3_formal as c


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(campaign):
    campaign = Path(campaign).resolve()
    manifest = c.read(campaign / 'manifest.json')
    c.verify_manifest(manifest, check_inputs=True)
    environment = c.read(campaign / 'environment_contract.json')
    queue = c.read(campaign / 'queue.json')
    seen = set()
    by_job = {}
    by_policy = Counter()
    groups = defaultdict(list)
    global_question_ids = {}
    recorded_attempts = 0
    # Snapshot the published paths first. A concurrently published later record
    # is out of this audit's scope, not a missing/error row.
    frozen_paths = {job['id']: sorted((campaign / 'jobs' / job['id'] / 'queries').glob('q*.json'))
                    for job in manifest['jobs']}
    for job in manifest['jobs']:
        files = [p for p in frozen_paths[job['id']] if re.fullmatch(r'q\d{3}_[a-z_]+\.json', p.name)]
        if not files:
            continue
        directory = campaign / 'jobs' / job['id']
        meta = c.read(directory / 'snapshot_identity.json')
        context = c.read(directory / 'context_identity.json')
        require(meta['contract_sha256'] == manifest['contract_sha256'], 'snapshot campaign mismatch')
        require(meta['provenance_sha256'] == c.digest(context), 'context provenance mismatch')
        require(context['seed'] == job['seed'] and context['context_index'] == job['context_index'], 'context identity mismatch')
        require(context['environment'] == environment, 'snapshot environment drift')
        require(c.sha(directory / 'frozen_bank.pt') == meta['snapshot_file_sha256'], 'snapshot byte hash mismatch')
        by_job[job['id']] = dict(records=0, complete_six_policy_questions=0,
                                 snapshot_file_sha256=meta['snapshot_file_sha256'], snapshot_verified=True)
        for path in files:
            envelope = c.read(path)
            signed = dict(envelope)
            checksum = signed.pop('content_sha256')
            require(checksum == c.digest(signed), f'record checksum mismatch: {path}')
            require(envelope['complete'] is True, 'incomplete published record')
            ident = envelope['identity']
            row = envelope['record']
            runtime = envelope['runtime_result']
            q, policy = ident['query_id'], ident['policy']
            require(policy in c.POLICIES and 0 <= q < c.QUESTIONS, 'out-of-scope query/policy')
            require(ident == c.identity(manifest, job, q, policy), 'envelope identity mismatch')
            require(path == c.record_path(campaign, job, q, policy), 'path identity mismatch')
            require(envelope['provenance'] == meta, 'record provenance mismatch')
            require(all(row[k] == ident[k] for k in ('seed', 'context_index', 'query_id', 'policy')), 'row identity mismatch')
            key = c.digest(ident)
            require(key not in seen, 'duplicate output identity')
            seen.add(key)
            require(row['context_id'] == context['context_id'], 'row context mismatch')
            question_hash = hashlib.sha256(row['question'].encode()).hexdigest()
            require(question_hash == context['question_identity'][q]['question_text_hash'], 'question text drift')
            require(row['query_write_count_delta'] == runtime['query_write_count_delta'] == 0, 'query writes occurred')
            require(row['snapshot_unchanged'] is True and runtime['query_read_only_enforced'] is True, 'read-only contract failed')
            require(runtime['bank_snapshot_changed_after_query'] is False and runtime['query_retrieval_invocation_count'] == 1,
                    'frozen state/retrieval count failed')
            require(len(row['selection']) == 1, 'multiple selection traces')
            trace = row['selection'][0]
            n = trace['n']
            selected = trace['selected_indices']
            require(n in (0, 1, 2) and trace['latent_count'] == 8 * n, 'latent budget mismatch')
            require(len(selected) == len(set(selected)) == n, 'selection count mismatch')
            require(all(0 <= i < len(trace['ages']) for i in selected), 'invalid slot index')
            require(policy == 'native_order' or selected == sorted(selected), 'canonical order mismatch')
            attempts = list((directory / 'attempts' / f'q{q:03d}_{policy}').glob('attempt-*.json'))
            require(row['generation_attempts'] == len(attempts) and len(attempts) >= 1, 'attempt accounting mismatch')
            recorded_attempts += len(attempts)
            signature = c.digest([row['context_id'], row['question'], row['gold_answers'], row['query_sha256'],
                                  runtime['rendered_query_prompt']])
            shared_key = (job['context_index'], q)
            if shared_key in global_question_ids:
                require(global_question_ids[shared_key] == signature, 'question/gold/prompt drift across seeds/policies')
            global_question_ids[shared_key] = signature
            groups[(job['id'], q)].append((policy, n, trace['latent_count']))
            by_job[job['id']]['records'] += 1
            by_policy[policy] += 1
    for (job_id, q), group in groups.items():
        require(len({(n, latent) for _, n, latent in group}) == 1, f'paired budget drift: {job_id}/{q}')
        if len(group) == len(c.POLICIES):
            require({p for p, _, _ in group} == set(c.POLICIES), 'duplicate/missing paired policy')
            by_job[job_id]['complete_six_policy_questions'] += 1
    require(seen <= c.expected(manifest), 'unexpected records')
    return dict(schema='m3-partial-integrity/v1', status='PASS_PARTIAL', checked_at=c.now(),
                campaign=manifest['campaign'], contract_sha256=manifest['contract_sha256'],
                audit_source_sha256=c.sha(__file__), source_input_hashes_verified=True,
                verified_records=len(seen), expected_records=manifest['expected_records'],
                verified_execution_attempts_started=recorded_attempts,
                by_job=by_job, by_policy=dict(by_policy),
                queue_status_counts=dict(Counter(s['status'] for s in queue['jobs'].values())),
                caveat='atomic prefix only; not formal completion, scoring, significance or retrieval effectiveness',
                next_route='continue current M3 generation; aggregate only after all 25 jobs complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = audit(args.campaign)
    except Exception as exc:
        c.atomic(args.output, dict(status='FAILED_AUDIT', checked_at=c.now(), error=f'{type(exc).__name__}: {exc}'))
        raise
    c.atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
