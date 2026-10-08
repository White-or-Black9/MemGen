"""Model-heavy implementation for the separately frozen M3 campaign."""
from pathlib import Path
import os
import subprocess
import traceback

from scripts.eval import review_m3_formal as c
from scripts.eval.review_m2_smoke import eventqa, metrics, preflight, verify_query
from scripts.eval.review_retrieval_policies import (
    capture_bank, capture_rng, derive_last_write, fingerprint, load_snapshot,
    policy_adapter, restore_bank, restore_rng, save_snapshot,
)


def worker(campaign, gpu):
    campaign = campaign.resolve()
    worker_id = c.owner(gpu)
    manifest = c.read(campaign / 'manifest.json')
    active_job = None
    model = None
    try:
        c.verify_manifest(manifest, check_inputs=True)
        args = eventqa.build_parser().parse_args([])
        for name, value in manifest['protocol'].items():
            setattr(args, name, value)
        args.model_path = manifest['model_path']
        args.checkpoint_path = manifest['checkpoint_path']
        args.snapshot = None
        args.output_root = str(campaign)
        check = preflight(args)
        print(check, flush=True)
        if check['gate'] != 'PASS' or check.get('gpu_name') != 'NVIDIA RTX A6000' or check.get('free_vram_bytes', 0) < 12 * 1024**3:
            raise RuntimeError('fresh GPU/cache gate failed; no model loaded')
        c.atomic(campaign / 'workers' / f'gpu{gpu}-{os.getpid()}.json',
                 dict(owner=worker_id, preflight=check, started_at=c.now()))
        rows = eventqa._load_rows(args.parquet, eventqa.SUB_DATASET)
        if len(rows) != 5:
            raise ValueError('expected five EventQA contexts')
        config = eventqa._eventqa_bank_config(args)
        model, capacity = eventqa.weaver_bank._load_model(args)
        runtime = eventqa._runtime_reproducibility_metadata(args, selected_context_indices=list(c.CONTEXTS), model=model)
        environment = {key: runtime[key] for key in ('python_version', 'python_executable', 'torch_version',
                       'transformers_version', 'cuda_version', 'dtype', 'checkpoint_path', 'tokenizer_path')}
        with c.locked(campaign, 'environment'):
            path = campaign / 'environment_contract.json'
            if path.exists() and c.read(path) != environment:
                raise ValueError('worker environment drift')
            if not path.exists():
                c.atomic(path, environment)
        c.atomic(campaign / 'workers' / f'gpu{gpu}-{os.getpid()}-runtime.json', runtime)
        contexts = {}

        def ensure_snapshot(job):
            ctx = job['context_index']
            if ctx not in contexts:
                # Freeze timestamp too: prompt construction must not depend on worker start time.
                context = eventqa.build_context_payload(args, rows[ctx], ctx, manifest['created_at'])
                if len(eventqa._question_identity_records(context)) != c.QUESTIONS:
                    raise ValueError('expected 100 questions per context')
                contexts[ctx] = context
            context = contexts[ctx]
            output = campaign / 'jobs' / job['id']
            output.mkdir(parents=True, exist_ok=True)
            provenance = dict(contract_sha256=manifest['contract_sha256'], seed=job['seed'],
                              context_index=ctx, context_id=context['context_id'], bank_config=config,
                              environment=environment,
                              chunk_sha256=[metrics.text_sha(chunk) for chunk in context['chunks']],
                              question_identity=eventqa._question_identity_records(context))
            snapshot = output / 'frozen_bank.pt'
            if not snapshot.exists():
                # An interrupted construction is not a successful construction. Preserve its attempt.
                attempts = len(list(output.glob('construction_attempt_*.json')))
                c.atomic(output / f'construction_attempt_{attempts + 1}.json',
                         dict(started_at=c.now(), owner=worker_id, seed=job['construction_seed']))
                eventqa._prepare_context_rng(base_seed=job['seed'], context_index=ctx, reseed_per_context=True)
                construction = eventqa._run_eventqa_model(
                    args, model, capacity, eventqa._construction_only_payload(context), 'on', config,
                    preserve_bank=True, construction_only=True, recorded_bank_config=config)
                bank = construction.pop('_retained_bank')
                last_write = derive_last_write(construction['construction_turn_diagnostics'], len(bank))
                c.atomic(output / 'construction.json', construction)
                temporary = output / f'frozen_bank.{os.getpid()}.tmp'
                save_snapshot(temporary, capture_bank(bank), capture_rng(), last_write, provenance)
                # Validate bytes before publishing a bank for all subsequent questions.
                load_snapshot(temporary)
                os.replace(temporary, snapshot)
                bank.reset()
                del bank
            saved = load_snapshot(snapshot)
            if saved['provenance'] != provenance:
                raise ValueError('snapshot provenance drift')
            if any(slot.memory.shape[0] != 8 for slot in saved['bank']['_slots']):
                raise ValueError('expected eight latent tokens per slot')
            meta = dict(provenance_sha256=c.digest(provenance), contract_sha256=manifest['contract_sha256'],
                        snapshot_file_sha256=c.sha(snapshot), bank_sha256=saved['bank_sha256'])
            identity_file = output / 'snapshot_identity.json'
            if identity_file.exists() and c.read(identity_file) != meta:
                raise ValueError('snapshot identity drift')
            c.atomic(identity_file, meta)
            c.atomic(output / 'context_identity.json', provenance)
            c.atomic(output / 'runtime.json', dict(owner=worker_id, runtime=runtime,
                                                 base_seed=job['seed'], construction_seed=job['construction_seed']))
            return saved, meta

        def execute_query(saved, job, q, policy):
            question = eventqa.build_question_payload(contexts[job['context_index']], q)
            payload = eventqa._query_only_payload(question)
            bank = restore_bank(saved['bank'])
            restore_rng(saved['rng'])
            before = fingerprint(capture_bank(bank))
            trace = []
            with policy_adapter(bank, policy, last_write=saved['last_write'], seed=job['seed'],
                                context=job['context_index'], query=q, trace=trace):
                result = eventqa._run_eventqa_model(args, model, capacity, payload, 'on', config,
                                                  external_bank=bank, preserve_bank=True, recorded_bank_config=config)
            result.pop('_retained_bank', None)
            verify_query(result, trace, before, bank)
            record = dict(policy=policy, seed=job['seed'], context_index=job['context_index'], query_id=q,
                          context_id=question.get('context_id', contexts[job['context_index']]['context_id']),
                          question=question['question'], gold_answers=question['gold_answers'],
                          prediction=result['prediction'], query_sha256=metrics.text_sha(payload['query_prompt']),
                          selection=trace, format_flags=eventqa._format_flags(result['prediction']),
                          query_write_count_delta=result['query_write_count_delta'], snapshot_unchanged=True,
                          generation_attempts=1, owner=worker_id)
            bank.reset()
            return record, result

        print(f'WORKER_READY gpu={gpu} pid={os.getpid()}', flush=True)
        while True:
            c.verify_manifest(manifest)
            active_job = c.claim(campaign, worker_id)
            if active_job is None:
                break
            args.seed = active_job['seed']
            args.context_index = active_job['context_index']
            print(f'JOB_START {active_job["id"]} gpu={gpu}', flush=True)
            c.run_job(campaign, manifest, active_job, ensure_snapshot, execute_query)
            # Recheck input/source identity at the evidence boundary, not just startup.
            c.verify_manifest(manifest, check_inputs=True)
            c.finish(campaign, active_job, worker_id)
            print(f'JOB_COMPLETE {active_job["id"]} gpu={gpu}', flush=True)
            active_job = None
        if c.read(campaign / 'queue.json')['status'] == 'GENERATED':
            subprocess.run([c.SCORER_PYTHON, str(c.ROOT / 'scripts/eval/review_m3_formal.py'),
                            'score', '--campaign', str(campaign)], cwd=c.ROOT, check=True)
        print(f'WORKER_EXIT gpu={gpu}', flush=True)
        return 0
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        c.atomic(campaign / 'workers' / f'gpu{gpu}-{os.getpid()}-error.json',
                 dict(error=error, traceback=traceback.format_exc(), job=active_job, owner=worker_id, time=c.now()))
        if active_job is not None:
            c.finish(campaign, active_job, worker_id, error=error)
        else:
            with c.locked(campaign):
                queue = c.read(campaign / 'queue.json')
                queue.update(status='STOPPED', worker_error=error)
                c.atomic(campaign / 'queue.json', queue)
        raise
