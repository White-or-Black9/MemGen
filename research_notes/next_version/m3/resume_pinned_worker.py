"""Recovery-only GPU affinity for the last M3 jobs; inference code is unchanged.

Wrap only the locked claim operation, using resume's previous_owner.gpu. This
prevents a fast recovered worker from taking a different GPU's unfinished job.
The wrapper's own SHA and queue snapshots are saved as recovery evidence.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.eval import review_m3_formal as c


def pinned_claim(campaign, worker):
    with c.locked(campaign):
        queue = c.read(Path(campaign) / 'queue.json')
        if queue['status'] == 'STOPPED':
            return None
        for job in c.jobs():
            state = queue['jobs'][job['id']]
            if state['status'] != 'pending':
                continue
            previous = state.get('previous_owner')
            if not previous:
                raise RuntimeError('recovery requires audited previous ownership')
            if previous['gpu'] != worker['gpu']:
                continue
            state.update(status='running', owner=worker, started_at=c.now())
            queue['status'] = 'RUNNING'
            c.atomic(Path(campaign) / 'queue.json', queue)
            return job
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--gpu', required=True, choices=['0', '6', '7'])
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    campaign = Path(args.campaign).resolve()
    log = ROOT / 'runtime_logs/review_m3_formal' / campaign.name / f'gpu{args.gpu}'
    log.mkdir(parents=True, exist_ok=True)
    with (log / 'run.log').open('a', buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        print(f'RESUME_START time={c.now()} gpu={args.gpu} pid={os.getpid()}', flush=True)
        c.atomic(campaign / 'recoveries' / f'20261008-gpu{args.gpu}-{os.getpid()}.json',
                 dict(time=c.now(), owner=c.owner(args.gpu), scheduler_sha256=c.sha(__file__),
                      scheduler='previous_owner GPU affinity; frozen inference functions unchanged'))
        c.claim = pinned_claim
        status = 1
        try:
            from scripts.eval.review_m3_worker import worker
            status = worker(campaign, args.gpu)
            return status
        finally:
            with (log / 'exit_code').open('a') as output:
                output.write(f'{status}\n')
            c.atomic(log / f'resume-exit-{os.getpid()}.json', dict(time=c.now(), exit_code=status))


if __name__ == '__main__':
    raise SystemExit(main())
