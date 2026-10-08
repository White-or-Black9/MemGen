"""No torch/model imports: queue, exact identities, crash/resume and engine tests."""
import contextlib
import io
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.eval import review_m3_formal as c


def competing_claim(path, results):
    job = c.claim(path, c.owner('mock'))
    results.put(job['id'])


class FormalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.manifest = dict(campaign='test', jobs=c.jobs())
        c.atomic(self.path / 'queue.json', dict(status='READY', jobs={j['id']: dict(status='pending') for j in c.jobs()}))

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_25_jobs_15000_unique_records(self):
        self.assertEqual(len(c.jobs()), 25)
        self.assertEqual(len(c.expected(self.manifest)), 15000)
        for j in c.jobs():
            self.assertEqual(j['construction_seed'], j['seed'] + j['context_index'])

    def test_all_eligible_cards_and_threshold(self):
        sample = ('index, name, memory.free [MiB], memory.used [MiB], utilization.gpu [%]\n'
                  '0, NVIDIA RTX A6000, 12288 MiB, 35000 MiB, 100 %\n'
                  '1, NVIDIA RTX A6000, 12287 MiB, 35000 MiB, 100 %\n'
                  '2, Different GPU, 20000, 10000, 0\n'
                  '6, NVIDIA RTX A6000, 17600, 30000, 100\n')
        self.assertEqual(c.eligible_gpus(sample), ['0', '6'])

    def test_concurrent_claims_are_unique(self):
        queue = mp.Queue()
        processes = [mp.Process(target=competing_claim, args=(self.path, queue)) for _ in range(8)]
        for p in processes:
            p.start()
        claimed = [queue.get(timeout=20) for _ in processes]
        for p in processes:
            p.join(timeout=20)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(len(set(claimed)), 8)

    def test_finish_ownership_and_stop(self):
        owner = c.owner(0)
        job = c.claim(self.path, owner)
        with self.assertRaises(RuntimeError):
            c.finish(self.path, job, c.owner(1))
        c.finish(self.path, job, owner, error='contract failure')
        self.assertIsNone(c.claim(self.path, c.owner(1)))

    def test_atomic_hash_complete_and_corruption_preserved(self):
        path = self.path / 'q.json'
        c.save_record(path, {'id': 1}, {'sha': 2}, {'prediction': 'A'}, {})
        self.assertTrue(c.load_record(path, {'id': 1}, {'sha': 2})['complete'])
        envelope = c.read(path)
        envelope['record']['prediction'] = 'changed'
        c.atomic(path, envelope)
        self.assertIsNone(c.load_record(path, {'id': 1}, {'sha': 2}))
        self.assertEqual(len(list(self.path.glob('q.json.corrupt.*'))), 1)

    def test_valid_wrong_provenance_is_not_retried(self):
        path = self.path / 'q.json'
        c.save_record(path, {'id': 1}, {'sha': 2}, {}, {})
        with self.assertRaises(RuntimeError):
            c.load_record(path, {'id': 1}, {'sha': 3})
        self.assertTrue(path.exists())

    def test_source_drift_rejected(self):
        m = {'source_sha256': {'x.py': 'old'}}
        m['contract_sha256'] = c.digest(m)
        with patch.object(c, 'source_paths', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'source drift'):
                c.verify_manifest(m)

    def test_resume_does_not_steal_live_worker(self):
        live = c.claim(self.path, c.owner(0))
        dead = c.claim(self.path, dict(host=c.socket.gethostname(), pid=999999999, gpu='1'))
        c.atomic(self.path / 'manifest.json', self.manifest)
        with patch.object(c, 'verify_manifest'):
            result = c.resume(self.path)
        self.assertEqual(result['jobs'][live['id']]['status'], 'running')
        self.assertEqual(result['jobs'][dead['id']]['status'], 'pending')

    def test_interruption_resume_one_construction_and_query_reset(self):
        job = c.jobs()[0]
        self.assertEqual(c.claim(self.path, c.owner('mock')), job)
        constructions, calls = [], []
        state = {}

        def snapshot(_):
            if not state:
                constructions.append(1)
                state.update(bank=0, rng=42)
            return dict(state), {'source': 'fixed'}

        def execute(saved, _, q, p):
            self.assertEqual(saved, {'bank': 0, 'rng': 42})
            calls.append((q, p))
            if len(calls) == 9:
                raise InterruptedError('mock crash')
            return {'selection': [{'n': 2, 'latent_count': 16}]}, {}

        with patch.object(c, 'QUESTIONS', 3), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(InterruptedError):
                c.run_job(self.path, self.manifest, job, snapshot, execute)
            first = len(calls)
            c.run_job(self.path, self.manifest, job, snapshot, execute)
        self.assertEqual(len(constructions), 1)
        self.assertEqual(len(calls) - first, 10)  # 18 total outputs, eight already durable.
        self.assertEqual(len(list((self.path / 'jobs' / job['id'] / 'queries').glob('q*.json'))), 18)
        recovered = c.read(c.record_path(self.path, job, 1, 'last_written'))
        self.assertEqual(recovered['record']['generation_attempts'], 2)
        self.assertEqual(len(list((self.path / 'jobs' / job['id'] / 'attempts').rglob('attempt-*.json'))), 19)

    def test_budget_mismatch_stops(self):
        job = c.jobs()[0]
        def execute(_, __, q, p):
            return {'selection': [{'n': 1 if p == 'random' else 2, 'latent_count': 8 if p == 'random' else 16}]}, {}
        with patch.object(c, 'QUESTIONS', 1), self.assertRaisesRegex(RuntimeError, 'budget mismatch'):
            c.run_job(self.path, self.manifest, job, lambda _: ({}, {}), execute)

    def test_partial_campaign_refuses_scoring(self):
        c.atomic(self.path / 'manifest.json', self.manifest)
        with patch.object(c, 'verify_manifest'), self.assertRaisesRegex(ValueError, 'not all 25'):
            c.score(self.path)

    def test_transient_io_retries_once_no_generation_retry(self):
        real = c.atomic
        attempts = []
        def flaky(path, value):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError(11, 'temporary unavailable')
            real(path, value)
        with patch.object(c, 'atomic', side_effect=flaky):
            c.save_record(self.path / 'q.json', {}, {}, {}, {})
        self.assertEqual(len(attempts), 2)

    def test_block_aggregation_pairs_by_seed_context_not_query_alone(self):
        from scripts.eval import review_eventqa_decomposition as m
        official = m.load_official()
        question = "Below is a list of possible subsequent events: ['alpha', 'beta']"
        info = m.candidate_info(question, ['alpha'], official)
        rows = []
        for s in c.SEEDS:
            for ctx in c.CONTEXTS:
                for p in c.POLICIES:
                    prediction = 'beta' if p in ('random', 'last_written') else 'alpha'
                    row = dict(seed=s, context_index=ctx, query_id=0, policy=p,
                               selection=[dict(n=2, latent_count=16, selected_indices=[0, 1],
                                               cosine=[.4, .5], ages=[1, 2], final_scores=[.2, .3])])
                    row.update(m.evaluate(prediction, ['alpha'], info, {}, official))
                    rows.append(row)
        with patch.object(c, 'QUESTIONS', 1):
            result = c.aggregate_rows(rows)
        self.assertTrue(result['direction_stable'])
        self.assertEqual(result['paired_full']['random']['wins'], 25)
        self.assertEqual(result['paired_full']['random']['mean_delta'], 1)
        self.assertEqual(result['paired_full']['random']['exploratory_context_block_ci95'], [1, 1])
        self.assertEqual(result['paired_full']['recency_only']['mean_delta'], 0)
        self.assertEqual(result['full_recency_same_set']['numerator'], 25)

    def test_full_collect_and_pair_identity_failure(self):
        job = c.jobs()[0]
        manifest = dict(campaign='test', jobs=[job])
        snapshot = self.path / 'jobs' / job['id'] / 'frozen_bank.pt'
        snapshot.parent.mkdir(parents=True)
        snapshot.touch()
        provenance = dict(snapshot_file_sha256=c.sha(snapshot))
        c.atomic(snapshot.parent / 'snapshot_identity.json', provenance)
        runtime = dict(query_write_count_delta=0, query_read_only_enforced=True,
                       bank_snapshot_changed_after_query=False, query_retrieval_invocation_count=1)
        for p in c.POLICIES:
            row = dict(seed=job['seed'], context_index=job['context_index'], query_id=0, policy=p,
                       context_id='ctx', question='same', gold_answers=['A'], query_sha256='query',
                       selection=[dict(n=2, latent_count=16, selected_indices=[0, 1])],
                       query_write_count_delta=0, snapshot_unchanged=True, generation_attempts=1)
            c.save_record(c.record_path(self.path, job, 0, p), c.identity(manifest, job, 0, p), provenance, row, runtime)
            c.atomic(snapshot.parent / 'attempts' / f'q000_{p}' / 'attempt-1.json', {'status': 'SAVED'})
        with patch.object(c, 'QUESTIONS', 1):
            self.assertEqual(len(c.collect(self.path, manifest)), 6)
            row['question'] = 'different'
            c.save_record(c.record_path(self.path, job, 0, p), c.identity(manifest, job, 0, p), provenance, row, runtime)
            with self.assertRaisesRegex(ValueError, 'paired question'):
                c.collect(self.path, manifest)


if __name__ == '__main__':
    unittest.main()
