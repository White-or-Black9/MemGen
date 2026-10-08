"""No-model acceptance of recovery ownership, exclusion and stop behavior."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('m3_recovery', Path(__file__).with_name('resume_pinned_worker.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
c = r.c


class RecoveryTests(unittest.TestCase):
    def test_three_original_cards_claim_only_their_own_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue = dict(status='RUNNING', jobs={j['id']: dict(status='done') for j in c.jobs()})
            for job, gpu in [('s442-c2', '6'), ('s442-c3', '7'), ('s442-c4', '0')]:
                queue['jobs'][job] = dict(status='pending', previous_owner={'gpu': gpu})
            c.atomic(root / 'queue.json', queue)
            # Claim in a different order than the manifest to test affinity.
            for gpu, expected in [('0', 's442-c4'), ('7', 's442-c3'), ('6', 's442-c2')]:
                worker = c.owner(gpu)
                self.assertEqual(r.pinned_claim(root, worker)['id'], expected)
                self.assertIsNone(r.pinned_claim(root, worker))
                c.finish(root, {'id': expected}, worker)
            self.assertEqual(c.read(root / 'queue.json')['status'], 'GENERATED')

    def test_stopped_queue_cannot_be_claimed(self):
        with tempfile.TemporaryDirectory() as folder:
            c.atomic(Path(folder) / 'queue.json', {'status': 'STOPPED'})
            self.assertIsNone(r.pinned_claim(folder, c.owner('0')))


if __name__ == '__main__':
    unittest.main()
