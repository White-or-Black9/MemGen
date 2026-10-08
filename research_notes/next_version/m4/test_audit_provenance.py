"""M4-A pure CPU tests; no Torch, dataset, models or parent mutations."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('m4_audit', Path(__file__).with_name('audit_provenance.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture(actions):
    turns, slots = [], []
    for i, (action, index, retrieved) in enumerate(actions):
        step = i + 1
        before = len(slots)
        for k in retrieved:
            slots[k]['access_count'] += 1
            slots[k]['last_retrieved_step'] = step
        value = dict(slot_index=index, created_step=step, last_retrieved_step=step, access_count=0)
        if action == 'insert':
            slots.append(value)
        else:
            slots[index] = value
        turns.append(dict(construction_turn_index=i, write_action=action, target_slot_index=index,
                          slot_count_before_write=before, slot_count_after_write=len(slots),
                          write_count_after_write=step, retrieved_indices_before_write=retrieved,
                          replaced_slot_index=index if action == 'replace_matched' else None,
                          evicted_slot_index=index if action == 'evict_oldest_insert' else None,
                          created_step_after_write=step, last_retrieved_step_after_write=step,
                          access_count_after_write=0,
                          per_slot_access_count_after_write=[s['access_count'] for s in slots]))
    for s in slots:
        s['last_retrieved_age'] = len(turns) - s['last_retrieved_step']
    return turns, dict(slot_count=len(slots), write_count=len(turns), slots=slots), ['hash'] * len(turns)


class ProvenanceTests(unittest.TestCase):
    def test_eviction_reuses_index_not_content_version(self):
        args = fixture([('insert', 0, []), ('insert', 1, []), ('evict_oldest_insert', 0, [1])])
        saved = copy.deepcopy(args)
        slots, events = a.replay(*args)
        self.assertEqual(slots[0]['slot_version'], 'slot0-v2')
        self.assertEqual(slots[0]['direct_source_chunk'], 2)
        self.assertEqual(slots[0]['potential_source_chunks'], [1, 2])
        self.assertEqual(events[-1]['superseded_version'], 'slot0-v1')
        self.assertEqual(args, saved)

    def test_refresh_does_not_assume_old_semantics_preserved(self):
        slots, _ = a.replay(*fixture([('insert', 0, []), ('replace_matched', 0, [])]))
        self.assertEqual(slots[0]['refresh_count'], 1)
        self.assertEqual(slots[0]['potential_source_chunks'], [1])
        self.assertIsNone(slots[0]['latent_semantic_relevance'])

    def test_refresh_retrieval_links_old_version_only_as_possible_dependency(self):
        slots, _ = a.replay(*fixture([('insert', 0, []), ('replace_matched', 0, [0])]))
        self.assertEqual(slots[0]['prior_retrieved_versions'], ['slot0-v1'])
        self.assertEqual(slots[0]['potential_source_chunks'], [0, 1])
        self.assertEqual(slots[0]['access_count'], 0)

    def test_missing_turn_rejected(self):
        turns, summary, chunks = fixture([('insert', 0, []), ('insert', 1, [])])
        turns[1]['construction_turn_index'] = 2
        with self.assertRaisesRegex(ValueError, 'missing'):
            a.replay(turns, summary, chunks)

    def test_summary_drift_rejected(self):
        turns, summary, chunks = fixture([('insert', 0, [])])
        summary['slots'][0]['created_step'] = 7
        with self.assertRaisesRegex(ValueError, 'metadata drift'):
            a.replay(turns, summary, chunks)

    def test_chunk_hash_drift_rejected(self):
        a.verify_chunks(['hello'], [a.text_sha('hello')])
        with self.assertRaisesRegex(ValueError, 'chunk hash'):
            a.verify_chunks(['changed'], [a.text_sha('hello')])

    def test_original_offsets_with_joined_whitespace(self):
        spans = a.chunk_spans('First.\n\nSecond.', ['First. Second.'], ['First.', 'Second.'])
        self.assertEqual(spans[0]['char_end'], 15)
        self.assertEqual(spans[0]['sentence_spans'], [[0, 6], [8, 15]])

    def test_boundary_match_is_lexical_not_semantic(self):
        spans = [dict(chunk_index=0, char_start=0, char_end=5),
                 dict(chunk_index=1, char_start=6, char_end=10)]
        row = a.locate_gold('alpha beta', ['alpha beta'], spans)
        self.assertTrue(row['lexical_matches'][0]['crosses_chunk_boundary'])
        self.assertIsNone(row['source_evidence_overlap'])

    def test_multiple_matches_and_multigold_no_first_match_selection(self):
        row = a.locate_gold('A B A', ['A', 'B'], [dict(chunk_index=0, char_start=0, char_end=5)])
        self.assertEqual(row['lexical_status'], 'multiple')
        self.assertEqual(len(row['lexical_matches']), 3)
        self.assertIsNone(row['latent_semantic_relevance'])

    def test_no_match_is_unknown_not_negative_evidence(self):
        row = a.locate_gold('source prose', ['paraphrased event'], [])
        self.assertEqual(row['lexical_status'], 'absent')
        self.assertEqual(row['evidence_status'], 'unknown_no_source_span')
        self.assertIsNone(row['source_evidence_overlap'])

    def test_blind_sample_has_fifty_unique_questions_not_seed_repeats(self):
        rows = [dict(context_index=ctx, query_id=q) for ctx in range(5) for q in range(100)]
        sample = a.blind_sample(rows)
        self.assertEqual(len(sample), 50)
        self.assertEqual(sample, a.blind_sample(list(reversed(rows))))
        self.assertEqual(len({(r['context_index'], r['query_id']) for r in sample}), 50)

    def test_parent_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a.c.atomic(root / 'record.json', {'original': True})
            before = a.inventory(root)
            a.check_unchanged(before, root)
            a.c.atomic(root / 'record.json', {'original': False})
            with self.assertRaisesRegex(ValueError, 'parent files changed'):
                a.check_unchanged(before, root)


if __name__ == '__main__':
    unittest.main()
