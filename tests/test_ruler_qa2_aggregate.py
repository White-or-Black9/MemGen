import unittest

from scripts.eval.ruler_qa2_aggregate import aggregate_records


class RulerQA2AggregateTest(unittest.TestCase):
    def test_aggregate_records_counts_accuracy_and_memory_usage(self):
        records = [
            {"correct": True, "memory_write_count": 2, "retrieval_count": 1},
            {"correct": False, "memory_write_count": 2, "retrieval_count": 0},
        ]
        summary = aggregate_records(records)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["memory_write_count"], 4)
        self.assertEqual(summary["retrieval_count"], 1)


if __name__ == "__main__":
    unittest.main()
