"""Use the actual local official scorer; no model or inference fixtures."""
import unittest

from scripts.eval.review_eventqa_decomposition import (
    candidate_info, candidates, check_identity, evaluate, fraction,
    load_official, paired, summarize,
)


class DecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = load_official()

    def info(self, values=None, gold=None):
        values = values or ["Gold event.", "Wrong event."]
        return candidate_info("Below is a list of possible subsequent events:\n" + repr(values) + "\nChoose one.", gold or ["Gold event."], self.official)

    def row(self, prediction, flags=None, info=None, gold=None):
        return evaluate(prediction, gold or ["Gold event."], info or self.info(), flags or {}, self.official)

    def test_answer_prefix_is_parsed_despite_flag(self):
        r = self.row("Answer: Gold event.", {"contains_answer_prefix": True})
        self.assertTrue(r["candidate_valid"] and r["strict_candidate_correct"] and r["heuristic_flag"])

    def test_multiline_gold_in_later_line(self):
        r = self.row("Wrong event.\nGold event.", {"multiline_output": True})
        self.assertEqual(r["official_em"], 0)
        self.assertEqual(r["official_raw_recall"], 1)
        self.assertEqual(r["category"], "gold_mentioned_wrong_valid")

    def test_empty_has_undefined_conditional_accuracy(self):
        r = self.row("")
        self.assertTrue(r["parser_empty"])
        self.assertFalse(r["parser_none"])
        self.assertIsNone(summarize([r])["parsed_conditional_em"]["value"])

    def test_braces_can_be_correct_after_official_normalization(self):
        r = self.row('{"Answer": "Gold event."}', {"contains_json_brace": True})
        self.assertTrue(r["heuristic_flag"])
        self.assertEqual(r["official_raw_recall"], 1)

    def test_long_valid_candidate_is_not_parser_failure(self):
        gold = " ".join("word" + str(i) for i in range(30))
        r = self.row(gold, {"verbose_output": True}, self.info([gold, "Wrong event."], [gold]), [gold])
        self.assertTrue(r["strict_candidate_correct"] and r["parsed_nonempty"] and r["heuristic_flag"])

    def test_wrong_candidate_without_gold_is_gold_absent(self):
        r = self.row("Wrong event.")
        self.assertTrue(r["candidate_valid"])
        self.assertFalse(r["strict_candidate_correct"])
        self.assertEqual(r["category"], "gold_absent")

    def test_partial_and_multiple_candidates_not_valid(self):
        self.assertFalse(self.row("Gold")["candidate_valid"])
        r = self.row("Gold event. Wrong event.")
        self.assertFalse(r["candidate_valid"])
        self.assertEqual(r["official_em"], 1)
        self.assertEqual(r["category"], "gold_mentioned_no_valid")

    def test_safe_list_extraction_brackets_and_malicious_expression(self):
        self.assertEqual(candidates("Below is a list of possible subsequent events:\n['x[y]', 'z']\nChoose."), ["x[y]", "z"])
        with self.assertRaises(ValueError):
            candidates("Below is a list of possible subsequent events:\n[__import__('os')]\nChoose.")

    def test_duplicate_collision_missing_gold_and_multi_gold(self):
        i = self.info(["Gold event.", "Gold event.", "Wrong event."])
        self.assertEqual(i["duplicate_count"], 1)
        self.assertEqual(i["chance"], .5)
        i = self.info(["The gold event.", "Gold event."])
        self.assertIn("normalization_collision", i["errors"])
        self.assertIsNone(i["chance"])
        self.assertFalse(self.row("Gold event.", info=i)["candidate_valid"])
        self.assertIn("gold_not_covered", self.info(["Wrong event."])["errors"])
        i = self.info(gold=["Gold event.", "Wrong event."])
        self.assertEqual(i["chance"], 1.)
        r = self.row("Gold event.", info=i, gold=["Gold event.", "Wrong event."])
        self.assertEqual(r["official_raw_recall"], 0)
        self.assertTrue(r["raw_gold_any"] and r["strict_candidate_correct"])

    def test_identity_mismatch_and_positional_only_fail(self):
        c = {"context_id": "a", "qa_pair_id": "q", "query_sha256": "h"}
        for raw in [{"context_id": "b"}, {"qa_pair_id": "x"}, {"query_sha256": "x"}, {}]:
            with self.assertRaises(ValueError):
                check_identity(raw, {}, c, False)

    def test_paired_shared_subset_and_product_decomposition(self):
        b, t = [self.row("Gold event."), self.row("Wrong event.")], [self.row(""), self.row("Wrong event.")]
        for group in [b, t]:
            for i, r in enumerate(group):
                r["key"] = (0, i)
        p = paired(b, t)
        self.assertEqual(p["all"]["strict_candidate_correct"]["wins"], 1)
        self.assertEqual(p["both_candidate_valid"]["strict_candidate_correct"]["n"], 1)
        d = p["em_product_decomposition"]
        self.assertAlmostEqual(d["delta_em"], d["parsed_rate_term"] + d["conditional_em_term"])
        self.assertIsNone(fraction(0, 0)["value"])


if __name__ == "__main__":
    unittest.main()
