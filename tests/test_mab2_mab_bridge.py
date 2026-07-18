import unittest

from scripts.eval import mab2_mab_bridge as bridge


class MAB2MABBridgeDetectiveQATest(unittest.TestCase):
    def test_context_prefix_tracks_benchmark_family(self):
        self.assertEqual(
            bridge.context_prefix_for_sub_dataset("detective_qa"),
            "long-range-understanding",
        )
        self.assertEqual(
            bridge.context_prefix_for_sub_dataset("factconsolidation_sh_6k"),
            "conflict-resolution",
        )
        self.assertEqual(
            bridge.context_prefix_for_sub_dataset("ruler_qa2_421K"),
            "accurate-retrieval",
        )

    def test_detectiveqa_extracts_json_answer_field(self):
        prediction = (
            '{"answer":"B. He is not satisfied with this job.", '
            '"reasoning":"Based on the context"}'
        )

        self.assertEqual(
            bridge.extract_detectiveqa_answer(prediction),
            "B. He is not satisfied with this job.",
        )

    def test_detectiveqa_extracts_boxed_answer(self):
        prediction = "【B. Mr. and Mrs. MacNamara】\nThe answer is B. Mr. and Mrs. MacNamara."

        self.assertEqual(
            bridge.extract_detectiveqa_answer(prediction),
            "B. Mr. and Mrs. MacNamara",
        )

    def test_detectiveqa_extracts_last_option_when_prompt_prefix_leaks(self):
        prediction = (
            "条件\n\n[story text]\n\nQuestion: Where is the evening tea going?\n\n"
            "A. Abandoned and left behind"
        )

        self.assertEqual(
            bridge.extract_detectiveqa_answer(prediction),
            "A. Abandoned and left behind",
        )

    def test_build_detectiveqa_queries_keeps_all_questions(self):
        row = {
            "questions": ["Q1", "Q2"],
            "answers": [["A1"], ["A2"]],
        }

        def get_template(sub_dataset, template_type, agent_type):
            self.assertEqual(sub_dataset, "detective_qa")
            self.assertEqual(template_type, "query")
            self.assertEqual(agent_type, "Long_context_agent")
            return "Prompt: {question}"

        queries = bridge.build_detectiveqa_queries(
            row,
            sub_dataset="detective_qa",
            get_template=get_template,
        )

        self.assertEqual(
            queries,
            [
                {"query_id": 0, "question": "Q1", "query_prompt": "Prompt: Q1", "gold_answers": ["A1"]},
                {"query_id": 1, "question": "Q2", "query_prompt": "Prompt: Q2", "gold_answers": ["A2"]},
            ],
        )


if __name__ == "__main__":
    unittest.main()
