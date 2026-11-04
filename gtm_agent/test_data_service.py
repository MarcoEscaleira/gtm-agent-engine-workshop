import json
import os
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.object(data_service, "PROSPECTS", deepcopy(data_service.PROSPECTS)))
        self.enterContext(patch.object(data_service, "_PROFILES", {}))
        self.prospect_id = "LEAD-71001"
        self.offering = data_service.get_offering("OFFER-10005")

    def test_update_then_score_uses_saved_technology(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Kafka", profile["prospect_profile"]["tech_stack"])

        result = gtm_agent.update_prospect_info.invoke({
            "prospect_id": self.prospect_id, "technology": "Kafka",
        })

        saved_stack = data_service.fetch_tech_stack(self.prospect_id)
        self.assertEqual(result, {
            "prospect_id": self.prospect_id,
            "technology": "Kafka",
            "updated": True,
            "found": True,
            "tech_stack": saved_stack,
        })
        self.assertIn("Kafka", saved_stack)
        self.assertIn("Kafka", data_service.get_prospect_record(self.prospect_id)["tech_stack"])
        self.assertIsNone(data_service.get_profile_from_db(self.prospect_id)["prospect_profile"])
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertIn("Kafka", rebuilt["prospect_profile"]["tech_stack"])
        cached = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(cached, rebuilt)
        self.assertIn("Kafka", data_service.get_profile_from_db(self.prospect_id)["prospect_profile"]["tech_stack"])

        def score_saved_stack(messages):
            supplied = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
            self.assertEqual(supplied["tech_stack"], saved_stack)
            required = self.offering["required_tech_stack"]
            missing = [tech for tech in required if tech not in supplied["tech_stack"]]
            return gtm_agent.ProspectScore(
                score=100 if not missing else 75,
                justification="Missing technologies: " + (", ".join(missing) or "none"),
                rubric_breakdown=gtm_agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100 if not missing else 75,
                    segment_fit=100,
                ),
            )

        model = Mock(invoke=score_saved_stack)
        with patch.object(gtm_agent, "_scoring_llm", model):
            score = gtm_agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"], "offering": self.offering,
            })
        self.assertEqual(score["justification"], "Missing technologies: none")
        self.assertEqual(score["score"], 100)

    def test_duplicate_technology_is_not_added_twice(self):
        data_service.update_prospect_info(self.prospect_id, "Kafka")
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")
        self.assertTrue(result["updated"])
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id).count("Kafka"), 1)

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(data_service.update_prospect_info("unknown", "Kafka"), {
            "updated": False, "found": False,
        })
        self.assertIsNone(data_service.get_prospect_record("unknown"))

    def test_write_failure_does_not_report_success(self):
        class UnwritableRecord(dict):
            def __setitem__(self, key, value):
                raise OSError("Record store is unavailable")

        original = deepcopy(data_service.get_prospect_record(self.prospect_id))
        data_service.PROSPECTS[self.prospect_id] = UnwritableRecord(original)
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        with self.assertRaisesRegex(OSError, "Record store is unavailable"):
            gtm_agent.update_prospect_info.invoke({
                "prospect_id": self.prospect_id, "technology": "Kafka",
            })
        self.assertEqual(data_service.get_prospect_record(self.prospect_id), original)
        self.assertEqual(
            data_service.get_profile_from_db(self.prospect_id)["prospect_profile"],
            profile["prospect_profile"],
        )

    def test_score_without_update_uses_saved_stack(self):
        saved_stack = list(data_service.fetch_tech_stack(self.prospect_id))
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        supplied = {**profile["prospect_profile"], "tech_stack": ["Kafka"]}
        expected = gtm_agent.ProspectScore(
            score=75, justification="Missing Kafka",
            rubric_breakdown=gtm_agent.RubricBreakdown(
                revenue_fit=100, tech_stack_match=75, segment_fit=100,
            ),
        )
        with patch.object(gtm_agent, "_scoring_llm") as model:
            model.invoke.return_value = expected
            result = gtm_agent.score_prospect.invoke({
                "prospect_profile": supplied, "offering": self.offering,
            })
        messages = model.invoke.call_args.args[0]
        scored = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertEqual(scored["tech_stack"], saved_stack)
        self.assertNotIn("Kafka", scored["tech_stack"])
        self.assertEqual(result, expected.model_dump())


if __name__ == "__main__":
    unittest.main()
