import json
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langsmith import trace, tracing_context

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service, gtm_agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-12853"
        self.record = deepcopy(data_service.PROSPECTS[self.prospect_id])
        self.cache = patch.dict(data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = tracing_context(enabled="local", client=Mock())
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)

    def assert_no_billing(self, value):
        serialized = json.dumps(value, default=str)
        self.assertNotIn("billing_qualification", serialized)
        for sensitive_value in self.record["billing_qualification"].values():
            self.assertNotIn(sensitive_value, serialized)

    def assert_clean_trace(self, run):
        self.assert_no_billing(run.inputs)
        self.assert_no_billing(run.outputs)
        for child in run.child_runs:
            self.assert_clean_trace(child)

    def test_contact_lookup_and_email_trace(self):
        for prospect_id, record in data_service.PROSPECTS.items():
            with self.subTest(prospect_id=prospect_id):
                with trace("email") as run:
                    result = gtm_agent.get_prospect.invoke({"prospect_id": prospect_id})
                    contact = result["prospect"]
                    sent = gtm_agent.send_prospect_email.func(
                        contact, "Demo", "Schedule a demo", None,
                        from_rep={"name": "Rep", "email": "rep@example.com"},
                    )
                    run.end(outputs={"contact": result, "sent": sent})
                self.assertEqual(result, {
                    "prospect": {
                        "prospect_id": prospect_id,
                        **{k: record[k] for k in ("name", "email", "disqualified")},
                    },
                    "found": True,
                })
                self.assertEqual(sent["to"], record["email"])
                self.assert_clean_trace(run)

    def test_new_profile_is_clean_before_cache_save(self):
        with trace("build_profile") as run:
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            run.end(outputs=result)
        expected = {k: v for k, v in self.record.items() if k != "billing_qualification"}
        expected["prospect_id"] = self.prospect_id
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertEqual(data_service._PROFILES[self.prospect_id], expected)
        self.assert_clean_trace(run)
        self.assertIn("save_profile_to_db", [child.name for child in run.child_runs[0].child_runs])
        self.assertEqual(data_service.PROSPECTS[self.prospect_id], self.record)

    def test_legacy_cached_profile_and_scoring_trace(self):
        legacy = {"prospect_id": self.prospect_id, **self.record}
        data_service._PROFILES[self.prospect_id] = legacy
        captured = []
        with trace("cached_score") as run:
            cached = data_service.get_profile_from_db(self.prospect_id)
            profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            with patch.object(gtm_agent, "_scoring_llm") as model:
                model.invoke.side_effect = lambda messages: (
                    captured.append(messages) or SimpleNamespace(model_dump=lambda: {"score": 80})
                )
                gtm_agent.score_prospect.invoke({
                    "prospect_profile": profile["prospect_profile"],
                    "offering": data_service.OFFERINGS["OFFER-10001"],
                })
            run.end(outputs={"cached": cached, "profile": profile})
        self.assert_no_billing(cached)
        self.assertEqual(cached["prospect_profile"], profile["prospect_profile"])
        self.assertEqual(cached["prospect_profile"]["email"], self.record["email"])
        self.assert_clean_trace(run)
        self.assertIn("get_profile_from_db", [child.name for child in run.child_runs])
        self.assert_no_billing(captured)
        self.assertNotIn(self.record["email"], json.dumps(captured))

    def test_scoring_filters_supplied_profile_and_preserves_scoring_data(self):
        for with_id in (True, False):
            with self.subTest(with_id=with_id):
                profile = deepcopy(self.record)
                if with_id:
                    profile.update(prospect_id=self.prospect_id, tech_stack=["stale"])
                original = deepcopy(profile)
                with patch.object(gtm_agent, "_scoring_llm") as model:
                    model.invoke.return_value.model_dump.return_value = {"score": 80}
                    result = gtm_agent.score_prospect.invoke({
                        "prospect_profile": profile,
                        "offering": data_service.OFFERINGS["OFFER-10001"],
                    })
                messages = model.invoke.call_args.args[0]
                scoring_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
                self.assert_no_billing(scoring_profile)
                self.assertNotIn("email", scoring_profile)
                self.assertEqual(scoring_profile["annual_revenue"], self.record["annual_revenue"])
                self.assertEqual(scoring_profile["tech_stack"], self.record["tech_stack"])
                self.assertEqual(profile, original)
                self.assertEqual(result, {"score": 80})

    def test_missing_prospect(self):
        self.assertEqual(gtm_agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(gtm_agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})


if __name__ == "__main__":
    unittest.main()
