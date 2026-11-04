import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ProspectToolsTest(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})
        self.cache = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_email_resolves_contact_and_runtime_sender(self):
        result = agent.send_prospect_email.func(
            prospect_id="LEAD-12853", subject="Demo", body="Next week?", runtime=self.runtime,
        )
        contact = agent.data_service.get_prospect_record("LEAD-12853")
        sender = agent.data_service.get_rep("rep_amills")
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], contact["email"])
        self.assertEqual(result["to_name"], contact["name"])
        self.assertEqual(result["from"], sender["email"])
        self.assertEqual(result["from_name"], sender["name"])
        self.assertEqual(result["subject"], "Demo")
        self.assertEqual(result["body"], "Next week?")

    def test_email_preserves_explicit_sender(self):
        result = agent.send_prospect_email.func(
            "LEAD-12853", "Demo", "Hello", self.runtime,
            from_rep={"name": "Sender", "email": "sender@example.com"},
        )
        self.assertEqual(result["from"], "sender@example.com")

    def test_email_missing_contact_or_email_does_not_send(self):
        for contact, error in ((None, "not found"), ({"name": "No email"}, "no email")):
            with self.subTest(contact=contact), \
                    patch.object(agent.data_service, "get_prospect_record", return_value=contact), \
                    patch.object(agent.uuid, "uuid4") as message_id:
                result = agent.send_prospect_email.func("unknown", "Demo", "Hello", self.runtime)
                self.assertEqual(result["status"], "failed")
                self.assertIn(error, result["error"])
                self.assertNotIn("message_id", result)
                message_id.assert_not_called()

    def test_score_loads_records_and_preserves_structured_result(self):
        expected = agent.ProspectScore(
            score=70, justification="Partial fit",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=60, segment_fit=70,
            ),
        )
        with patch.object(agent, "_scoring_llm") as scoring:
            scoring.invoke.return_value = expected
            result = agent.score_prospect.invoke({
                "prospect_id": "LEAD-71001", "offering_id": "OFFER-10005",
            })
        self.assertEqual(result, expected.model_dump())
        payload = scoring.invoke.call_args.args[0][1]["content"]
        offering_text, profile_text = payload.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering_text), agent.data_service.get_offering("OFFER-10005"))
        self.assertEqual(json.loads(profile_text), agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})["prospect_profile"])

    def test_scoring_refreshes_cached_tech_stack(self):
        agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["Fresh technology"]), \
                patch.object(agent, "_scoring_llm") as scoring:
            agent.score_prospect.invoke({"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"})
        payload = scoring.invoke.call_args.args[0][1]["content"]
        profile = json.loads(payload.split("\n\nProspect profile:\n")[1])
        self.assertEqual(profile["tech_stack"], ["Fresh technology"])

    def test_agent_can_send_directly_or_resolve_identity_for_content(self):
        for needs_identity in (False, True):
            with self.subTest(needs_identity=needs_identity):
                responses = []
                if needs_identity:
                    responses.append(AIMessage(content="", tool_calls=[{
                        "name": "get_current_rep", "args": {}, "id": "rep",
                    }]))
                responses.extend([
                    AIMessage(content="", tool_calls=[{
                        "name": "send_prospect_email", "id": "send",
                        "args": {"prospect_id": "LEAD-12853", "subject": "Demo", "body": "Hello"},
                    }]),
                    AIMessage(content="Sent."),
                ])
                graph = agent.create_deep_agent(
                    model=ToolCallingModel(responses=responses),
                    tools=[agent.send_prospect_email, agent.get_current_rep],
                    system_prompt=agent.SYSTEM_PROMPT,
                )
                with patch.object(agent, "gtm_agent", graph):
                    result = agent.run_agent("Email LEAD-12853", user_id="rep_amills")
                tools = [message for message in result["messages"] if message.type == "tool"]
                self.assertEqual([message.name for message in tools],
                                 ["get_current_rep", "send_prospect_email"] if needs_identity else ["send_prospect_email"])
                sent = json.loads(tools[-1].content)
                self.assertEqual(sent["status"], "sent")
                self.assertEqual(sent["from"], agent.data_service.get_rep("rep_amills")["email"])
                self.assertEqual(result["reply"], "Sent.")

    def test_invalid_scoring_inputs_do_not_invoke_scoring(self):
        cases = [
            ("unknown", "OFFER-10005", "not found"),
            ("LEAD-71001", "unknown", "valid offering"),
        ]
        with patch.object(agent, "_scoring_llm") as scoring:
            for prospect_id, offering_id, error in cases:
                with self.subTest(prospect_id=prospect_id, offering_id=offering_id):
                    result = agent.score_prospect.invoke({"prospect_id": prospect_id, "offering_id": offering_id})
                    self.assertIsNone(result["score"])
                    self.assertIn(error, result["error"])
            for field in ("required_tech_stack", "min_annual_revenue", "description"):
                invalid = dict(agent.data_service.get_offering("OFFER-10005"))
                invalid.pop(field)
                with self.subTest(field=field), \
                        patch.object(agent.data_service, "get_offering", return_value=invalid):
                    result = agent.score_prospect.invoke({"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"})
                    self.assertIsNone(result["score"])
                    self.assertIn("valid offering", result["error"])
            scoring.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
