import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_test"}})
        self.prospect = {"prospect_id": "LEAD-test", "name": "Prospect", "email": "model@example.com"}
        self.rep = {"name": "Rep", "email": "rep@example.com"}

    def send(self):
        return agent.send_prospect_email.func(
            self.prospect, "Introduction", "Hello", self.runtime
        )

    def test_disqualified_or_missing_record_is_blocked_before_send(self):
        for record in (None, {"disqualified": True, "email": "crm@example.com"}):
            with self.subTest(record=record):
                with patch.object(agent.data_service, "get_prospect_record", return_value=record):
                    with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Send path reached")):
                        self.assertEqual(self.send(), {
                            "status": "blocked",
                            "error": "Prospect is disqualified; email not sent.",
                        })

    def test_eligible_record_uses_authoritative_email_and_preserves_receipt(self):
        record = {"disqualified": False, "email": "crm@example.com"}
        with patch.object(agent.data_service, "get_prospect_record", return_value=record):
            with patch.object(agent.data_service, "get_rep", return_value=self.rep):
                result = self.send()
        self.assertEqual(result.pop("status"), "sent")
        self.assertRegex(result.pop("message_id"), r"^msg-[0-9a-f]{12}$")
        self.assertEqual(result, {
            "to": "crm@example.com", "to_name": "Prospect",
            "from": "rep@example.com", "from_name": "Rep",
            "subject": "Introduction", "body": "Hello",
        })

    def test_authoritative_record_without_email_fails(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value={"disqualified": False}):
            with patch.object(agent.data_service, "get_rep", return_value=self.rep):
                self.assertEqual(self.send(), {
                    "status": "failed", "error": "Prospect record has no email address.",
                })

    def test_explicit_sender_is_preserved_for_eligible_prospect(self):
        record = {"email": "crm@example.com"}
        with patch.object(agent.data_service, "get_prospect_record", return_value=record):
            result = agent.send_prospect_email.func(
                self.prospect, "Introduction", "Hello", self.runtime, from_rep=self.rep
            )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual(result["to"], record["email"])


if __name__ == "__main__":
    unittest.main()
