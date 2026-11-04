import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class AgentTests(unittest.TestCase):
    def test_application_tools_and_runtime(self):
        calls = []
        responses = iter([
            AIMessage(content="", tool_calls=[{
                "name": "get_current_rep", "args": {}, "id": "rep",
            }]),
            AIMessage(content="", tool_calls=[{
                "name": "send_prospect_email",
                "args": {
                    "prospect": {"name": "Prospect", "email": "prospect@example.com"},
                    "subject": "Demo next week", "body": "Can we schedule a demo?",
                },
                "id": "email",
            }]),
            AIMessage(content="Email sent."),
        ])

        def generate(model, messages, **kwargs):
            calls.append((messages, kwargs))
            return ChatResult(generations=[ChatGeneration(message=next(responses))])

        with patch.object(ChatOpenAI, "_generate", generate), patch.dict(
            os.environ, {"LANGSMITH_TRACING": "false"}
        ):
            result = agent.run_agent("Email the prospect.", user_id="rep_amills")

        self.assertEqual(result["reply"], "Email sent.")
        expected = {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        }
        for messages, kwargs in calls:
            self.assertEqual({t["function"]["name"] for t in kwargs["tools"]}, expected)
            self.assertEqual(messages[0].content, agent.SYSTEM_PROMPT)
            self.assertNotIn("cache_control", json.dumps(kwargs))
            for tool in kwargs["tools"]:
                self.assertNotIn("runtime", tool["function"]["parameters"]["properties"])
        outputs = {
            message.name: json.loads(message.content)
            for message in result["messages"] if isinstance(message, ToolMessage)
        }
        rep = agent.data_service.get_rep("rep_amills")
        self.assertEqual(outputs["get_current_rep"], {"rep": rep, "found": True})
        self.assertEqual(outputs["send_prospect_email"]["status"], "sent")
        self.assertEqual(outputs["send_prospect_email"]["from"], rep["email"])

    def test_run_configuration_is_preserved(self):
        with patch.object(agent.gtm_agent, "invoke", return_value={
            "messages": [AIMessage(content="Done")],
        }) as invoke:
            result = agent.run_agent(
                "Email LEAD-12853", user_id="rep_amills",
                environment="test", thread_id="thread-123",
            )
        args, kwargs = invoke.call_args
        self.assertEqual(args, ({"messages": [{"role": "user", "content": "Email LEAD-12853"}]},))
        self.assertEqual(kwargs["config"], {
            "run_name": "GTM Assistant",
            "run_id": agent.uuid.UUID(result["run_id"]),
            "metadata": {
                "thread_id": "thread-123", "user_id": "rep_amills",
                "environment": "test", "request_intent": "send_email",
            },
        })


if __name__ == "__main__":
    unittest.main()
