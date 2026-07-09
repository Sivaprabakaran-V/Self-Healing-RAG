import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add app directory to Python path
app_path = str(Path(__file__).resolve().parent.parent / "app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from critic.critic import CriticAgent
from critic.exceptions import ParserError
from critic.parser import StructuredParser
from critic.models import CriticEvaluation

class TestParser(unittest.TestCase):
    def test_parser_to_model_success(self):
        data = {
            "grounded": True,
            "grounded_score": 0.90,
            "relevant": True,
            "relevance_score": 0.90,
            "complete": True,
            "completeness_score": 0.90,
            "hallucination": False,
            "hallucination_score": 0.0,
            "prompt_injection_attempt": False,
            "prompt_injection_success": False,
            "overall_confidence": 0.90,
            "decision": "PASS",
            "failure_reason": "NONE",
            "reason": "Test"
        }
        model = StructuredParser.parse_to_model(data)
        self.assertIsInstance(model, CriticEvaluation)
        self.assertEqual(model.grounded_score, 0.90)

    def test_parser_to_model_invalid_raises_parser_error(self):
        data = {
            "grounded": "invalid-boolean",  # causes pydantic validation error
            "grounded_score": 0.90
        }
        with self.assertRaises(ParserError):
            StructuredParser.parse_to_model(data)

    def test_empty_context_fallback(self):
        mock_llm = MagicMock()
        agent = CriticAgent(mock_llm)
        res = agent.evaluate("Question", "", "Answer")
        
        self.assertEqual(res["decision"], "FAIL")
        self.assertEqual(res["failure_reason"], "EMPTY_CONTEXT")
        self.assertIn("Empty context", res["reason"])

    def test_empty_answer_fallback(self):
        mock_llm = MagicMock()
        agent = CriticAgent(mock_llm)
        res = agent.evaluate("Question", "Context", "")
        
        self.assertEqual(res["decision"], "FAIL")
        self.assertEqual(res["failure_reason"], "INCOMPLETE_ANSWER")
        self.assertIn("Empty answer", res["reason"])

    def test_llm_failure_retry_fallback(self):
        from langchain_core.runnables import Runnable
        
        class MockFailedStructuredLLM(Runnable):
            def invoke(self, *args, **kwargs):
                raise ValueError("API Error")
                
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MockFailedStructuredLLM()
        
        agent = CriticAgent(mock_llm)
        res = agent.evaluate("Question", "Context", "Answer")
        
        self.assertEqual(res["decision"], "FAIL")
        self.assertEqual(res["failure_reason"], "PARSER_ERROR")
        self.assertIn("Critic parsing failed", res["reason"])

if __name__ == "__main__":
    unittest.main()
