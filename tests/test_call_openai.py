import unittest

from tokreate import CallAction


class TestModels(unittest.TestCase):
    def test_3_5_call(self):
        greetings = CallAction(
            prompt="Hello, my name is {{ name }}; what's yours?",
            model="gpt-3.5-turbo",
            parameters=dict(temperature=1.0)
        )

        # tokreate always returns a history of the conversation
        history = greetings.run(name="Alice")

        assert history[0].content == "Hello, my name is Alice; what's yours?"
        assert "GPT" in history[1].content
