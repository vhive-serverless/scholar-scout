"""
MIT License

Copyright (c) 2024 Dmitrii Ustiugov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import os

from slack_sdk.errors import SlackApiError

from paper import Paper
from slack_notifier import ResearchTopic, SlackNotifier
from config import load_config
from dotenv import load_dotenv

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class TestSlackNotifier(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Load test configuration
        env_path = os.path.join(os.path.dirname(__file__), ".env.test")
        if not load_dotenv(env_path, override=True):
            # For CI environment, ensure required env vars are set
            required_vars = ["GMAIL_USERNAME", "GMAIL_APP_PASSWORD", "PPLX_API_KEY"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise RuntimeError(f"Missing required environment variables: {missing_vars}")
        config_path = os.path.join(os.path.dirname(__file__), "test_config.yml")
        self.config = load_config(config_path)
        self.token = self.config["slack"]["api_token"]
        self.default_channel = self.config["slack"]["default_channel"]
        self.notifier = SlackNotifier(self.token, self.default_channel, self.config)

        # Sample paper
        self.paper = Paper(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="This is a test abstract",
            url="https://example.com/paper",
            venue="arXiv preprint",
        )

        # Sample topics from config
        self.topic1 = ResearchTopic(
            name=self.config["research_topics"][0]["name"],
            description=self.config["research_topics"][0]["description"],
            keywords=self.config["research_topics"][0]["keywords"],
            slack_users=self.config["research_topics"][0]["slack_users"],
            slack_channel=self.config["research_topics"][0]["slack_channel"],
        )

        self.topic2 = ResearchTopic(
            name=self.config["research_topics"][1]["name"],
            description=self.config["research_topics"][1]["description"],
            keywords=self.config["research_topics"][1]["keywords"],
            slack_users=self.config["research_topics"][1]["slack_users"],
            slack_channel=self.config["research_topics"][1]["slack_channel"],
        )

    @patch("slack_sdk.WebClient")
    def test_notify_matches_success(self, mock_client):
        # Setup mock
        mock_client.return_value.chat_postMessage.return_value = {"ok": True}
        self.notifier.client = mock_client.return_value

        # Test with multiple topics
        topics = [self.topic1, self.topic2]
        # Create a list of paper-topic pairs
        paper_results = [(self.paper, topics)]
        self.notifier.notify_matches(paper_results)

        # Check calls to Slack API
        calls = mock_client.return_value.chat_postMessage.call_args_list
        self.assertEqual(len(calls), 2)

        # Check first call (topic with specific channel)
        self.assertEqual(calls[0][1]["channel"], "#scholar-scout-llm")
        self.assertIn("@test1", calls[0][1]["text"])
        self.assertIn("Test Paper", calls[0][1]["text"])

        # Check second call (topic using default channel)
        self.assertEqual(calls[1][1]["channel"], "#scholar-scout-serverless")
        self.assertIn("@test3", calls[1][1]["text"])

    @patch("slack_sdk.WebClient")
    def test_notify_matches_empty_topics(self, mock_client):
        # Should not make any API calls if no topics
        self.notifier.notify_matches([])
        mock_client.return_value.chat_postMessage.assert_not_called()

    @patch("slack_sdk.WebClient")
    def test_notify_matches_api_error(self, mock_client):
        # Setup mock to raise error
        mock_client.return_value.chat_postMessage.side_effect = SlackApiError(
            "Error", {"error": "channel_not_found"}
        )
        self.notifier.client = mock_client.return_value

        # Should handle error gracefully
        try:
            paper_results = [(self.paper, [self.topic1])]
            self.notifier.notify_matches(paper_results)
        except Exception as e:
            self.fail(f"Should not raise exception, but raised {e}")

    def test_format_paper_message(self):
        message = self.notifier._format_paper_message(self.paper, self.topic1)

        # Check message formatting
        self.assertIn("*New paper matching topic: LLM Inference*", message)
        self.assertIn("*Title:* Test Paper", message)
        self.assertIn("*Authors:* Author One, Author Two", message)
        self.assertIn("*Venue:* arXiv preprint", message)
        self.assertIn("*URL:* https://example.com/paper", message)
        self.assertIn("*Abstract:* This is a test abstract", message)


if __name__ == "__main__":
    unittest.main()
