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

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from slack_sdk.errors import SlackApiError
from dataclasses import dataclass
from slack_notifier import SlackNotifier, ResearchTopic
from typing import List

from paper import Paper

class TestSlackNotifier(unittest.TestCase):
    def setUp(self):
        self.token = "xoxb-test-token"
        self.default_channel = "#default"
        self.notifier = SlackNotifier(self.token, self.default_channel)
        
        # Sample paper
        self.paper = Paper(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="This is a test abstract",
            url="https://example.com/paper",
            venue="arXiv preprint"
        )
        
        # Sample topics
        self.topic1 = ResearchTopic(
            name="LLM Inference",
            description="Test description",
            keywords=["llm", "inference"],
            slack_user="@user1",
            slack_channel="#llm-papers"
        )
        
        self.topic2 = ResearchTopic(
            name="Serverless",
            description="Test description",
            keywords=["serverless"],
            slack_user="@user2"  # No specific channel
        )

    @patch('slack_sdk.WebClient')
    def test_notify_matches_success(self, mock_client):
        # Setup mock
        mock_client.return_value.chat_postMessage.return_value = {"ok": True}
        self.notifier.client = mock_client.return_value
        
        # Test with multiple topics
        topics = [self.topic1, self.topic2]
        self.notifier.notify_matches(self.paper, topics)
        
        # Check calls to Slack API
        calls = mock_client.return_value.chat_postMessage.call_args_list
        self.assertEqual(len(calls), 2)
        
        # Check first call (topic with specific channel)
        self.assertEqual(calls[0][1]['channel'], '#llm-papers')
        self.assertIn('@user1', calls[0][1]['text'])
        self.assertIn('Test Paper', calls[0][1]['text'])
        
        # Check second call (topic using default channel)
        self.assertEqual(calls[1][1]['channel'], self.default_channel)
        self.assertIn('@user2', calls[1][1]['text'])

    @patch('slack_sdk.WebClient')
    def test_notify_matches_empty_topics(self, mock_client):
        # Should not make any API calls if no topics
        self.notifier.notify_matches(self.paper, [])
        mock_client.return_value.chat_postMessage.assert_not_called()

    @patch('slack_sdk.WebClient')
    def test_notify_matches_api_error(self, mock_client):
        # Setup mock to raise error
        mock_client.return_value.chat_postMessage.side_effect = SlackApiError(
            "Error", {"error": "channel_not_found"}
        )
        self.notifier.client = mock_client.return_value
        
        # Should handle error gracefully
        try:
            self.notifier.notify_matches(self.paper, [self.topic1])
        except Exception as e:
            self.fail(f"Should not raise exception, but raised {e}")

    def test_format_paper_message(self):
        message = self.notifier._format_paper_message(self.paper, [self.topic1])
        
        # Check message formatting
        self.assertIn("*Test Paper*", message)
        self.assertIn("Author One, Author Two", message)
        self.assertIn("arXiv preprint", message)
        self.assertIn("https://example.com/paper", message)
        self.assertIn("This is a test abstract", message)
        self.assertIn("LLM Inference", message)

if __name__ == '__main__':
    unittest.main() 