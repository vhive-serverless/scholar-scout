"""
MIT License

Copyright (c) 2024 Dmitrii Ustiugov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import unittest
from datetime import datetime
from unittest import TestCase, mock

from scholar_classifier import ScholarClassifier
from paper import Paper
from config import ResearchTopic


def test_extract_and_classify_papers(mocker):
    """Test the paper extraction and classification functionality using pytest-mock."""
    # Set up mock email message with multipart content
    email_message = mocker.Mock()
    email_message.is_multipart.return_value = True

    # Configure mock to return subject when accessed like a dictionary
    email_message.__getitem__ = mocker.Mock(
        side_effect=lambda x: {"subject": "New articles in your Google Scholar alert"}.get(x)
    )

    # Create sample HTML content simulating a Google Scholar alert
    html_content = """
    <div>
        <h3>Title: Efficient LLM Inference on Serverless Platforms</h3>
        <div>Authors: John Doe, Jane Smith</div>
        <div>Conference: SOSP 2024</div>
        <div>Abstract: This paper presents novel techniques for optimizing LLM inference in serverless environments.</div>
        <a href="http://example.com/paper">Link to paper</a>
    </div>
    """

    # Mock email part containing HTML content
    email_part = mocker.Mock()
    email_part.get_content_type.return_value = "text/html"
    email_part.get_payload.return_value = html_content.encode()

    # Set up email structure
    email_message.walk.return_value = [email_part]

    # Create mock response from Perplexity API
    mock_response = mocker.Mock()
    mock_response.choices = [
        mocker.Mock(
            message=mocker.Mock(
                # Simulated API response with paper classification
                content="""[{
                    "title": "Efficient LLM Inference on Serverless Platforms",
                    "authors": ["John Doe", "Jane Smith"],
                    "venue": "SOSP 2024",
                    "url": "http://example.com/paper",
                    "abstract": "This paper presents novel techniques for optimizing LLM inference in serverless environments.",
                    "relevant_topics": ["LLM Inference", "Serverless Computing"]
                }]"""
            )
        )
    ]

    # Define test configuration with research topics
    config = {
        "email": {"username": "test@example.com", "password": "test"},
        "slack": {"api_token": "test-token", "default_channel": "#scholar-scout-default"},
        "perplexity": {"api_key": "test-key"},
        "research_topics": [
            {
                "name": "LLM Inference",
                "keywords": ["llm", "inference"],
                "slack_user": "@test1",
                "slack_channel": "#scholar-scout-test",
                "description": "LLM inference research",
            },
            {
                "name": "Serverless Computing",
                "keywords": ["serverless"],
                "slack_user": "@test2",
                "slack_channel": "#scholar-scout-test",
                "description": "Serverless computing research",
            },
        ],
    }

    # Mock the OpenAI/Perplexity API call
    mocker.patch("openai.OpenAI.chat.completions.create", return_value=mock_response)

    # Initialize classifier with test config
    classifier = ScholarClassifier(config_dict=config)

    # Execute the method being tested
    results = classifier.extract_and_classify_papers(email_message)

    # Verify results
    assert len(results) == 1, "Should extract exactly one paper"
    paper, topics = results[0]

    # Verify paper details
    assert paper.title == "Efficient LLM Inference on Serverless Platforms"
    assert paper.authors == ["John Doe", "Jane Smith"]
    assert paper.venue == "SOSP 2024"
    assert paper.url == "http://example.com/paper"
    assert "LLM inference" in paper.abstract.lower()

    # Verify topic classification
    assert len(topics) == 2, "Should identify two relevant topics"
    topic_names = [t.name for t in topics]
    assert "LLM Inference" in topic_names
    assert "Serverless Computing" in topic_names


class TestScholarClassifier(TestCase):
    """Test suite for ScholarClassifier using unittest framework."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create test configuration with mock credentials and topics
        self.config = {
            "email": {"username": "test@example.com", "password": "test"},
            "slack": {"api_token": "test-token", "default_channel": "#scholar-scout-default"},
            "perplexity": {"api_key": "test-key"},
            "research_topics": [
                {
                    "name": "LLM Inference",
                    "keywords": ["llm", "inference"],
                    "slack_user": "@test1",
                    "description": "LLM inference research",
                    "slack_channel": "#scholar-scout-test",
                },
                {
                    "name": "Serverless Computing",
                    "keywords": ["serverless"],
                    "slack_user": "@test2",
                    "description": "Serverless computing research",
                    "slack_channel": "#scholar-scout-test",
                },
            ],
        }

    @mock.patch("scholar_classifier.OpenAI")
    def test_extract_and_classify_papers(self, mock_openai_class):
        """Test paper extraction and classification with unittest mocks."""
        # Set up mock email with Google Scholar format
        email_message = mock.Mock()
        email_message.is_multipart.return_value = True
        email_message.__getitem__ = mock.Mock(
            side_effect=lambda x: (
                "New articles in your Google Scholar alert" if x == "subject" else None
            )
        )

        # Create HTML content matching Google Scholar format
        html_content = """
        <div class="gs_r gs_or gs_scl">
            <h3>
                <a class="gse_alrt_title" href="http://example.com/paper">Efficient LLM Inference on Serverless Platforms</a>
            </h3>
            <div>John Doe, Jane Smith</div>
            <div>This paper presents novel techniques for optimizing LLM inference in serverless environments.</div>
        </div>
        """

        # Mock email part with proper encoding handling
        email_part = mock.Mock()
        email_part.get_content_type.return_value = "text/html"
        email_part.get_payload.return_value = html_content.encode("utf-8")
        email_part.get_payload.side_effect = lambda decode: (
            html_content.encode("utf-8") if decode else html_content
        )

        email_message.walk.return_value = [email_part]

        # Set up OpenAI client mock
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client

        # Create mock API response
        mock_response = mock.Mock()
        mock_response.choices = [
            mock.Mock(
                message=mock.Mock(
                    content="""{
                        "title": "Efficient LLM Inference on Serverless Platforms",
                        "authors": ["John Doe", "Jane Smith"],
                        "venue": "SOSP 2024",
                        "link": "http://example.com/paper",
                        "abstract": "This paper presents novel techniques for optimizing LLM inference in serverless environments.",
                        "relevant_topics": ["LLM Inference", "Serverless Computing"]
                    }"""
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Execute test
        classifier = ScholarClassifier(config_dict=self.config)
        results = classifier.extract_and_classify_papers(email_message)

        # Verify results
        assert len(results) == 1, "Should extract exactly one paper"
        paper, topics = results[0]

        # Check paper metadata
        assert paper.title == "Efficient LLM Inference on Serverless Platforms"
        assert "John Doe" in paper.authors[0]
        assert paper.url == "http://example.com/paper"
        assert "llm inference" in paper.abstract.lower()

        # Verify topic classification
        assert len(topics) == 2, "Should identify two relevant topics"
        topic_names = [t.name for t in topics]
        assert "LLM Inference" in topic_names
        assert "Serverless Computing" in topic_names

    @mock.patch("scholar_classifier.datetime")
    def test_weekly_update_notification(self, mock_datetime):
        """Test weekly update notification functionality."""
        # Mock current time to a fixed date for testing
        mock_now = datetime(2024, 3, 15, 12, 0)
        mock_last_update = datetime(2024, 3, 7, 12, 0)  # 8 days ago
        mock_datetime.now.return_value = mock_now
        mock_datetime.min = datetime.min

        # Create temporary file to simulate last update timestamp
        with open(".last_weekly_update", "w") as f:
            f.write(mock_last_update.isoformat())

        try:
            # Set up test environment
            mock_slack_notifier = mock.Mock()
            classifier = ScholarClassifier(config_dict=self.config)
            classifier.slack_notifier = mock_slack_notifier

            # Add some test papers to processed_papers
            test_paper = Paper(
                title="Test LLM Paper",
                authors=["Test Author"],
                abstract="Test abstract",
                venue="Test Venue",
                url="http://test.com",
            )
            test_topic = ResearchTopic(
                name="LLM Inference",
                keywords=["llm"],
                slack_user="@test",
                description="Test description",
                slack_channel="#test",
            )
            classifier.processed_papers = [(test_paper, [test_topic])]

            # Execute the method being tested
            classifier.send_weekly_update_notification()

            # Verify notifications were sent to correct channels
            actual_calls = [
                call.kwargs["channel"] for call in mock_slack_notifier.send_message.call_args_list
            ]

            # Assert correct channels were notified
            expected_channels = ["#systems-for-ai", "#serverless-systems"]
            assert set(actual_calls) == set(expected_channels), "Should notify correct channels"

            # Verify message content for each call
            for call in mock_slack_notifier.send_message.call_args_list:
                kwargs = call.kwargs
                message = kwargs["message"]
                assert "Weekly Scholar Scout Update" in message
                if kwargs["channel"] == "#systems-for-ai":
                    assert "Test LLM Paper" in message
                if kwargs["channel"] == "#serverless-systems":
                    assert "No relevant papers were found" in message

        finally:
            # Clean up test artifacts
            import os

            if os.path.exists(".last_weekly_update"):
                os.remove(".last_weekly_update")


if __name__ == "__main__":
    unittest.main()
