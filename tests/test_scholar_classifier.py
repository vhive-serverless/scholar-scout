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
from unittest import TestCase, mock
import os

from scholar_classifier import ScholarClassifier
from utils import load_config
from dotenv import load_dotenv


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

    def test_should_process_email_with_chinese_subject(self):
        """Test email filtering with Chinese subject lines."""
        # Create a mock email message with Chinese subject
        email_message = mock.Mock()
        email_message.get.return_value = "新文章 - Google Scholar 学术搜索"
        
        # Create classifier with test config that includes Chinese subjects
        config = {
            "email": {"username": "test@example.com", "password": "test"},
            "slack": {"api_token": "test-token", "default_channel": "#test"},
            "perplexity": {"api_key": "test-key"},
            "research_topics": []
        }
        
        classifier = ScholarClassifier(config_dict=config)
        
        # Mock the search_criteria.yml file to include Chinese subjects
        mock_criteria = {
            'email_filter': {
                'from': 'scholaralerts-noreply@google.com',
                'subject': ['new articles', '新文章', 'new results', '新结果'],
                'time_window': '7D'
            }
        }
        
        # Mock the file reading and YAML loading for _should_process_email
        mock_file_content = '''email_filter:
  from: "scholaralerts-noreply@google.com"
  subject:
    - "new articles"
    - "新文章"
    - "new results"
    - "新结果"
  time_window: "7D"'''
        
        with mock.patch('builtins.open', mock.mock_open(read_data=mock_file_content)):
            with mock.patch('yaml.safe_load', return_value=mock_criteria):
                # Test that Chinese subject should be processed
                result = classifier._should_process_email(email_message)
                self.assertTrue(result, "Chinese subject should be processed")

    def test_should_process_email_with_unrelated_subject(self):
        """Test email filtering with unrelated subject lines."""
        # Create a mock email message with unrelated subject
        email_message = mock.Mock()
        email_message.get.return_value = "Your weekly newsletter"
        
        config = {
            "email": {"username": "test@example.com", "password": "test"},
            "slack": {"api_token": "test-token", "default_channel": "#test"},
            "perplexity": {"api_key": "test-key"},
            "research_topics": []
        }
        
        classifier = ScholarClassifier(config_dict=config)
        
        mock_criteria = {
            'email_filter': {
                'from': 'scholaralerts-noreply@google.com',
                'subject': ['new articles', '新文章', 'new results', '新结果'],
                'time_window': '7D'
            }
        }
        
        # Mock the file reading and YAML loading for _should_process_email
        mock_file_content = '''email_filter:
  from: "scholaralerts-noreply@google.com"
  subject:
    - "new articles"
    - "新文章"
    - "new results"
    - "新结果"
  time_window: "7D"'''
        
        with mock.patch('builtins.open', mock.mock_open(read_data=mock_file_content)):
            with mock.patch('yaml.safe_load', return_value=mock_criteria):
                # Test that unrelated subject should NOT be processed
                result = classifier._should_process_email(email_message)
                self.assertFalse(result, "Unrelated subject should NOT be processed")

    def test_should_process_email_with_empty_subject(self):
        """Test email filtering with empty subject."""
        # Create a mock email message with empty subject
        email_message = mock.Mock()
        email_message.get.return_value = ""
        
        config = {
            "email": {"username": "test@example.com", "password": "test"},
            "slack": {"api_token": "test-token", "default_channel": "#test"},
            "perplexity": {"api_key": "test-key"},
            "research_topics": []
        }
        
        classifier = ScholarClassifier(config_dict=config)
        
        mock_criteria = {
            'email_filter': {
                'from': 'scholaralerts-noreply@google.com',
                'subject': ['new articles', '新文章', 'new results', '新结果'],
                'time_window': '7D'
            }
        }
        
        # Mock the file reading and YAML loading for _should_process_email
        mock_file_content = '''email_filter:
  from: "scholaralerts-noreply@google.com"
  subject:
    - "new articles"
    - "新文章"
    - "new results"
    - "新结果"
  time_window: "7D"'''
        
        with mock.patch('builtins.open', mock.mock_open(read_data=mock_file_content)):
            with mock.patch('yaml.safe_load', return_value=mock_criteria):
                # Test that empty subject should NOT be processed
                result = classifier._should_process_email(email_message)
                self.assertFalse(result, "Empty subject should NOT be processed")

    def test_build_email_search_query_with_chinese_subjects(self):
        """Test IMAP search query building with Chinese subjects."""
        config = {
            "email": {"username": "test@example.com", "password": "test"},
            "slack": {"api_token": "test-token", "default_channel": "#test"},
            "perplexity": {"api_key": "test-key"},
            "research_topics": []
        }
        
        classifier = ScholarClassifier(config_dict=config)
        
        # Mock the search_criteria.yml file
        mock_criteria = {
            'email_filter': {
                'from': 'scholaralerts-noreply@google.com',
                'subject': ['new articles', '新文章', 'new results', '新结果'],
                'time_window': '7D'
            }
        }
        
        with mock.patch('builtins.open', mock.mock_open(read_data='email_filter:\n  from: "scholaralerts-noreply@google.com"\n  subject:\n    - "new articles"\n    - "新文章"\n    - "new results"\n    - "新结果"\n  time_window: "7D"')):
            with mock.patch('yaml.safe_load', return_value=mock_criteria):
                from_query, since_query, subjects = classifier._build_email_search_query()
                
                # Verify FROM query
                self.assertEqual(from_query, 'FROM "scholaralerts-noreply@google.com"')
                
                # Verify SINCE query (should contain a date)
                self.assertIn('SINCE', since_query)
                # Check that it contains a valid date format (DD-MMM-YYYY)
                import re
                date_pattern = r'\d{2}-[A-Za-z]{3}-\d{4}'
                self.assertIsNotNone(re.search(date_pattern, since_query), "Should contain valid date format")
                
                # Verify subjects list contains both English and Chinese
                self.assertIn('new articles', subjects)
                self.assertIn('new results', subjects)
                
                # Verify Chinese subjects are properly encoded (not in original form)
                chinese_subjects = [s for s in subjects if 'SUBJECT' in s]
                self.assertEqual(len(chinese_subjects), 2, "Should have 2 encoded Chinese subjects")
                
                # Verify the encoded subjects are not the original Chinese text
                self.assertNotIn('新文章', subjects, "Chinese subjects should be encoded, not in original form")
                self.assertNotIn('新结果', subjects, "Chinese subjects should be encoded, not in original form")

    def test_mixed_language_subject_processing(self):
        """Test processing of emails with mixed language subjects."""
        # Test various subject combinations
        test_cases = [
            ("新文章 - New articles in your Google Scholar alert", True),
            ("New articles - 新文章 in your Google Scholar alert", True),
            ("Google Scholar: 新结果 for your search", True),
            ("Weekly newsletter from Google", False),
            ("新文章", True),
            ("new articles", True),
            ("", False),
            (None, False)
        ]
        
        config = {
            "email": {"username": "test@example.com", "password": "test"},
            "slack": {"api_token": "test-token", "default_channel": "#test"},
            "perplexity": {"api_key": "test-key"},
            "research_topics": []
        }
        
        classifier = ScholarClassifier(config_dict=config)
        
        mock_criteria = {
            'email_filter': {
                'from': 'scholaralerts-noreply@google.com',
                'subject': ['new articles', '新文章', 'new results', '新结果'],
                'time_window': '7D'
            }
        }
        
        # Mock the file reading and YAML loading for _should_process_email
        mock_file_content = '''email_filter:
  from: "scholaralerts-noreply@google.com"
  subject:
    - "new articles"
    - "新文章"
    - "new results"
    - "新结果"
  time_window: "7D"'''
        
        with mock.patch('builtins.open', mock.mock_open(read_data=mock_file_content)):
            with mock.patch('yaml.safe_load', return_value=mock_criteria):
                for subject, expected_result in test_cases:
                    email_message = mock.Mock()
                    email_message.get.return_value = subject
                    
                    result = classifier._should_process_email(email_message)
                    self.assertEqual(
                        result,
                        expected_result,
                        f"Subject '{subject}' should {'be' if expected_result else 'NOT be'} processed"
                    )


if __name__ == "__main__":
    unittest.main()
