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
from scholar_classifier import ScholarClassifier

def test_extract_and_classify_papers(mocker):
    # Mock email message
    email_message = mocker.Mock()
    email_message.is_multipart.return_value = True
    email_message.__getitem__.return_value = "New articles in your Google Scholar alert"
    
    # Mock email part with HTML content
    html_content = """
    <div>
        <h3>Title: Efficient LLM Inference on Serverless Platforms</h3>
        <div>Authors: John Doe, Jane Smith</div>
        <div>Conference: SOSP 2024</div>
        <div>Abstract: This paper presents novel techniques for optimizing LLM inference in serverless environments.</div>
        <a href="http://example.com/paper">Link to paper</a>
    </div>
    """
    
    email_part = mocker.Mock()
    email_part.get_content_type.return_value = "text/html"
    email_part.get_payload.return_value = html_content.encode()
    
    # Mock email walk
    email_message.walk.return_value = [email_part]
    
    # Mock Perplexity API response
    mock_response = mocker.Mock()
    mock_response.choices = [
        mocker.Mock(
            message=mocker.Mock(
                content='''[{
                    "title": "Efficient LLM Inference on Serverless Platforms",
                    "authors": ["John Doe", "Jane Smith"],
                    "venue": "SOSP 2024",
                    "link": "http://example.com/paper",
                    "abstract": "This paper presents novel techniques for optimizing LLM inference in serverless environments.",
                    "relevant_topics": ["LLM Inference", "Serverless Computing"]
                }]'''
            )
        )
    ]
    
    # Create test configuration
    config = {
        'email': {'username': 'test@example.com', 'password': 'test'},
        'slack': {'api_token': 'test-token'},
        'perplexity': {'api_key': 'test-key'},
        'research_topics': [
            {
                'name': 'LLM Inference',
                'keywords': ['llm', 'inference'],
                'slack_user': '@test1',
                'description': 'LLM inference research'
            },
            {
                'name': 'Serverless Computing',
                'keywords': ['serverless'],
                'slack_user': '@test2',
                'description': 'Serverless computing research'
            }
        ]
    }
    
    # Mock Perplexity client
    mocker.patch('openai.OpenAI.chat.completions.create', return_value=mock_response)
    
    # Create classifier instance
    classifier = ScholarClassifier(config_dict=config)
    
    # Run the test
    results = classifier.extract_and_classify_papers(email_message)
    
    # Assertions
    assert len(results) == 1
    paper, topics = results[0]
    
    assert paper.title == "Efficient LLM Inference on Serverless Platforms"
    assert paper.authors == ["John Doe", "Jane Smith"]
    assert paper.venue == "SOSP 2024"
    assert paper.url == "http://example.com/paper"
    assert "LLM inference" in paper.abstract.lower()
    
    assert len(topics) == 2
    topic_names = [t.name for t in topics]
    assert "LLM Inference" in topic_names
    assert "Serverless Computing" in topic_names 

class TestScholarClassifier(TestCase):
    def setUp(self):
        # Create test configuration
        self.config = {
            'email': {'username': 'test@example.com', 'password': 'test'},
            'slack': {'api_token': 'test-token'},
            'perplexity': {'api_key': 'test-key'},
            'research_topics': [
                {
                    'name': 'LLM Inference',
                    'keywords': ['llm', 'inference'],
                    'slack_user': '@test1',
                    'description': 'LLM inference research'
                },
                {
                    'name': 'Serverless Computing',
                    'keywords': ['serverless'],
                    'slack_user': '@test2',
                    'description': 'Serverless computing research'
                }
            ]
        }

    @mock.patch('scholar_classifier.OpenAI')
    def test_extract_and_classify_papers(self, mock_openai_class):
        # Define mock_response first
        mock_response = mock.Mock()
        mock_response.choices = [
            mock.Mock(
                message=mock.Mock(
                    content='''[{
                        "title": "Efficient LLM Inference on Serverless Platforms",
                        "authors": ["John Doe", "Jane Smith"],
                        "venue": "SOSP 2024",
                        "link": "http://example.com/paper",
                        "abstract": "This paper presents novel techniques for optimizing LLM inference in serverless environments.",
                        "relevant_topics": ["LLM Inference", "Serverless Computing"]
                    }]'''
                )
            )
        ]

        # Set up the mock OpenAI client with Perplexity base URL
        mock_client = mock.Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        # Verify that OpenAI is instantiated with correct parameters
        def verify_openai_init(*args, **kwargs):
            assert kwargs.get('base_url') == "https://api.perplexity.ai"
            assert kwargs.get('api_key') == "test-key"
            return mock_client
        mock_openai_class.side_effect = verify_openai_init

        # Create email message mock
        email_message = mock.Mock()
        email_message.is_multipart.return_value = True
        email_message.__getitem__ = mock.Mock(side_effect=lambda x: "New articles in your Google Scholar alert" if x == 'subject' else None)

        # Mock email part with HTML content
        html_content = """
        <div>
            <h3>Title: Efficient LLM Inference on Serverless Platforms</h3>
            <div>Authors: John Doe, Jane Smith</div>
            <div>Conference: SOSP 2024</div>
            <div>Abstract: This paper presents novel techniques for optimizing LLM inference in serverless environments.</div>
            <a href="http://example.com/paper">Link to paper</a>
        </div>
        """
        
        email_part = mock.Mock()
        email_part.get_content_type.return_value = "text/html"
        email_part.get_payload.return_value = html_content.encode()
        email_message.walk.return_value = [email_part]

        # Create classifier instance and run test
        classifier = ScholarClassifier(config_dict=self.config)
        results = classifier.extract_and_classify_papers(email_message)
        
        # Verify that OpenAI was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs['model'] == "llama-3.1-sonar-small-128k-online"
        
        # Assertions for results
        assert len(results) == 1
        paper, topics = results[0]
        
        assert paper.title == "Efficient LLM Inference on Serverless Platforms"
        assert paper.authors == ["John Doe", "Jane Smith"]
        assert paper.venue == "SOSP 2024"
        assert paper.url == "http://example.com/paper"
        assert "llm inference" in paper.abstract.lower() or "optimizing llm" in paper.abstract.lower()
        
        assert len(topics) == 2
        topic_names = [t.name for t in topics]
        assert "LLM Inference" in topic_names
        assert "Serverless Computing" in topic_names

if __name__ == '__main__':
    unittest.main()