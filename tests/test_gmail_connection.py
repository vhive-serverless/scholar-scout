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

import email
import os
import sys
import unittest
from pathlib import Path

from dotenv import load_dotenv
from utils import load_config

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scholar_classifier import ScholarClassifier


class TestGmailConnection(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Try loading from .env.test file first (for local development)
        env_path = os.path.join(os.path.dirname(__file__), ".env.test")
        if not load_dotenv(env_path, override=True):
            # For CI environment, ensure required env vars are set
            required_vars = ["GMAIL_USERNAME", "GMAIL_APP_PASSWORD", "PPLX_API_KEY"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            if missing_vars:
                raise RuntimeError(f"Missing required environment variables: {missing_vars}")

        # Load test configuration
        config_path = os.path.join(os.path.dirname(__file__), "test_config.yml")
        self.config = load_config(config_path)
        self.classifier = ScholarClassifier(config_dict=self.config)

    def test_gmail_connection_and_retrieval(self):
        """
        Test connecting to Gmail and retrieving Google Scholar emails without marking them as read.
        """
        try:
            # Connect to Gmail
            mail = self.classifier.connect_to_gmail()

            # Print connection details for debugging
            print("\nAttempting connection with:")
            print(f"Username: {self.config['email']['username']}")
            print(f"Password length: {len(self.config['email']['password'])} chars")

            # Select the folder
            # for dmitrii's account
            # folder = '"news &- papers/scholar"'
            # for hyscale's account
            folder_name = self.config['email'].get('folder', 'INBOX')
            # Handle labels with spaces
            if ' ' in folder_name and not folder_name.startswith('"'):
                folder_name = f'"{folder_name}"'

            mail.select(folder_name)
            # Search for emails from Google Scholar with 'new articles' in subject
            _, message_numbers = mail.search(
                None, '(FROM "scholaralerts-noreply@google.com" SUBJECT "new articles")'
            )

            messages = message_numbers[0].split()
            print(f"\nFound {len(messages)} Google Scholar alert emails with 'new articles'")

            # Test processing of 2 most recent emails
            for num in messages[-2:] if len(messages) > 2 else messages:
                _, msg_data = mail.fetch(num, "(RFC822)")
                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)

                # Print email details for verification
                print("\nEmail details:")
                print(f"Subject: {email_message['subject']}")
                print(f"From: {email_message['from']}")
                print(f"Date: {email_message['date']}")

                # Extract papers
                results = self.classifier.extract_and_classify_papers(email_message)
                print(f"\nNumber of papers extracted: {len(results)}")

                # Basic assertions
                self.assertTrue(len(results) > 0, "Should extract at least one paper")
                paper, topics = results[0]  # Unpack the tuple
                self.assertTrue(len(paper.title) > 0, "Paper should have a title")
                self.assertTrue(len(paper.authors) > 0, "Paper should have authors")

                # Print paper details
                for i, (paper, topics) in enumerate(results):
                    print(f"\nPaper {i + 1}:")
                    print(f"Title: {paper.title}")
                    print(f"Authors: {', '.join(paper.authors)}")
                    print(f"Venue: {paper.venue}")
                    print(f"Abstract preview: {paper.abstract[:200]}...")
                    print(f"Matched Topics: {[topic.name for topic in topics]}")

            mail.close()
            mail.logout()

        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")


if __name__ == "__main__":
    unittest.main()
