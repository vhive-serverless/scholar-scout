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
import os
import imaplib
import email
from dotenv import load_dotenv
from scholar_classifier import ScholarClassifier
import yaml

class TestGmailConnection(unittest.TestCase):
    def setUp(self):
        # Get the absolute path to .env
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        
        # Load with explicit path
        result = load_dotenv(env_path)
        print(f".env file loaded: {result}")
        
        # Create config using environment variables
        self.config = {
            'email': {
                'username': os.getenv('GMAIL_USERNAME'),
                'password': os.getenv('GMAIL_PASSWORD')
            },
            'slack': {
                'api_token': 'test-token'
            },
            'perplexity': {
                'api_key': os.getenv('PPLX_API_KEY')  # Make sure this matches your .env variable name
            },
            'research_topics': [{
                'name': 'Test Topic',
                'keywords': ['test'],
                'slack_user': '@test',
                'description': 'Test description'
            }]
        }
        
        self.classifier = ScholarClassifier(config_dict=self.config)

    def test_gmail_connection_and_retrieval(self):
        """Test connecting to Gmail and retrieving Google Scholar emails without marking them as read."""
        try:
            # Connect to Gmail
            mail = self.classifier.connect_to_gmail()
            
            # Print connection details for debugging
            print("\nAttempting connection with:")
            print(f"Username: {self.config['email']['username']}")
            print(f"Password length: {len(self.config['email']['password'])} chars")
            print(f"Password first/last 2 chars: {self.config['email']['password'][:2]}...{self.config['email']['password'][-2:]}")
            
            # Select the folder
            folder = '"news &- papers/scholar"'
            print(f"\nSelecting folder: {folder}")
            mail.select(folder)
            
            # Search for emails from Google Scholar with 'new articles' in subject
            _, message_numbers = mail.search(None, 
                                          '(FROM "scholaralerts-noreply@google.com" SUBJECT "new articles")')
            
            messages = message_numbers[0].split()
            print(f"\nFound {len(messages)} Google Scholar alert emails with 'new articles' in subject")
            
            # Test processing of 2 most recent emails
            for num in messages[-2:] if len(messages) > 2 else messages:
                _, msg_data = mail.fetch(num, '(RFC822)')
                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)
                
                # Print email details for verification
                print("\nEmail details:")
                print(f"Subject: {email_message['subject']}")
                print(f"From: {email_message['from']}")
                print(f"Date: {email_message['date']}")
                
                # Extract papers
                papers = self.classifier.extract_papers_from_email(email_message)
                print(f"\nNumber of papers extracted: {len(papers)}")
                
                # Basic assertions
                self.assertTrue(len(papers) > 0, "Should extract at least one paper")
                paper = papers[0]
                self.assertTrue(len(paper.title) > 0, "Paper should have a title")
                self.assertTrue(len(paper.authors) > 0, "Paper should have authors")
                
                # Print paper details
                for i, paper in enumerate(papers):
                    print(f"\nPaper {i+1}:")
                    print(f"Title: {paper.title}")
                    print(f"Authors: {', '.join(paper.authors)}")
                    print(f"Venue: {paper.venue}")
                    print(f"Abstract preview: {paper.abstract[:200]}...")
                
            mail.close()
            mail.logout()
            
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

if __name__ == '__main__':
    unittest.main() 