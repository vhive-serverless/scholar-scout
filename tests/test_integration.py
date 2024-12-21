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
import os
import email
import imaplib
from dotenv import load_dotenv
from scholar_classifier import ScholarClassifier
from bs4 import BeautifulSoup
import html2text
from email.utils import parsedate_to_datetime
from pathlib import Path
from config import load_config
import logging


class TestGmailIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Setup detailed logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Enable debug logging for specific components
        logging.getLogger('slack_notifier').setLevel(logging.DEBUG)
        logging.getLogger('scholar_classifier').setLevel(logging.INFO)
        
        # Add a stream handler if messages aren't showing
        slack_logger = logging.getLogger('slack_notifier')
        if not slack_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s [%(name)s]: %(message)s'))
            slack_logger.addHandler(handler)
            slack_logger.propagate = True  # Ensure messages propagate to root logger
        
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(env_path)
        
        config_path = Path(__file__).parent / "test_config.yml"

        cls.config = load_config(config_path)
        # cls.config = {
        #     'email': {
        #         'username': os.getenv('GMAIL_USERNAME'),
        #         'password': os.getenv('GMAIL_PASSWORD')
        #     },
        #     'slack': {
        #         'api_token': 'test-token',
        #         'default_channel': '#scholar-scout-default'
        #     },
        #     'perplexity': {
        #         'api_key': os.getenv('PPLX_API_KEY')
        #     },
        #     'research_topics': [
        #         {
        #             'name': 'LLM Inference',
        #             'keywords': ['llm', 'inference', 'serving', 'latency', 'throughput'],
        #             'slack_channel': '#scholar-scout-llm',
        #             'slack_user': '@test_user',
        #             'description': 'Research broadly related to LLM and VLM applications and systems'
        #         },
        #         {
        #             'name': 'Serverless systems', 
        #             'keywords': ['serverless', 'cloud', 'edge', 'edge computing', 'cluster management', 'virtualization', 'hypervisor'],
        #             'slack_user': '@test_user',
        #             'slack_channel': '#scholar-scout-serverless',
        #             'description': 'Research related to serverless systems'
        #         }
        #     ]
        # }
        
        cls.classifier = ScholarClassifier(config_dict=cls.config)

    def test_end_to_end_pipeline(self):
        """Test the entire pipeline from Gmail connection to paper classification."""
        try:
            mail = self.classifier.connect_to_gmail()
            
            folder = '"news &- papers/scholar"'
            status, _ = mail.select(folder)
            self.assertEqual(status, 'OK', f"Failed to select folder: {folder}")
            
            _, message_numbers = mail.search(None, 
                'FROM "scholaralerts-noreply@google.com" SUBJECT "new articles"')
            
            messages = message_numbers[0].split()
            self.assertTrue(len(messages) > 0, "No Google Scholar alert emails found")
            print(f"\nFound {len(messages)} Google Scholar alert emails")
            
            # Take 5 most recent emails
            num_emails_to_test = min(3, len(messages))
            test_messages = messages[-num_emails_to_test:]
            
            for i, num in enumerate(test_messages, 1):
                try:
                    print(f"\n=== Processing Email {i}/{num_emails_to_test} ===")
                    _, msg_data = mail.fetch(num, '(BODY.PEEK[])')
                    email_message = email.message_from_bytes(msg_data[0][1])
                    
                    # Parse and format the date
                    date_str = email_message['date']
                    if date_str:
                        date = parsedate_to_datetime(date_str)
                        formatted_date = date.strftime("%Y-%m-%d %H:%M")
                    else:
                        formatted_date = "Unknown date"
                    
                    print(f"Google Scholar Alerts{' ' * 40}{formatted_date}")
                    print()
                    
                    if email_message.is_multipart():
                        for part in email_message.walk():
                            if part.get_content_type() == "text/html":
                                html = part.get_payload(decode=True).decode('utf-8', errors='replace')
                                self.classifier._extract_papers_from_html(html)
                                break  # Only process the first HTML part
                    
                    print("-" * 80)
                    
                    try:
                        results = self.classifier.extract_and_classify_papers(email_message)
                        
                        self.assertTrue(len(results) > 0, f"No papers extracted from email {i}")
                        
                        print(f"\nExtracted {len(results)} papers:")
                        for j, (paper, topics) in enumerate(results, 1):
                            print(f"\n--- Paper {j}/{len(results)} ---")
                            print(f"Title: {paper.title}")
                            print(f"Authors: {', '.join(paper.authors)}")
                            print(f"Venue: {paper.venue}")
                            print(f"URL: {paper.url}")
                            print(f"Abstract: {paper.abstract[:200]}...")
                            print(f"Matched Topics: {[topic.name for topic in topics]}")
                            
                            # Verify paper structure
                            self.assertTrue(len(paper.title) > 0, "Paper missing title")
                            self.assertTrue(len(paper.authors) > 0, "Paper missing authors")
                            self.assertTrue(len(paper.abstract) > 0, "Paper missing abstract")
                    
                    except Exception as e:
                        print(f"Error processing paper: {str(e)}")
                        print("Continuing with next email...")
                        continue
                
                except Exception as e:
                    print(f"Error processing email {i}: {str(e)}")
                    print("Continuing with next email...")
                    continue
            
            mail.close()
            mail.logout()
            
        except Exception as e:
            self.fail(f"Integration test failed: {str(e)}") 