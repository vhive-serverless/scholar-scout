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

import imaplib
import email
import os
from email.header import decode_header
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from openai import OpenAI
import yaml
import logging
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from bs4 import BeautifulSoup

@dataclass
class ResearchTopic:
    name: str
    keywords: List[str]
    slack_user: str
    description: str

@dataclass
class Paper:
    title: str
    authors: List[str]
    abstract: str
    url: str = ""
    venue: str = ""

class ScholarClassifier:
    def __init__(self, config_path='config.yml', config_dict=None):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='scholar_notifications.log'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration from either source
        if config_dict is not None:
            self.config = config_dict
        else:
            self.config = self._load_config(config_path)
        
        # Initialize clients
        self.slack_client = WebClient(token=self.config['slack']['api_token'])
        self.pplx_client = OpenAI(api_key=self.config['perplexity']['api_key'], base_url="https://api.perplexity.ai")
        
        # Load research topics
        self.topics = self._load_research_topics()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def _load_research_topics(self) -> List[ResearchTopic]:
        """Load research topics from configuration."""
        topics = []
        for topic_config in self.config['research_topics']:
            topics.append(ResearchTopic(
                name=topic_config['name'],
                keywords=topic_config['keywords'],
                slack_user=topic_config['slack_user'],
                description=topic_config['description']
            ))
        return topics

    def connect_to_gmail(self) -> imaplib.IMAP4_SSL:
        """Establish connection to Gmail using IMAP."""
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(self.config['email']['username'], self.config['email']['password'])
            return mail
        except Exception as e:
            self.logger.error(f"Error connecting to Gmail: {e}")
            raise

    def extract_and_classify_papers(self, email_message):
        """Extract papers from email and classify them in one API call."""
        print("\nDEBUG: Starting paper extraction and classification")
        
        if not email_message['subject'] or 'new articles' not in email_message['subject'].lower():
            print("DEBUG: Not a valid Google Scholar alert email")
            return []

        # Get email content
        content = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/html":
                    html_content = part.get_payload(decode=True).decode()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    content = soup.get_text('\n', strip=True)
                    break
        else:
            content = email_message.get_payload(decode=True).decode()
            if email_message.get_content_type() == 'text/html':
                soup = BeautifulSoup(content, 'html.parser')
                content = soup.get_text('\n', strip=True)

        if not content:
            print("DEBUG: No content found")
            return []

        # Construct topics description
        topics_description = "\n".join([
            f"- {topic.name}: {topic.description}" 
            for topic in self.topics
        ])

        # Combined prompt for extraction and classification
        prompt = f"""Parse this Google Scholar alert email and for each paper:
1. Extract the paper details
2. Classify it according to the research topics below

Research Topics:
{topics_description}

Email content:
{content}

Return a JSON array where each paper has this structure:
{{
    "title": "paper title",
    "authors": ["author1", "author2"],
    "venue": "conference/journal name or arxiv",
    "link": "link to paper (if available)",
    "abstract": "paper abstract or summary",
    "relevant_topics": ["Topic1", "Topic2"]  // Only include confident matches
}}

Only return the JSON array, nothing else."""

        print(f"DEBUG: Prompt: {prompt}")
        try:
            response = self.pplx_client.chat.completions.create(
                model="llama-3.1-sonar-small-128k-online",
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0]
            
            papers_data = json.loads(content)
            
            results = []
            for paper_data in papers_data:
                paper = Paper(
                    title=paper_data['title'],
                    authors=paper_data['authors'],
                    abstract=paper_data.get('abstract', ''),
                    venue=paper_data.get('venue', ''),
                    url=paper_data.get('link', '')
                )
                
                relevant_topics = [
                    topic for topic in self.topics 
                    if topic.name in paper_data.get('relevant_topics', [])
                ]
                
                results.append((paper, relevant_topics))

            print(f"DEBUG: Results: {results}")
            
            return results
            
        except Exception as e:
            print(f"DEBUG: Error in extraction and classification: {str(e)}")
            raise

    def run(self, folder='"news &- papers/scholar"'):
        """Main execution loop."""
        try:
            mail = self.connect_to_gmail()
            mail.select(folder)
            
            _, message_numbers = mail.search(None, 
                                          '(FROM "scholaralerts-noreply@google.com" UNSEEN)')
            
            self.logger.info(f"Searching in folder: {folder}")
            
            for num in message_numbers[0].split():
                _, msg_data = mail.fetch(num, '(RFC822)')
                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)
                
                # Extract and classify papers in one step
                paper_results = self.extract_and_classify_papers(email_message)
                
                # Send notifications for each paper and its relevant topics
                for paper, relevant_topics in paper_results:
                    for topic in relevant_topics:
                        self.send_slack_notification(topic, paper)
                
                # Mark email as read
                # mail.store(num, '+FLAGS', '\\Seen')
            
            mail.logout()
            
        except Exception as e:
            self.logger.error(f"Error in main execution: {e}")
            raise

    def list_folders(self):
        """List all available folders in the Gmail account."""
        try:
            mail = self.connect_to_gmail()
            _, folders = mail.list()
            for folder in folders:
                folder_name = folder.decode().split('"/"')[-1].strip('"')
                self.logger.info(f"Found folder: {folder_name}")
            mail.logout()
            return folders
        except Exception as e:
            self.logger.error(f"Error listing folders: {e}")
            raise

if __name__ == "__main__":
    # Example configuration file structure (config.yml):
    """
    email:
      username: "your-email@gmail.com"
      password: "your-app-specific-password"
    
    slack:
      api_token: "xoxb-your-slack-token"
    
    perplexity:
      api_key: "your-perplexity-api-key"
    
    research_topics:
      - name: "LLM Inference"
        keywords: ["llm", "inference", "serving", "latency", "throughput"]
        slack_user: "@bill"
        description: "Research related to serving and optimizing LLM inference, including latency optimization, throughput improvement, and deployment strategies"
      
      - name: "Serverless Computing"
        keywords: ["serverless", "faas", "function-as-a-service", "cloud"]
        slack_user: "@jenny"
        description: "Research on serverless computing platforms, Function-as-a-Service (FaaS), and cloud-native architectures"
      
      - name: "Sustainable Computing"
        keywords: ["green computing", "energy efficiency", "carbon footprint"]
        slack_user: "@jim"
        description: "Research on reducing the environmental impact of computing systems, including energy efficiency and carbon footprint reduction"
    """
    
    # Using config file
    classifier = ScholarClassifier(config_path='config.yml')

    # Using config dictionary
    config = {
        'email': {'username': 'email@example.com', 'password': 'pass'},
        'slack': {'api_token': 'token'},
        'perplexity': {'api_key': 'key'},
        'research_topics': [...]
    }
    classifier = ScholarClassifier(config_dict=config)
    classifier.run()