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

    def extract_papers_from_email(self, email_message):
        """Extract papers from a Google Scholar alert email using Perplexity AI."""
        print("\nDEBUG: Starting paper extraction")
        print(f"DEBUG: Email subject: {email_message['subject']}")

        if not self.pplx_client or not hasattr(self.pplx_client, 'chat'):
            print("DEBUG: Perplexity client not properly initialized")
            raise ValueError("Perplexity client not properly initialized. Check your API key.")

        if not email_message['subject'] or 'new articles' not in email_message['subject'].lower():
            print("DEBUG: Not a valid Google Scholar alert email")
            return []

        # Get email content
        content = ""
        if email_message.is_multipart():
            print("DEBUG: Processing multipart email")
            for part in email_message.walk():
                content_type = part.get_content_type()
                print(f"DEBUG: Found part with content type: {content_type}")
                if content_type == "text/html":
                    html_content = part.get_payload(decode=True).decode()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    content = soup.get_text('\n', strip=True)
                    break
        else:
            print("DEBUG: Processing single part email")
            content = email_message.get_payload(decode=True).decode()
            if email_message.get_content_type() == 'text/html':
                soup = BeautifulSoup(content, 'html.parser')
                content = soup.get_text('\n', strip=True)

        if not content:
            print("DEBUG: No content found")
            return []

        print(f"DEBUG: Final content length: {len(content)}")
        print(f"DEBUG: Content preview: {content[:500]}")

        # Construct prompt for Perplexity
        prompt = f"""Please parse this Google Scholar alert email and extract all papers mentioned. 
For each paper, provide the title, authors, venue (if any), and abstract.
Return the results as a JSON array where each paper has the following structure:
{{
    "title": "paper title",
    "authors": ["author1", "author2"],
    "venue": "conference/journal name",
    "abstract": "paper abstract"
}}

Here's the email content:
{content}

Only return the JSON array, nothing else."""

        try:
            print("DEBUG: Sending request to Perplexity API")
            response = self.pplx_client.chat.completions.create(
                model="llama-3.1-sonar-small-128k-online",
                messages=[{"role": "user", "content": prompt}]
            )
            
            print("DEBUG: Got response from Perplexity API")
            content = response.choices[0].message.content
            print(f"DEBUG: Response content: {content[:200]}...")
            
            # Clean up markdown formatting if present
            if content.startswith('```json'):
                content = content.split('```json')[1]
                if '```' in content:
                    content = content.split('```')[0]
            
            # Remove any leading/trailing whitespace
            content = content.strip()
            
            # Parse the JSON
            papers_data = json.loads(content)
            
            # Convert to Paper objects
            papers = [
                Paper(
                    title=paper['title'],
                    authors=paper['authors'],
                    abstract=paper.get('abstract', ''),
                    venue=paper.get('venue', '')
                )
                for paper in papers_data
            ]
            
            print(f"DEBUG: Extracted {len(papers)} papers")
            for i, paper in enumerate(papers):
                print(f"\nDEBUG: Paper {i+1}:")
                print(f"Title: {paper.title}")
                print(f"Authors: {', '.join(paper.authors)}")
                print(f"Venue: {paper.venue}")
                print(f"Abstract preview: {paper.abstract[:100]}...")
            
            return papers
            
        except Exception as e:
            print(f"DEBUG: Error parsing with Perplexity: {str(e)}")
            if "401" in str(e):
                raise ValueError("Invalid Perplexity API key. Please check your configuration.")
            raise

    def _process_email(self, email_message) -> Tuple[str, str]:
        """Process email to extract subject and body."""
        subject = ""
        body = ""

        # Get subject
        if email_message['subject']:
            subject = decode_header(email_message['subject'])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()

        # Get body
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = email_message.get_payload(decode=True).decode()

        return subject, body

    def classify_paper(self, paper: Paper) -> List[ResearchTopic]:
        """Use AI to classify paper into relevant research topics."""
        # Construct the prompt
        topics_description = "\n".join([
            f"- {topic.name}: {topic.description}" 
            for topic in self.topics
        ])
        
        prompt = f"""Please analyze this research paper and determine which of the following research topics it belongs to. A paper can belong to multiple topics if relevant.

Research Topics:
{topics_description}

Paper Information:
Title: {paper.title}
Authors: {', '.join(paper.authors)}
Abstract: {paper.abstract}

Please respond with a JSON array containing the names of the relevant topics. Include a topic only if you are confident it is relevant. For example:
["LLM Inference", "Sustainable Computing"]

Only return the JSON array, nothing else."""

        try:
            response = self.pplx_client.chat.completions.create(
                model="llama-3.1-sonar-small-128k-online",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the response
            matched_topics = json.loads(response.choices[0].message.content)
            
            # Map topic names to ResearchTopic objects
            return [
                topic for topic in self.topics 
                if topic.name in matched_topics
            ]
            
        except Exception as e:
            self.logger.error(f"Error classifying paper with AI: {e}")
            return []

    def send_slack_notification(self, topic: ResearchTopic, paper: Paper):
        """Send notification to appropriate Slack user."""
        try:
            message = (
                f"🎯 *New Relevant Paper in {topic.name}*\n\n"
                f"*Title:* {paper.title}\n"
                f"*Authors:* {', '.join(paper.authors)}\n\n"
                f"*Abstract:*\n{paper.abstract}\n\n"
                f"*Why you might be interested:* This paper appears relevant to your work on {topic.name}."
            )
            
            self.slack_client.chat_postMessage(
                channel=topic.slack_user,  # This can be a user ID or channel
                text=message,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": message}
                    }
                ]
            )
            self.logger.info(f"Notification sent to {topic.slack_user} for topic {topic.name}")
            
        except SlackApiError as e:
            self.logger.error(f"Error sending Slack notification: {e}")

    def run(self, folder='"news &- papers/scholar"'):
        """Main execution loop."""
        try:
            mail = self.connect_to_gmail()
            mail.select(folder)  # Allow specifying which folder to search
            
            # Search for unread emails from Google Scholar
            _, message_numbers = mail.search(None, 
                                          '(FROM "scholaralerts-noreply@google.com" UNSEEN)')
            
            self.logger.info(f"Searching in folder: {folder}")
            
            for num in message_numbers[0].split():
                # Fetch email message
                _, msg_data = mail.fetch(num, '(RFC822)')
                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)
                
                # Extract paper information
                paper = self.extract_paper_info(email_message)
                
                # Classify paper using AI
                relevant_topics = self.classify_paper(paper)
                
                # Send notifications for each relevant topic
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