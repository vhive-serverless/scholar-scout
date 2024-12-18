import imaplib
import email
import os
from email.header import decode_header
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import re
from datetime import datetime
import yaml
import logging

class ScholarNotificationSystem:
    def __init__(self, config_path='config.yml'):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='scholar_notifications.log'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Slack client
        self.slack_client = WebClient(token=self.config['slack']['api_token'])
        
        # Conference patterns for classification
        self.conference_patterns = {
            'ICML': r'ICML|International Conference on Machine Learning',
            'NeurIPS': r'NeurIPS|Neural Information Processing Systems',
            'ICLR': r'ICLR|International Conference on Learning Representations',
            'AAAI': r'AAAI|Association for the Advancement of Artificial Intelligence',
            # Add more conferences as needed
        }

    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def connect_to_gmail(self):
        """Establish connection to Gmail using IMAP."""
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(self.config['email']['username'], self.config['email']['password'])
            return mail
        except Exception as e:
            self.logger.error(f"Error connecting to Gmail: {e}")
            raise

    def process_email(self, email_message):
        """Extract and process relevant information from email."""
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

    def classify_conference(self, text):
        """Classify the conference based on text content."""
        for conf, pattern in self.conference_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return conf
        return "Other"

    def calculate_relevance_score(self, text):
        """Calculate relevance score based on keywords and other factors."""
        relevance_score = 0
        keywords = self.config['keywords']
        
        # Simple keyword matching
        for keyword in keywords:
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE))
            relevance_score += count * self.config['keyword_weights'].get(keyword, 1)
            
        # Normalize score between 0 and 1
        max_possible_score = len(keywords) * 5  # Assuming max 5 occurrences per keyword
        normalized_score = min(relevance_score / max_possible_score, 1)
        
        return normalized_score

    def send_slack_notification(self, conference, subject, relevance_score, body_preview):
        """Send notification to appropriate Slack channel."""
        try:
            channel = self.config['slack']['channels'].get(conference, 
                                                         self.config['slack']['default_channel'])
            
            message = (f"*New Paper Alert*\n"
                      f"*Conference:* {conference}\n"
                      f"*Title:* {subject}\n"
                      f"*Relevance Score:* {relevance_score:.2f}\n"
                      f"*Preview:* {body_preview[:200]}...")
            
            self.slack_client.chat_postMessage(
                channel=channel,
                text=message,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": message}
                    }
                ]
            )
            self.logger.info(f"Notification sent to {channel} for {conference}")
            
        except SlackApiError as e:
            self.logger.error(f"Error sending Slack notification: {e}")

    def run(self):
        """Main execution loop."""
        try:
            mail = self.connect_to_gmail()
            mail.select('inbox')
            
            # Search for unread emails from Google Scholar
            _, message_numbers = mail.search(None, 
                                          '(FROM "scholaralerts-noreply@google.com" UNSEEN)')
            
            for num in message_numbers[0].split():
                # Fetch email message
                _, msg_data = mail.fetch(num, '(RFC822)')
                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)
                
                # Process email
                subject, body = self.process_email(email_message)
                
                # Classify conference
                conference = self.classify_conference(subject + " " + body)
                
                # Calculate relevance
                relevance_score = self.calculate_relevance_score(subject + " " + body)
                
                # Send notification if relevance score meets threshold
                if relevance_score >= self.config['minimum_relevance_score']:
                    self.send_slack_notification(
                        conference,
                        subject,
                        relevance_score,
                        body
                    )
                
                # Mark email as read
                mail.store(num, '+FLAGS', '\\Seen')
                
            mail.logout()
            
        except Exception as e:
            self.logger.error(f"Error in main execution: {e}")
            raise

if __name__ == "__main__":
    # Example configuration file structure (config.yml):
    """
    email:
      username: "your-email@gmail.com"
      password: "your-app-specific-password"
    
    slack:
      api_token: "xoxb-your-slack-token"
      default_channel: "#general"
      channels:
        ICML: "#icml-papers"
        NeurIPS: "#neurips-papers"
        ICLR: "#iclr-papers"
        AAAI: "#aaai-papers"
    
    keywords:
      - "machine learning"
      - "deep learning"
      - "neural networks"
      - "reinforcement learning"
      - "artificial intelligence"
    
    keyword_weights:
      "machine learning": 2
      "deep learning": 2
      "neural networks": 1.5
      "reinforcement learning": 1.5
      "artificial intelligence": 1
    
    minimum_relevance_score: 0.3
    """
    
    notifier = ScholarNotificationSystem()
    notifier.run()