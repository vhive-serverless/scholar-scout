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
import html2text

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

    def _extract_paper_urls(self, html):
        """Extract paper URLs and titles from HTML content."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            paper_links = {}
            
            # Find all paper title links (they have both class='gse_alrt_title' and href containing 'scholar_url')
            for title_link in soup.find_all('a', class_='gse_alrt_title'):
                title = title_link.get_text(strip=True)
                url = title_link.get('href', '')
                
                # Extract the actual paper URL from the Google Scholar redirect URL
                if url and 'scholar_url?url=' in url:
                    try:
                        # The actual URL is the value of the 'url' parameter in the query string
                        from urllib.parse import urlparse, parse_qs
                        parsed = urlparse(url)
                        actual_url = parse_qs(parsed.query)['url'][0]
                        if actual_url:
                            paper_links[title] = actual_url
                    except Exception as e:
                        print(f"Error extracting actual URL: {str(e)}")
                        continue
                    
            return paper_links
            
        except Exception as e:
            print(f"Error extracting URLs: {str(e)}")
            return {}

    def _extract_paper_metadata(self, html):
        """Extract paper metadata from HTML content."""
        try:
            print("\nDEBUG: Parsing HTML content")
            soup = BeautifulSoup(html, 'html.parser')
            papers = []
            
            # Find all paper entries (each paper is in an h3 followed by two divs)
            title_links = soup.find_all('a', class_='gse_alrt_title')
            print(f"Found {len(title_links)} title links")
            
            for title_link in title_links:
                # Get the h3 parent
                h3 = title_link.find_parent('h3')
                if not h3:
                    print("No h3 parent found for title link")
                    continue
                    
                # Extract title
                title = title_link.get_text(strip=True)
                print(f"\nFound paper title: {title}")
                
                # Get the next two divs after h3
                author_div = h3.find_next('div')
                abstract_div = author_div.find_next('div', class_='gse_alrt_sni') if author_div else None
                
                if not (author_div and abstract_div):
                    print("Missing author or abstract div")
                    continue
                
                # Extract author text (before the dash/hyphen)
                author_text = author_div.get_text(strip=True)
                print(f"Author text: {author_text}")
                
                if ' - ' in author_text:
                    authors = author_text.split(' - ')[0]
                elif ' – ' in author_text:  # em dash
                    authors = author_text.split(' – ')[0]
                else:
                    authors = author_text
                    
                # Clean up authors and split into list
                authors = [a.strip() for a in authors.split(',') if a.strip()]
                print(f"Extracted authors: {authors}")
                
                # Extract abstract
                abstract = abstract_div.get_text(strip=True)
                print(f"Abstract length: {len(abstract)}")
                
                papers.append({
                    'title': title,
                    'authors': authors,
                    'abstract': abstract
                })
                
            print(f"\nExtracted {len(papers)} papers total")
            return papers
            
        except Exception as e:
            print(f"Error parsing HTML: {str(e)}")
            print("HTML content:", html[:200])  # Print start of HTML for debugging
            return []

    def extract_and_classify_papers(self, email_message):
        """Extract papers from email and classify them according to research topics."""
        print("\nDEBUG: Starting paper extraction and classification")
        
        content = self._get_email_content(email_message)
        papers = self._extract_paper_metadata(content)
        if not papers:
            print("DEBUG: No papers found in email")
            return []
        
        results = []
        for paper in papers:
            prompt = f"""Below is the EXACT content from a Google Scholar alert email. Extract ONLY the venue and relevant topics:

Title: {paper['title']}
Authors: {paper['authors']}
Abstract: {paper['abstract']}

Return a JSON object with ONLY these fields:
{{
    "venue": "use these rules:
      - 'arXiv preprint' if author line has 'arXiv'
      - 'Patent Application' if author line has 'Patent'
      - text between dash and year for published papers
      - 'NOT-FOUND' otherwise",
    "relevant_topics": []  // From these topics:
{self._format_research_topics()}
}}

CRITICAL RULES:
1. Return ONLY the JSON object
2. Use venue rules exactly as specified
3. Only include topics that clearly match the paper
"""

            print(f"DEBUG: Prompt: {prompt}")
            try:
                response = self.pplx_client.chat.completions.create(
                    model="llama-3.1-sonar-small-128k-online",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                content = response.choices[0].message.content.strip()
                if content.startswith('```json'):
                    content = content.split('```json')[1].split('```')[0]
                elif content.startswith('```'):
                    content = content.split('```')[1].split('```')[0]
                
                papers_data = json.loads(content)
                if not papers_data:
                    continue
                    
                paper_data = papers_data[0]  # We only expect one paper per iteration
                
                # Create Paper object using original paper data
                paper_obj = Paper(
                    title=paper['title'],
                    authors=paper['authors'],
                    abstract=paper['abstract'],
                    venue=paper_data.get('venue', ''),  # Only venue from LLM
                    url=self._extract_paper_urls(content).get(paper['title'], '')
                )
                
                relevant_topics = [
                    topic for topic in self.topics 
                    if topic.name in paper_data.get('relevant_topics', [])
                ]
                
                results.append((paper_obj, relevant_topics))

                print(f"DEBUG: Results: {results}")
            except Exception as e:
                print(f"DEBUG: Error processing paper: {str(e)}")
                continue
        
        return results

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

    def _extract_papers_from_html(self, html):
        """Extract papers as plain text, just like reading an email."""
        soup = BeautifulSoup(html, 'html.parser')
        papers = []
        
        # Find all paper entries (h3 tags)
        for h3 in soup.find_all('h3'):
            if not h3.find('a'):  # Skip h3s without links
                continue
            
            # Get all text content from h3 (includes PDF tag if present)
            title = h3.get_text(strip=True)
            
            # Get the next elements
            current = h3
            paper_text = [title]  # Start with title
            
            # Look for the next 2 divs (authors and abstract)
            for _ in range(2):
                current = current.find_next('div')
                if current:
                    text = current.get_text(strip=True)
                    if text:
                        paper_text.append(text)
            
            # Only add if we found some content
            if len(paper_text) > 1:  # At least title and one more element
                papers.append('\n'.join(paper_text))
        
        return '\n\n'.join(papers)

    def _get_email_content(self, email_message):
        """Extract HTML content from email message."""
        print("\nDEBUG: Email structure:")
        print(f"Is multipart: {email_message.is_multipart()}")
        print(f"Content type: {email_message.get_content_type()}")
        
        content = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/html":
                    print("Found HTML part in multipart message")
                    content = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    break
        else:
            print("Trying direct payload")
            payload = email_message.get_payload(decode=True)
            if payload:
                content = payload.decode('utf-8', errors='replace')
        
        print(f"Extracted content length: {len(content)}")
        if len(content) > 0:
            print("First 200 chars of content:", content[:200])
        
        return content

    def _extract_text_from_html(self, html):
        """Extract readable text from HTML content."""
        try:
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find the paper title and details
            title_link = soup.find('a', class_='gse_alrt_title')
            venue_div = soup.find('div', style='color:#006621')
            abstract_div = soup.find('div', class_='gse_alrt_sni')
            
            # Extract text
            parts = []
            if title_link:
                parts.append(title_link.get_text())
            if venue_div:
                parts.append(venue_div.get_text())
            if abstract_div:
                parts.append(abstract_div.get_text())
            
            # Join with newlines
            return '\n'.join(parts)
            
        except Exception as e:
            print(f"Error parsing HTML: {str(e)}")
            # Fall back to basic HTML to text conversion
            h = html2text.HTML2Text()
            h.ignore_links = True
            return h.handle(html)

    def _format_research_topics(self):
        """Format research topics for the prompt."""
        topics = []
        for topic in self.config['research_topics']:
            topics.append(f"- {topic['name']}: {topic['description']}")
        return "\n".join(topics)

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