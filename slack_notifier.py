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

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from paper import Paper

class SlackNotifier:
    def __init__(self, token: str, default_channel: str, config: dict):
        self.client = WebClient(token=token)
        self.default_channel = default_channel
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set the logging level for slack_sdk to INFO to reduce verbosity
        logging.getLogger('slack_sdk').setLevel(logging.INFO)

    def notify_matches(self, results: List[Tuple[Paper, List[str]]]) -> None:
        """Send notifications for papers based on their matched topics."""
        self.logger.debug(f"\n=== Starting notifications for {len(results)} papers ===")
        
        # Track unmatched papers to send to default channel
        unmatched_papers = []
        
        # Group papers by topic and channel
        topic_papers = {}  # Dict[str, List[Paper]]
        
        for paper, topics in results:
            self.logger.debug(f"\nProcessing paper: {paper.title}")
            self.logger.debug(f"Matched topics: {topics}")
            
            if not topics:
                self.logger.debug("No topics matched - adding to unmatched papers")
                unmatched_papers.append(paper)
            else:
                # Add paper to each matched topic's papers list
                for topic in topics:
                    if topic not in topic_papers:
                        topic_papers[topic] = []
                    topic_papers[topic].append(paper)
        
        # Send matched papers to their topic channels
        for topic, papers in topic_papers.items():
            channel = self.config['topics'][topic]['channel']  # Get channel from config
            self.logger.debug(f"\nSending {len(papers)} papers to channel {channel}")
            self._send_papers_to_channel(channel, topic, papers)
        
        # Send unmatched papers to default channel
        if unmatched_papers:
            self.logger.debug(f"\nSending {len(unmatched_papers)} unmatched papers to default channel")
            self._send_papers_to_channel(self.default_channel, None, unmatched_papers)

    def _send_papers_to_channel(self, channel: str, topic: Optional[str], papers: List[Paper]) -> None:
        """Send papers to a specific channel."""
        if topic:
            message = f"*{topic}:*\n"
        else:
            message = "*Unmatched Papers:*\n"
        
        message += "\n".join(f"• {paper.title}" for paper in papers)
        
        try:
            self.client.chat_postMessage(
                channel=channel,
                text=message,
                unfurl_links=False
            )
            self.logger.info(f"Sent {len(papers)} papers to {channel}")
        except SlackApiError as e:
            self.logger.error(f"Failed to send message to {channel}: {e.response['error']}")