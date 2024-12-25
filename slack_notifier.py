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

import logging
from typing import List, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from config import ResearchTopic
from paper import Paper


class SlackNotifier:
    def __init__(self, token: str, default_channel: str, config: dict):
        self.client = WebClient(token=token)
        self.default_channel = default_channel
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set the logging level for slack_sdk to INFO to reduce verbosity
        logging.getLogger("slack_sdk").setLevel(logging.INFO)

    def notify_matches(self, paper_results):
        """Notify about matching papers to their specific channels."""
        if not paper_results:
            return

        for paper, matched_topics in paper_results:
            for topic in matched_topics:
                # Get channel from topic, fall back to default if not specified
                channel = getattr(topic, "slack_channel", self.default_channel)
                users_mention = " ".join(getattr(topic, "slack_users", []))

                message = (
                    f"{users_mention}\n"  # Mention all users at once
                    f"New paper matching topic: {topic.name}\n"
                    f"Title: {paper.title}\n"
                    f"Authors: {', '.join(paper.authors)}\n"
                    f"Venue: {paper.venue}\n"
                    f"URL: {paper.url}\n"
                    f"Abstract: {paper.abstract[:500]}..."
                )

                try:
                    # Send to topic-specific channel
                    self.client.chat_postMessage(channel=channel, text=message, unfurl_links=True)
                    self.logger.info(
                        f"Notification sent to channel {channel} for topic {topic.name}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to send notification to {channel}: {str(e)}")

    def _send_papers_to_channel(
        self, channel: str, topic: Optional[str], papers: List[Paper]
    ) -> None:
        """Send papers to a specific channel."""
        if topic:
            message = f"*{topic}:*\n"
        else:
            message = "*Unmatched Papers:*\n"

        message += "\n".join(f"• {paper.title}" for paper in papers)

        try:
            self.client.chat_postMessage(channel=channel, text=message, unfurl_links=False)
            self.logger.info(f"Sent {len(papers)} papers to {channel}")
        except SlackApiError as e:
            self.logger.error(f"Failed to send message to {channel}: {e.response['error']}")

    def _format_paper_message(self, paper: Paper, topic: ResearchTopic) -> str:
        """Format a paper notification message for Slack."""
        message = [
            f"*New paper matching topic: {topic.name}*",
            f"*Title:* {paper.title}",
            f"*Authors:* {', '.join(paper.authors)}",
        ]

        if paper.venue:
            message.append(f"*Venue:* {paper.venue}")

        if paper.url:
            message.append(f"*URL:* {paper.url}")

        if paper.abstract:
            # Truncate abstract if it's too long
            abstract = paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract
            message.append(f"*Abstract:* {abstract}")

        # Join all parts with newlines
        return "\n".join(message)

    def send_message(self, channel: str, message: str, user: str = None) -> None:
        """Send a message to a Slack channel."""
        try:
            # Add user mention if provided
            if user:
                message = f"{user}\n{message}"

            self.client.chat_postMessage(
                channel=channel, text=message, unfurl_links=False  # Prevent link previews
            )
            self.logger.info(f"Sent message to {channel}")
        except SlackApiError as e:
            self.logger.error(f"Failed to send message to {channel}: {e.response['error']}")
