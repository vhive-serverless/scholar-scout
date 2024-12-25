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

import os
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import List, Tuple

import yaml


@dataclass(frozen=True)
class ResearchTopic:
    """
    Represents a research topic for paper classification.

    Attributes:
        name: Topic identifier (e.g., "LLM Inference")
        keywords: Search terms for this topic
        slack_users: Users to notify about papers in this topic
        slack_channel: Channel for notifications (optional)
        description: Detailed description of the topic
    """

    name: str
    keywords: Tuple[str, ...]  # Immutable sequence of keywords
    slack_users: List[str]
    slack_channel: str
    description: str

    def __init__(
        self,
        name: str,
        keywords: List[str],
        slack_users: List[str],
        description: str,
        slack_channel: str = None,
    ):
        """Initialize ResearchTopic with immutable attributes."""
        # Use object.__setattr__ because class is frozen
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "keywords", tuple(keywords))  # Convert list to tuple
        object.__setattr__(self, "slack_users", slack_users)
        object.__setattr__(self, "slack_channel", slack_channel)
        object.__setattr__(self, "description", description)

    def __post_init__(self):
        """Ensure slack_users is always a list."""
        # Convert single user to list if necessary
        if isinstance(self.slack_users, str):
            self.slack_users = [self.slack_users]


def load_config(config_path=None):
    """Load configuration from YAML file with environment variable substitution."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yml"

    with open(config_path) as f:
        # Read the file content
        template = Template(f.read())

        # Substitute environment variables
        yaml_content = template.safe_substitute(
            GMAIL_USERNAME=os.getenv("GMAIL_USERNAME", ""),
            GMAIL_APP_PASSWORD=os.getenv("GMAIL_APP_PASSWORD", ""),
            SLACK_API_TOKEN=os.getenv("SLACK_API_TOKEN", ""),
            PPLX_API_KEY=os.getenv("PPLX_API_KEY", ""),
        )

        # Load YAML with substituted values
        config = yaml.safe_load(yaml_content)

        # Verify required credentials are present
        if not config["email"]["username"] or not config["email"]["password"]:
            raise ValueError("Gmail credentials not found in environment variables")

        return config
