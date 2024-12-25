from dataclasses import dataclass
from typing import List


@dataclass
class Paper:
    """
    Represents a research paper with its metadata.

    Attributes:
        title: Paper title
        authors: List of author names
        abstract: Paper abstract
        url: Link to the paper (optional)
        venue: Publication venue (optional)
    """

    title: str
    authors: List[str]
    abstract: str
    url: str = ""  # Optional URL to the paper
    venue: str = ""  # Optional publication venue

    def __init__(
        self,
        title: str,
        authors: List[str],
        abstract: str,
        url: str = "",
        venue: str = "",
    ):
        """Initialize Paper object with immutable attributes."""
        # Use object.__setattr__ because class is frozen
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "authors", authors)
        object.__setattr__(self, "abstract", abstract)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "venue", venue)
