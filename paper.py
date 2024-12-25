from dataclasses import dataclass
from typing import List

@dataclass
class Paper:
    """Class representing a research paper."""
    title: str
    authors: List[str]
    abstract: str
    url: str
    venue: str 