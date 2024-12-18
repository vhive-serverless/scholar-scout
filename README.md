# Scholar Scout

A tool to monitor Google Scholar alerts and classify research papers using Perplexity AI.

## Features
- Connects to Gmail to fetch Google Scholar alert emails
- Uses Perplexity AI to parse and extract paper information
- Supports multiple research topics and keywords
- Sends notifications to Slack

## Setup
1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment: `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your credentials
6. Copy `config.example.yml` to `config.yml` and customize settings

## Configuration
Create a `.env` file with:
```
GMAIL_USERNAME=your.email@gmail.com
GMAIL_PASSWORD=your-app-specific-password
PPLX_API_KEY=your-perplexity-api-key
```

## Usage
Run the main script:
```bash
python scholar_classifier.py
```

Run tests:
```bash
python -m unittest tests/test_gmail_connection.py -v
```

## License
MIT
# scholar-scout
