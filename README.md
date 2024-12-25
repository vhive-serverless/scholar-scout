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

## Testing

### Integration Tests
To run the integration tests:
```bash
python -m unittest tests/test_integration.py -v
```

The integration tests require a `test_config.yml` file in the `tests/` directory with your Gmail credentials and settings. Example structure:

```yaml
email:
  username: your.email@gmail.com
  password: your-app-specific-password  # Gmail App Password, not your regular password
  folder: "news &- papers/scholar"      # IMAP folder where Scholar alerts are stored
```

Note: 
- You'll need to [create an App Password](https://support.google.com/accounts/answer/185833) for Gmail
- The tests expect Google Scholar alert emails from December 23, 2024 in the specified folder
- Make sure your Scholar alerts are being properly filtered to the specified folder

## License
MIT
# scholar-scout
