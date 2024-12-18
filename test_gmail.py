from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Scopes define the level of access requested
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Authenticate and build the Gmail service
def authenticate_gmail():
    creds = None
    # Use credentials.json downloaded from Google Cloud Console
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
    
    # Save the credentials for future use
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

# Retrieve emails
def get_emails():
    service = authenticate_gmail()
    results = service.users().messages().list(userId='me', maxResults=10).execute()
    messages = results.get('messages', [])

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        print(f"Message snippet: {msg['snippet']}")

if __name__ == "__main__":
    get_emails()
