
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

ROOT = Path(__file__).resolve().parents[1]
CREDENTIALS_FILE = ROOT / "credentials.json"
TOKEN_FILE = ROOT / "token.json"

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
LABEL_NAME = "AI-Operations-Test"


def main():
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_FILE), SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json())
        TOKEN_FILE.chmod(0o600)

    service = build("gmail", "v1", credentials=creds)

    labels = service.users().labels().list(
        userId="me"
    ).execute().get("labels", [])

    label = next(
        (item for item in labels if item["name"] == LABEL_NAME),
        None
    )

    if label is None:
        print("Test label not found in this Gmail account.")
        return

    messages = service.users().messages().list(
        userId="me",
        labelIds=[label["id"]],
        maxResults=5
    ).execute().get("messages", [])

    print(f"Found {len(messages)} labeled email(s).")

    for message in messages:
        details = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full"
        ).execute()

        headers = details["payload"].get("headers", [])
        subject = next(
            (h["value"] for h in headers
             if h["name"].lower() == "subject"),
            "(No subject)"
        )

        print("Subject:", subject)
        print("Preview:", details.get("snippet", ""))


if __name__ == "__main__":
    main()
