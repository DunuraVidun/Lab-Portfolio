# ============================================================
# LAB 06 - PART D: GOOGLE CALENDAR TOOLS
# ============================================================

import datetime as dt
import re
from pathlib import Path

from langchain_core.tools import tool


# ============================================================
# 1. CONFIGURATION
# ============================================================

SCOPES = ["https://www.googleapis.com/auth/calendar"]

BASE_DIR = Path(__file__).resolve().parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"   # downloaded from Google Cloud
TOKEN_FILE = BASE_DIR / "token.json"               # created after first sign-in

TIMEZONE_NAME = "Asia/Colombo"
LOCAL_TZ = dt.timezone(dt.timedelta(hours=5, minutes=30))   # Sri Lanka

DEFAULT_DURATION_MINUTES = 60


# ============================================================
# 2. GOOGLE AUTHENTICATION
# ============================================================

def get_calendar_service():
    """Return an authorised Google Calendar API client."""

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as error:
        raise RuntimeError(
            "Google libraries are missing. Run: pip install "
            "google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from error

    credentials = None

    if TOKEN_FILE.exists():
        credentials = Credentials.from_authorized_user_file(
            str(TOKEN_FILE), SCOPES
        )

    if not credentials or not credentials.valid:

        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at {CREDENTIALS_FILE}. "
                    "Download the OAuth Desktop client JSON from Google Cloud."
                )

            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            credentials = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(credentials.to_json())

    return build("calendar", "v3", credentials=credentials)


# ============================================================
# 3. TOOLS
# ============================================================

@tool
def calendar_today_tool(date: str = "") -> str:
    """
    Check the user's Google Calendar schedule for one day.

    Args:
        date: Day to check as YYYY-MM-DD. Leave empty for today.
    """

    try:
        if date:
            day = dt.datetime.strptime(date, "%Y-%m-%d").date()
        else:
            day = dt.datetime.now(LOCAL_TZ).date()

        day_start = dt.datetime.combine(day, dt.time.min, tzinfo=LOCAL_TZ)
        day_end = day_start + dt.timedelta(days=1)

        service = get_calendar_service()

        events = service.events().list(
            calendarId="primary",
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute().get("items", [])

        heading = day.strftime("%A %d %B %Y")

        if not events:
            return f"No events on {heading}. Your day is free."

        lines = [f"Schedule for {heading} ({len(events)} event(s)):"]

        for event in events:
            title = event.get("summary", "(No title)")
            start = event["start"]
            end = event["end"]

            if "dateTime" in start:
                start_time = dt.datetime.fromisoformat(start["dateTime"])
                end_time = dt.datetime.fromisoformat(end["dateTime"])
                when = (
                    f"{start_time.astimezone(LOCAL_TZ):%H:%M}"
                    f"-{end_time.astimezone(LOCAL_TZ):%H:%M}"
                )
            else:
                when = "All day"

            lines.append(f"- {when}  {title}")

        return "\n".join(lines)

    except ValueError:
        return "Calendar error: date must be in YYYY-MM-DD format."

    except Exception as error:
        return f"Calendar error: {error}"


@tool
def calendar_create_event_tool(summary: str, start: str, end: str = "") -> str:
    """
    Create a new event in the user's Google Calendar.

    Args:
        summary: Event title.
        start: Start time as YYYY-MM-DDTHH:MM (24-hour), e.g. 2026-11-15T14:30.
        end: End time in the same format. Leave empty for a 1-hour event.
    """

    try:
        start_time = dt.datetime.strptime(start, "%Y-%m-%dT%H:%M")

        if end:
            end_time = dt.datetime.strptime(end, "%Y-%m-%dT%H:%M")
        else:
            end_time = start_time + dt.timedelta(
                minutes=DEFAULT_DURATION_MINUTES
            )

        if end_time <= start_time:
            return "Calendar error: the end time must be after the start time."

        service = get_calendar_service()

        body = {
            "summary": summary,
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": TIMEZONE_NAME,
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": TIMEZONE_NAME,
            },
        }

        created = service.events().insert(
            calendarId="primary", body=body
        ).execute()

        return (
            f"Event created: {summary}\n"
            f"When: {describe_when(start_time, end_time)}\n"
            f"Link: {created.get('htmlLink', 'N/A')}"
        )

    except ValueError:
        return "Calendar error: times must be in YYYY-MM-DDTHH:MM format."

    except Exception as error:
        return f"Calendar error: {error}"


# ============================================================
# 4. NATURAL-LANGUAGE HELPERS 
# ============================================================

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

EVENT_WORDS = (
    "meeting", "event", "appointment", "reminder", "class",
    "call", "lunch", "dinner", "session", "interview",
)

CREATE_PATTERN = re.compile(
    r"\b(?:create|add|schedule|book|set up|make|put)\b.*\b(?:"
    + "|".join(EVENT_WORDS) + r")\b"
)

VIEW_PATTERN = re.compile(
    r"\b(?:calendar|agenda)\b"
    r"|\b(?:am i free|what do i have|anything on|my schedule|schedule for|"
    r"today's schedule|tomorrow's schedule)\b"
    r"|\bschedule\b.*\b(?:today|tomorrow)\b"
    r"|\b(?:today|tomorrow)\b.*\bschedule\b"
)

DOCUMENT_WORDS = re.compile(
    r"\b(?:exam|examination|lecture|module|semester|handbook|timetable)\b"
)

TIME_PATTERN = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b|\b([01]?\d|2[0-3]):([0-5]\d)\b"
)


def detect_calendar_intent(text: str):
    """Return 'create', 'view' or None."""

    lowered = text.lower()

    if CREATE_PATTERN.search(lowered):
        return "create"

    if VIEW_PATTERN.search(lowered) and not DOCUMENT_WORDS.search(lowered):
        return "view"

    return None


def parse_date(text: str):
    """Understand today / tomorrow / YYYY-MM-DD / weekday names."""

    lowered = text.lower()
    today = dt.datetime.now(LOCAL_TZ).date()

    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered)
    if iso:
        try:
            return dt.date(int(iso[1]), int(iso[2]), int(iso[3]))
        except ValueError:
            return None

    if "day after tomorrow" in lowered:
        return today + dt.timedelta(days=2)

    if "tomorrow" in lowered:
        return today + dt.timedelta(days=1)

    if re.search(r"\b(?:today|tonight)\b", lowered):
        return today

    weekday = re.search(r"\b(" + "|".join(WEEKDAYS) + r")\b", lowered)
    if weekday:
        ahead = (WEEKDAYS[weekday.group(1)] - today.weekday()) % 7 or 7
        return today + dt.timedelta(days=ahead)

    return None


def parse_schedule_date(text: str) -> str:
    """Date for the schedule tool: 'YYYY-MM-DD' or '' for today."""

    day = parse_date(text)
    return day.isoformat() if day else ""


def find_times(text: str):
    """Return all clock times found as (hour, minute) tuples."""

    times = []

    for match in TIME_PATTERN.finditer(text.lower()):

        if match.group(3):                       
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)

            if not 1 <= hour <= 12 or minute > 59:
                continue

            hour = hour % 12 + (12 if match.group(3) == "pm" else 0)

        else:                                    
            hour = int(match.group(4))
            minute = int(match.group(5))

        times.append((hour, minute))

    return times


def extract_title(text: str) -> str:
    """Pick an event title from the request."""

    quoted = re.search(r'"([^"]+)"|“([^”]+)”', text)
    if quoted:
        return (quoted.group(1) or quoted.group(2)).strip()

    stop = r"(?=\s+(?:on|at|for|from|tomorrow|today|next)\b|[,.?!]|$)"

    named = re.search(
        r"\b(?:called|named|titled)\s+(.+?)" + stop, text, flags=re.I
    )
    if named:
        return named.group(1).strip()

    with_who = re.search(
        r"\b(" + "|".join(EVENT_WORDS) + r")\s+(with|about)\s+(.+?)" + stop,
        text,
        flags=re.I,
    )
    if with_who:
        return (
            f"{with_who.group(1).capitalize()} "
            f"{with_who.group(2).lower()} {with_who.group(3).strip()}"
        )

    for word in EVENT_WORDS:
        if re.search(rf"\b{word}\b", text, flags=re.I):
            return word.capitalize()

    return "New event"


def parse_event_request(text: str) -> dict:
    """
    Convert a request such as
    'Create a meeting called Project Review tomorrow at 3pm for 2 hours'
    into {'summary', 'start', 'end'}, or {'error': message}.
    """

    times = find_times(text)

    if not times:
        return {
            "error": (
                "I need a start time to create the event. Example: "
                "\"Create a meeting called Project Review tomorrow at 3pm "
                "for 2 hours\"."
            )
        }

    day = parse_date(text)

    if not day:
        return {
            "error": (
            "Please specify the event date. "
            "For example: Create a meeting tomorrow at 3 PM."
            )
        }

    start = dt.datetime.combine(day, dt.time(*times[0]))

    if len(times) > 1:
        end = dt.datetime.combine(day, dt.time(*times[1]))
    else:
        duration = re.search(
            r"\bfor\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)\b",
            text.lower(),
        )
        if duration:
            amount = float(duration.group(1))
            minutes = amount * (60 if duration.group(2).startswith("h") else 1)
        else:
            minutes = DEFAULT_DURATION_MINUTES

        end = start + dt.timedelta(minutes=minutes)

    if end <= start:
        end = start + dt.timedelta(minutes=DEFAULT_DURATION_MINUTES)

    return {
        "summary": extract_title(text),
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
    }


def describe_when(start: dt.datetime, end: dt.datetime) -> str:
    return f"{start:%A %d %B %Y, %H:%M} - {end:%H:%M}"


def describe_event(event: dict) -> str:
    """Human-readable summary used for the approval question."""

    start = dt.datetime.strptime(event["start"], "%Y-%m-%dT%H:%M")
    end = dt.datetime.strptime(event["end"], "%Y-%m-%dT%H:%M")

    return f"{event['summary']} - {describe_when(start, end)}"