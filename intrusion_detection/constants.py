import enum

REAL_DATA_PATH = "data/live_logs_ids_bereinigt.csv"

MINIMAL_SEQUENCE_LENGTH = 2
MAXIMAL_SEQUENCE_LENGTH = 30


class ColumnNames(enum.Enum):
    SESSION = "connectionid"
    DATE = "time"
    EVENT = "event"
    USER = "patientId"


EVENTS_MAP = {
    "TC:Services:intrusionDetectionSystem:new connection": "CONNECTED",
    "TC:Services:intrusionDetectionSystem:new patient registered": "REGISTERED",
    "TC:Services:intrusionDetectionSystem:patient logged in": "LOGGED-IN",
    "TC:Services:intrusionDetectionSystem:patient logged out": "LOGGED-OUT",
    "TC:Services:intrusionDetectionSystem:patient manual login": "RESUME",
    "TC:Services:intrusionDetectionSystem:patient sent message": "SENT-MESSAGE",
    "TC:Services:intrusionDetectionSystem:patient stream subscription": "STREAM",
}

PADDING_TOKEN_NAME = "DISCONNECT(DETECTED)"
