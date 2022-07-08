import enum

# Data preprocessing.
REAL_DATA_PATH = "data/live_logs_ids_bereinigt.csv"
FAKE_DATA_PATH = "data/fake-data.csv"
ATTACK_DATA_PATH = "data/attacks2.csv"
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

# Model training
LEARNING_RATE = 0.01
N_EPOCHS = 1000
BATCH_SIZE = 32
