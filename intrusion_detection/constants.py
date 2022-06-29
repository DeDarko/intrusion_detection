import enum

REAL_DATA_PATH = "data/live_logs_ids_bereinigt.csv"

MINIMAL_SEQUENCE_LENGTH = 2


class ColumnNames(enum.Enum):
    SESSION = "connectionid"
    DATE = "time"
    EVENT = "event"
