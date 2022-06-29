import enum

REAL_DATA_PATH = (
    "/Users/dennisfitzner/Documents/Bacheor/Data/live_logs_ids_bereinigt.csv"
)

MINIMAL_SEQUENCE_LENGTH = 1


class ColumnNames(enum.Enum):
    SESSION = "connectionid"
    DATE = "time"
    EVENT = "event"
