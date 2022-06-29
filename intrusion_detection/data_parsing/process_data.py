import pandas as pd
from intrusion_detection import constants
from typing import List


def extract_sequences(data: pd.DataFrame) -> List[List[str]]:
    all_sequences = []
    for session_id in get_unique_session_ids(data):
        all_sequences.append(select_session_map_events(data, session_id))
    return all_sequences


def remove_sequence_min_occurences(
    sequences: List[List[str]], min_occurences: int
) -> List[List[str]]:
    return [sequence for sequence in sequences if len(sequence) >= min_occurences]


def select_session_map_events(data: pd.DataFrame, session: str) -> List:
    return map_events(
        list(
            data[data[constants.ColumnNames.SESSION.value] == session][
                constants.ColumnNames.EVENT.value
            ].values
        )
    )


def get_unique_session_ids(data: pd.DataFrame) -> pd.Series:
    return data[constants.ColumnNames.SESSION.value].unique()


def map_events(events: List[str]) -> List[str]:
    events_map = {
        "TC:Services:intrusionDetectionSystem:new connection": "CONNECTED",
        "TC:Services:intrusionDetectionSystem:new patient registered": "REGISTERED",
        "TC:Services:intrusionDetectionSystem:patient logged in": "LOGGED-IN",
        "TC:Services:intrusionDetectionSystem:patient logged out": "LOGGED-OUT",
        "TC:Services:intrusionDetectionSystem:patient manual login": "RESUME",
        "TC:Services:intrusionDetectionSystem:patient sent message": "SENT-MESSAGE",
        "TC:Services:intrusionDetectionSystem:patient stream subscription": "STREAM",
    }
    return [events_map[event] for event in events]
