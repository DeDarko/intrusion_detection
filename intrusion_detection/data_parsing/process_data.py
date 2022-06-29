import pandas as pd
from intrusion_detection import constants
from typing import List


def extract_sequences(data: pd.DataFrame) -> List[List[str]]:
    all_sequences = []
    for session_id in get_unique_session_ids(data):
        all_sequences.insert(select_session_extract_to_list(data, session_id))
    return all_sequences


def remove_sequence_min_occurences(
    sequences: List[List[str]], min_occurences: int
) -> List[List[str]]:
    return [sequence for sequence in sequences if len(sequence) >= min_occurences]


def select_session_extract_to_list(data: pd.DataFrame, session: str) -> List:
    return list(
        data[data[constants.ColumnNames.SESSION.value] == session][
            constants.ColumnNames.SESSION.value
        ].values
    )


def get_unique_session_ids(data: pd.DataFrame) -> pd.Series:
    return data[constants.ColumnNames.SESSION.value].unique()
