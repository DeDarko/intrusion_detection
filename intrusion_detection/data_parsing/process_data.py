from typing import List, Tuple

import numpy as np
import pandas as pd
from intrusion_detection import constants
from sklearn import preprocessing as sk_preprocessing


def extract_sequences(data: pd.DataFrame) -> List[List[str]]:
    all_sequences = []
    for session_id in get_unique_session_ids(data):
        all_sequences.append(select_session_map_events(data, session_id))
    return all_sequences


def remove_sequence_min_occurences(
    sequences: List[List[str]], min_occurences: int
) -> List[List[str]]:
    return [sequence for sequence in sequences if len(sequence) >= min_occurences]


def cut_to_max_length(sequences: List[List[str]], max_length: int) -> List[List[str]]:
    return [sequence[:max_length] for sequence in sequences]


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
    return [constants.EVENTS_MAP[event] for event in events]


def extract_user_actions(data: pd.DataFrame) -> List[List[str]]:
    all_useraction = []
    for user_id in get_unique_user_ids(data):
        all_useraction.append(select_user_map_events(data, user_id))
    return all_useraction


def select_user_map_events(data: pd.DataFrame, userid: str) -> List:
    return map_events(
        list(
            data[data[constants.ColumnNames.USER.value] == userid][
                constants.ColumnNames.EVENT.value
            ].values
        )
    )


def get_unique_user_ids(data: pd.DataFrame) -> pd.Series:
    return data[constants.ColumnNames.USER.value].unique()


def label_encode_sequences(
    sequences: List[List[str]],
) -> Tuple[List[List[int]], sk_preprocessing.LabelEncoder]:
    label_encoder = sk_preprocessing.LabelEncoder()
    label_encoder.fit(list(constants.EVENTS_MAP.values()))

    label_encoded_sequences = [
        list(label_encoder.transform(sequence)) for sequence in sequences
    ]
    return (label_encoded_sequences, label_encoder)


def pad_sequences_to_same_length(
    sequences: List[List[int]], required_length: int, padding_int: int
) -> List[List[int]]:
    return [
        sequence + [padding_int] * (required_length - len(sequence))
        for sequence in sequences
    ]


def sequence_to_matrix(sequences: List[List[str]]) -> np.array:
    return np.array(sequences)


def extract_and_remove_targets_from_sequence(
    sequences: List[List[str]],
) -> Tuple[List[List[str]], List[str]]:
    return (
        [sequence[:-1] for sequence in sequences],
        [sequence[-1] for sequence in sequences],
    )


def expand_sequence(
    sequence: List[str],
) -> List[List[str]]:
    return [sequence[:max_elem] for max_elem in range(2, len(sequence) + 1)]


def flatten_sequences(list_of_sequences: List[List[str]]) -> List[str]:
    return [x for xs in list_of_sequences for x in xs]
