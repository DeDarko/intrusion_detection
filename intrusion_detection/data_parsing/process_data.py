import pandas as pd
from intrusion_detection import constants


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    all_sequences = []
    for connection_id in get_unique_session_ids(data):
        pass


def get_unique_session_ids(data: pd.DataFrame) -> pd.Series:
    return data[constants.ColumnNames.SESSION.value].unique()
