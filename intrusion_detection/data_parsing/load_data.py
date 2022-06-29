import pandas as pd
from intrusion_detection import constants


def load_raw_data(path_to_data: str) -> pd.DataFrame:
    return pd.read_csv(path_to_data, sep=";").loc[:10]


def sort_by_date(data: pd.DataFrame) -> pd.DataFrame:
    data[constants.ColumnNames.DATE] = pd.to_datetime(data[constants.ColumnNames.DATE])
    data.sort_values()


def store_dataframe(data: pd.DataFrame, target_directory: str) -> None:
    data.to_csv(target_directory)
