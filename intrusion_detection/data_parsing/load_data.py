import pandas as pd
from intrusion_detection import constants


def load_raw_data(path_to_data: str) -> pd.DataFrame:
    return sort_by_date(pd.read_csv(path_to_data, sep=";").loc[:10])


def sort_by_date(data: pd.DataFrame) -> pd.DataFrame:
    data[constants.ColumnNames.DATE.value] = pd.to_datetime(
        data[constants.ColumnNames.DATE.value]
    )
    return data.sort_values(by=constants.ColumnNames.DATE.value)


def store_dataframe(data: pd.DataFrame, target_directory: str) -> None:
    data.to_csv(target_directory)
