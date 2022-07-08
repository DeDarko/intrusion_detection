import pandas as pd


def set_connection_id(data: pd.DataFrame, connection_id: str) -> pd.DataFrame:
    data["connection_id"] = connection_id
    return data


raw_fake_data = pd.read_csv(
    "/home/dennis/Documents/intrusion_detection/data/raw-fake-data.csv", sep=";"
)

complete_data = pd.concat(
    [
        set_connection_id(raw_fake_data.copy(), f"session-{current_index}")
        for current_index in range(1000)
    ]
)

complete_data.to_csv(
    "/home/dennis/Documents/intrusion_detection/data/fake-data.csv", sep=";"
)
