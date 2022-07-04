import pathlib

import numpy as np
import pandas as pd
import typer
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from intrusion_detection import constants
from intrusion_detection.data_parsing import load_data, process_data

intrusion_detection = typer.Typer()


@intrusion_detection.command()
def preprocess_data(target_directory: str):
    sequences = process_data.remove_sequence_min_occurences(
        sequences=process_data.extract_sequences(
            data=load_data.load_raw_data(path_to_data=constants.REAL_DATA_PATH),
        ),
        min_occurences=constants.MINIMAL_SEQUENCE_LENGTH,
    )

    encoded_sequences, label_encoder = process_data.label_encode_sequences(
        sequences=sequences
    )

    sequences_with_max_length = process_data.cut_to_max_length(
        sequences=encoded_sequences,
        max_length=constants.MAXIMAL_SEQUENCE_LENGTH,
    )

    sequences_x, sequences_y = process_data.extract_and_remove_targets_from_sequence(
        sequences=sequences_with_max_length
    )

    x = process_data.sequence_to_matrix(
        sequences=process_data.pad_sequences_to_same_length(
            sequences=sequences_x,
            required_length=constants.MAXIMAL_SEQUENCE_LENGTH,
            padding_int=label_encoder.transform([constants.PADDING_TOKEN_NAME])[-1],
        )
    )

    y = process_data.sequence_to_matrix(sequences_y)

    target_directory = pathlib.Path(target_directory)
    target_directory.mkdir(exist_ok=True)

    np.save(target_directory / "y.npy", y)
    np.save(target_directory / "X.npy", x)


@intrusion_detection.command()
def train_model(data_directory: str, target_directory: str):
    y = np.load(pathlib.Path(data_directory) / "y.npy")
    x = np.load(pathlib.Path(target_directory) / "X.npy")

    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.25))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.25))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.25))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer="adam", loss="mean_squared_error")

    regressor.fit(x, y, epochs=100, batch_size=32)


if __name__ == "__main__":
    intrusion_detection()
