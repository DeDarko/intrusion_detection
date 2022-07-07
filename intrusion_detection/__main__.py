import pathlib
import pickle

import numpy as np
import pandas as pd
import tensorflow
import typer
from sklearn import model_selection
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import pad_sequences

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
        sequences=pad_sequences(
            sequences=sequences_x,
            maxlen=constants.MAXIMAL_SEQUENCE_LENGTH,
            padding="pre",
        )
    )

    y = process_data.sequence_to_matrix(sequences_y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=1993
    )

    target_directory = pathlib.Path(target_directory).resolve()
    target_directory.mkdir(exist_ok=True, parents=True)

    np.save(target_directory / "y_train.npy", y_train)
    np.save(target_directory / "x_train.npy", x_train)
    np.save(target_directory / "y_test.npy", y_test)
    np.save(target_directory / "x_test.npy", x_test)
    with open(target_directory / "label_encoder.pickle", "wb") as label_encoder_handle:
        pickle.dump(label_encoder, label_encoder_handle)


@intrusion_detection.command()
def train_model(data_directory: str, target_directory: str):
    y = np.load(pathlib.Path(data_directory) / "y_train.npy")[:, np.newaxis]
    x = np.load(pathlib.Path(target_directory) / "x_train.npy")[
        :, :, np.newaxis
    ].astype(float)

    intrusion_detector = Sequential(
        [
            Embedding(input_dim=len(constants.EVENTS_MAP) + 1, output_dim=50),
            LSTM(
                units=50,
            ),
            Dense(units=100, activation="softmax"),
        ]
    )
    intrusion_detector.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=constants.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
    )

    intrusion_detector.fit(
        x,
        y,
        epochs=constants.N_EPOCHS,
        batch_size=constants.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[
            tensorflow.keras.callbacks.EarlyStopping(monitor="loss", patience=6)
        ],
    )
    with open(
        pathlib.Path(target_directory) / "intrusion_detector.pickle", "wb"
    ) as intrusion_detector_handel:
        pickle.dump(intrusion_detector, intrusion_detector_handel)


if __name__ == "__main__":
    intrusion_detection()
