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
from intrusion_detection.model_evaluation import verwirrungsgrad_computation

intrusion_detection = typer.Typer()


@intrusion_detection.command()
def preprocess_data(
    target_directory: str, expand_sequences: bool = typer.Argument(True)
):
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

    if expand_sequences:
        sequences_expanded = process_data.flatten_sequences(
            list_of_sequences=[
                process_data.expand_sequence(sequence=sequence)
                for sequence in sequences_with_max_length
            ]
        )
    else:
        expand_sequences = sequences_with_max_length

    sequences_x, sequences_y = process_data.extract_and_remove_targets_from_sequence(
        sequences=sequences_expanded
    )

    x = process_data.sequence_to_matrix(
        sequences=pad_sequences(
            sequences=sequences_x,
            maxlen=constants.MAXIMAL_SEQUENCE_LENGTH,
            value=len(constants.EVENTS_MAP) + 1.0,
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
            Embedding(input_dim=len(constants.EVENTS_MAP) + 2, output_dim=50),
            LSTM(
                units=50,
            ),
            Dense(units=200, activation="relu"),
            Dense(units=len(constants.EVENTS_MAP), activation="softmax"),
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


@intrusion_detection.command()
def evaluate_verwirrungsgrad(
    path_to_model: str, path_to_x: str, path_to_y: str, output_path: str
) -> None:
    with open(path_to_model, "rb") as model_handle:
        intrusion_detector = pickle.load(model_handle)

    x = np.load(path_to_x)
    y = np.load(path_to_y)

    data = np.hstack((x, y[:, np.newaxis]))

    average_verwirrungsgrad = [
        verwirrungsgrad_computation.average_verwirrungsgrad(
            sequence=data[sample, :],
            model=intrusion_detector,
        )
        for sample in range(data.shape[0])
    ]

    verwirrungsgrad = np.array(average_verwirrungsgrad)

    np.save(verwirrungsgrad, output_path)


@intrusion_detection.command()
def evaluate_verwirrungsgrad_for_fake_data(
    path_to_model: str, path_to_label_encoder: str, output_path: str
):
    with open(path_to_model, "rb") as model_handle:
        intrusion_detector = pickle.load(model_handle)

    with open(path_to_label_encoder, "rb") as label_encoder_handle:
        label_encoder = pickle.load(label_encoder_handle)

    sequences = process_data.extract_sequences(
        data=load_data.load_raw_data(path_to_data=constants.ATTACK_DATA_PATH),
    )

    sequences_as_labels = [label_encoder.transform(sequence) for sequence in sequences]

    average_verwirrungsgrad = [
        verwirrungsgrad_computation.average_verwirrungsgrad(
            sequence=sequence,
            model=intrusion_detector,
        )
        for sequence in sequences_as_labels
    ]

    verwirrungsgrad = np.array(average_verwirrungsgrad)

    np.save(
        output_path,
        verwirrungsgrad,
    )


if __name__ == "__main__":
    intrusion_detection()
