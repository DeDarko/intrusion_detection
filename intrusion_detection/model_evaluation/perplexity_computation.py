from typing import List

import numpy as np
from intrusion_detection import constants
from tensorflow import keras


def average_perplexity(sequence: np.ndarray, model: keras.Model) -> float:
    non_padded_sequence = remove_padding(list(sequence))

    total_perplexity = 0

    for current_index in range(len(non_padded_sequence) - 1):
        relevant_sequence = non_padded_sequence[: (current_index + 1)]
        target_event = non_padded_sequence[current_index + 1]
        total_perplexity += compute_single_perplexity(
            sequence=relevant_sequence, actual_event=target_event, model=model
        )

    return total_perplexity / (len(sequence) - 1)


def compute_single_perplexity(sequence: List, actual_event: int, model: keras.Model):
    predicted_event_distribution = create_single_prediction(
        sequence=sequence, model=model
    )
    return np.square(
        predicted_event_distribution - one_hot_encode_class(actual_event)
    ).mean()


def one_hot_encode_class(event: int) -> np.array:
    empty_array = np.full(len(constants.EVENTS_MAP), 0)
    empty_array[event] = 1
    return empty_array


def create_single_prediction(sequence: List, model: keras.Model) -> List:
    return model.predict(np.array(sequence)[np.newaxis, :], verbose=0)[0]


def remove_padding(sequence: List) -> List:
    return list(filter(lambda x: x != 8, sequence))
