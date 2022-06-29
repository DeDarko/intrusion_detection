import typer
from intrusion_detection.data_parsing import load_data, process_data
from intrusion_detection import constants

intrusion_detection = typer.Typer()


@intrusion_detection.command()
def preprocess_data(target_directory: str):
    load_data.store_dataframe(
        process_data.remove_sequence_min_occurences(
            sequences=process_data.extract_sequences(
                data=load_data.load_raw_data(path_to_data=constants.REAL_DATA_PATH),
            ),
            min_occurences=constants.MINIMAL_SEQUENCE_LENGTH,
        ),
        target_directory=target_directory,
    )


if __name__ == "__main__":
    intrusion_detection()