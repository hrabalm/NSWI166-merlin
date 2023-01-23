import pandas as pd
import pathlib
import urllib.request
import shutil
import zipfile

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ARCHIVE_FILENAME = "/nwsi166-merlin/datasets/ml-25m.zip"
DATASETS_DIR = "/nwsi166-merlin/datasets"
ML25_DIR = f"{DATASETS_DIR}/ml-25m"
RATINGS_FILENAME = f"{ML25_DIR}/ratings.csv"
MOVIES_FILENAME = f"{ML25_DIR}/movies.csv"

TRAIN_FILENAME = f"{ML25_DIR}/ml25_train.csv"
TEST_FILENAME = f"{ML25_DIR}/ml25_validation.csv"

def _download_file(url, dest):
    file = urllib.request.urlopen(url)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file, f)

def _download_if_not_exists():
    if not pathlib.Path(ARCHIVE_FILENAME).exists():
        filepath = pathlib.Path(ARCHIVE_FILENAME)
        filepath.parent.mkdir(parents=True, exist_ok=True) # ensure the parent directory exists - might fail if root?
        _download_file(MOVIELENS_URL, str(filepath))

def _extract_movielens():
    if not pathlib.Path(ML25_DIR).exists():
        with zipfile.ZipFile(ARCHIVE_FILENAME) as zf:
            zf.extractall(DATASETS_DIR)

def _split_and_preprocess():
    if not pathlib.Path(RATINGS_FILENAME).exists():
        ratings_df = pd.read_csv(RATINGS_FILENAME).drop("timestamp", axis=1)
        movies_df = pd.read_csv(MOVIES_FILENAME).drop("title", axis=1)
        movies_df["genres"] = movies_df["genres"].str.split("|")

        shuffled = ratings_df.sample(len(ratings_df), replace=False)  # shuffle

        validation_count = int(len(shuffled) * 0.25)
        train_count = len(shuffled) - validation_count

        train = ratings_df[:train_count]
        validation = ratings_df[train_count:]

        train.to_csv(TRAIN_FILENAME, index=False)
        validation.to_csv(TEST_FILENAME, index=False)

def prepare_datasets():
    _download_if_not_exists()
    _extract_movielens()
    _split_and_preprocess()

if __name__ == "__main__":
    prepare_datasets()
