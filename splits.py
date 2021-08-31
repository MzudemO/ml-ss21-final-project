import os
import shutil
import random

DATASET_PATH = os.environ["DATASET_PATH"]

random.seed(4505918)


def save_to_folder(filenames, split, tag):
    """
    Copies files to `/split/tag/filename.png`

    Parameters:
        filenames (list(str)):  List of filenames
        split (str):            Split designation. Examples: "train", "test"
        tag (str):              Genre tag
    """
    folder_path = DATASET_PATH + split + "/" + tag
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for filename in filenames:
        src_path = DATASET_PATH + tag + "/" + filename
        dest_path = folder_path + "/" + filename
        shutil.copy(src_path, dest_path)


def train_test_split(train_fraction=0.8):
    """
    Splits data into train and test and copies to `/train` and `/test` directories.
    
    Shuffles data, preserves distribution of data over labels.

    Parameters:
        train_fraction (float): Fraction of the data to train on.

    Returns:
        None
    """
    files = os.listdir(DATASET_PATH)
    print(files)
    for file in files:
        if os.path.isdir(DATASET_PATH + file):
            paths = os.listdir(DATASET_PATH + file)
            random.shuffle(paths)
            split_index = round(len(paths) * train_fraction)
            print(split_index)
            train = paths[:split_index]
            test = paths[split_index:]
            print(len(train))
            print(len(test))

            save_to_folder(train, "train", file)
            save_to_folder(test, "test", file)


if __name__ == "__main__":
    train_test_split()

