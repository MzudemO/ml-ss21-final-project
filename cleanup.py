import os
import collections
import json
import glob

DATASET_PATH = os.environ["DATASET_PATH"]


def find_duplicates():
    """
    Save duplicate filenames as JSON.

    Returns:
        None
    """
    fnames = []
    files = os.listdir(DATASET_PATH)
    print(files)
    for file in files:
        if os.path.isdir(DATASET_PATH + file):
            paths = os.listdir(DATASET_PATH + file)
            fnames.extend(paths)

    print(f"Number of files: {len(fnames)}")
    # Number of files: 21150
    print(f"Number of unique files: {len(set(fnames))}")
    # Number of unique files: 19532

    duplicated = [v for v, count in collections.Counter(fnames).items() if count > 1]
    with open("duplicates.json", "w") as f:
        json.dump(duplicated, f)


def deduplicate():
    """
    Deduplicate dataset by filename.
    Requires a `duplicates.json` file.

    Retains the first tag a filename occurs in, deletes further occurences.
    Does not prevent duplicate images with different filenames.

    Returns:
        None
    """
    duplicates = []

    with open("duplicates.json", "r") as f:
        duplicates = json.load(f)

    for d in duplicates:
        path = f"{DATASET_PATH}*/{d}".replace("[", "[[]")
        files = glob.glob(path)
        print(files)
        if len(files) > 1:
            for f in files[1:]:
                os.remove(f)


if __name__ == "__main__":
    find_duplicates()
    deduplicate()
