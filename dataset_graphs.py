import os
import matplotlib.pyplot as plt
import numpy as np

# DATASET_PATH = os.environ["DATASET_PATH"] + "train/"
DATASET_PATH = "/mnt/g/machine-learning/dataset/train/"


def training_distribution():
    counts = []
    files = os.listdir(DATASET_PATH)
    for file in files:
        if os.path.isdir(DATASET_PATH + file):
            counts.append(len(os.listdir(DATASET_PATH + file)))

    counts = np.array(counts)

    print(f"Least samples: {files[np.argmin(counts)]} {min(counts)}")
    print(f"Least samples: {files[np.argmax(counts)]} {max(counts)}")

    plt.bar(range(1, 22), counts)
    plt.title("Class distribution in training data")
    plt.ylabel("Number of samples")
    plt.xlabel("Class index")
    plt.show()


if __name__ == "__main__":
    training_distribution()
