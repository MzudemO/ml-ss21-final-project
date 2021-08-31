import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2, ResNet50V2

import matplotlib.pyplot as plt
import numpy as np
import json

# Create train, test, and validation datasets
train_ds = image_dataset_from_directory(
    directory="dataset/train/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(174, 174),
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=4505918,
)

val_ds = image_dataset_from_directory(
    directory="dataset/train/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(174, 174),
    shuffle=True,
    validation_split=0.1,
    subset="validation",
    seed=4505918,
)

test_ds = image_dataset_from_directory(
    directory="dataset/test/",
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(174, 174),
    shuffle=True,
    seed=4505918,
)

label_names = [
    "alternative rock",
    "ambient",
    "blues",
    "classic rock",
    "country",
    "dance",
    "death metal",
    "folk",
    "hard rock",
    "heavy metal",
    "Hip-Hop",
    "house",
    "indie",
    "jazz",
    "k-pop",
    "metalcore",
    "punk",
    "rap",
    "reggae",
    "soul",
    "trance",
]


def one_hot_to_label(vec):
    idx = np.where(vec.numpy() == 1)[0][0]
    return label_names[idx]


# Plot preview
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(one_hot_to_label(labels[i]))
        plt.axis("off")

# Create model from scratch
model = MobileNetV2(
    input_shape=(174, 174, 3), weights=None, include_top=True, classes=21
)


epochs = 50

# Save best checkpoints and stop early to save time
callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}_scratch.h5", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5),
]

# Higher learning rates = fast divergence
# Hyperparameters unoptimized
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Double size of training data and augment it
train_ds = train_ds.repeat(2)

data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        keras.layers.experimental.preprocessing.RandomRotation(0.3),
        keras.layers.experimental.preprocessing.RandomZoom(0.2),
    ]
)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# MobileNet Preprocessing
train_ds.map(
    lambda img, label: (tf.keras.applications.mobilenet_v2.preprocess_input(img), label)
)
val_ds.map(
    lambda img, label: (tf.keras.applications.mobilenet_v2.preprocess_input(img), label)
)

# Train model
history = model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds
)

# Save history for later
with open("history_lr0.001_sgd_train_full_scratch.json", "w") as f:
    json.dump(history.history, f)

# Plot loss graph
plt.plot(history.history["loss"])

# Evaluate
test_ds = test_ds.map(
    lambda img, label: (tf.keras.applications.mobilenet_v2.preprocess_input(img), label)
)
# Optionally: load best checkpoint
model.evaluate(test_ds)

# Confusion matrix
y_pred = model.predict(test_ds)
labels_it = test_ds.unbatch().map(lambda _, y: y).as_numpy_iterator()
y_true = np.array(list(labels_it))

t = tf.convert_to_tensor(list(map(lambda l: np.where(l == 1)[0], y_true)))
p = tf.convert_to_tensor(list(map(lambda l: np.argmax(l), y_pred)))

confusion_matrix = tf.math.confusion_matrix(labels=t, predictions=p).numpy()
confusion_matrix = np.around(
    confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis],
    decimals=2,
)

plt.matshow(confusion_matrix)
