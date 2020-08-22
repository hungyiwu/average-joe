import json
import shutil
from pathlib import Path

import numpy as np

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


def shuffle_label(arr: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    Shuffle array while preserving labels.

    Args:
        arr: N-dim array
            Array to shuffle by dimension zero.
        label: 1-dim array
            Label to preserve.

    Return: np.ndarray, same shape and dtype as input array.
    """
    arr2 = np.empty_like(arr)
    for u in np.unique(label):
        i = np.argwhere(label == u)[:, 0]
        j = i.copy()
        np.random.shuffle(j)
        arr2[i, ...] = arr[j, ...]
    return arr2


# params
test_split = 0.3
out_fp = "./saved_data"

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# normalize data
a_max = np.iinfo(x.dtype).max
x = np.expand_dims(x, -1).astype("float32") / a_max

# split
xt, xe, yt, ye = train_test_split(x, y, test_size=test_split)

# shuffle, preserve label
xt2 = shuffle_label(xt, yt)
xe2 = shuffle_label(xe, ye)

# save
out_fp = Path(out_fp)
if out_fp.exists():
    shutil.rmtree(out_fp)
out_fp.mkdir()

np.save(out_fp / "x_train.npy", xt)
np.save(out_fp / "x_train_shuffled.npy", xt2)
np.save(out_fp / "x_test.npy", xe)
np.save(out_fp / "x_test_shuffled.npy", xe2)

with open(out_fp / "config.json", "w") as f:
    json.dump(dict(test_split=test_split), f)
