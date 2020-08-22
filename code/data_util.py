import numpy as np
from skimage import img_as_float32


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


def preprocess(arr: np.ndarray) -> np.ndarray:
    """
    Normalize data range to [0, 1] in float32, and add one new axis at the end.

    Args:
        arr: N-dim array
            Array to be processed.

    Returns: processed array.
    """
    return img_as_float32(arr)[..., np.newaxis]
