import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

import autoencoder
import data_util


# paths
checkpoint_fp = "./checkpoints"
figure_fp = "../figures"

# restore model
checkpoint_fp = Path(checkpoint_fp)
with open(checkpoint_fp / "config.json", "r") as f:
    config = json.load(f)

model = autoencoder.conv_ae(
    input_shape=config["input_shape"], latent_dim=config["latent_dim"],
)
model.load_weights(checkpoint_fp / "weights")

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = data_util.preprocess(x_train)
x_test = data_util.preprocess(x_test)

# predict
x_train_pred = model.decoder(model.encoder(x_train))
x_test_pred = model.decoder(model.encoder(x_test))

# plot settings
colormap = "gray"
figure_fp = Path(figure_fp)
imshow_param = dict(cmap=colormap, vmin=0, vmax=1)

# plot training
x_train_shuffled = data_util.shuffle_label(x_train, y_train)
for digit in [2, 3, 4]:
    index = np.argwhere(y_train == digit)[:, 0]
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(6.4, 6.4))

    for row in range(3):
        i = index[row]
        axes[row, 0].imshow(x_train[i, ...], **imshow_param)
        axes[row, 1].imshow(x_train_shuffled[i, ...], **imshow_param)
        axes[row, 2].imshow(x_train_pred[i, ...], **imshow_param)

    for ax in axes.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    axes[0, 0].set_title("input")
    axes[0, 1].set_title("target")
    axes[0, 2].set_title("predict")

    fig.tight_layout()
    plt.savefig(figure_fp / f"train_{digit}.png")
    plt.close()

# plot evaluation
for digit in np.unique(y_test):
    index = np.argwhere(y_test == digit)[:, 0]
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(4.8, 6.4))

    for row in range(3):
        i = index[row]
        axes[row, 0].imshow(x_test[i, ...], **imshow_param)
        axes[row, 1].imshow(x_test_pred[i, ...], **imshow_param)

    for ax in axes.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    axes[0, 0].set_title("input")
    axes[0, 1].set_title("predict")

    fig.tight_layout()
    plt.savefig(figure_fp / f"eval_{digit}.png")
    plt.close()
