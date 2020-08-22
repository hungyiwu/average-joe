import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import autoencoder


# paths
data_fp = "./saved_data"
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
data_fp = Path(data_fp)
xt = np.load(data_fp / "x_train.npy")
xt_shuffled = np.load(data_fp / "x_train_shuffled.npy")
yt = np.load(data_fp / "y_train.npy")
xe = np.load(data_fp / "x_test.npy")
ye = np.load(data_fp / "y_test.npy")

# predict
xt_pred = model.decoder(model.encoder(xt))
xe_pred = model.decoder(model.encoder(xe))

# plot settings
colormap = "gray"
figure_fp = Path(figure_fp)
imshow_param = dict(cmap=colormap, vmin=0, vmax=1)

# plot training
for digit in [2, 3, 4]:
    index = np.argwhere(yt == digit)[:, 0]
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(6.4, 6.4))

    for row in range(3):
        i = index[row]
        axes[row, 0].imshow(xt[i, ...], **imshow_param)
        axes[row, 1].imshow(xt_shuffled[i, ...], **imshow_param)
        axes[row, 2].imshow(xt_pred[i, ...], **imshow_param)

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
for digit in np.unique(ye):
    index = np.argwhere(ye == digit)[:, 0]
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(4.8, 6.4))

    for row in range(3):
        i = index[row]
        axes[row, 0].imshow(xe[i, ...], **imshow_param)
        axes[row, 1].imshow(xe_pred[i, ...], **imshow_param)

    for ax in axes.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    axes[0, 0].set_title("input")
    axes[0, 1].set_title("predict")

    fig.tight_layout()
    plt.savefig(figure_fp / f"eval_{digit}.png")
    plt.close()
