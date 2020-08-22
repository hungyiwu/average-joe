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
xe = np.load(data_fp / "x_test.npy")
ye = np.load(data_fp / "y_test.npy")

# test
xe2 = model.decoder(model.encoder(xe))

# plot settings
colormap = "gray"
figure_fp = Path(figure_fp)

# plot
imshow_param = dict(cmap=colormap, vmin=0, vmax=1)
for digit in np.unique(ye):
    index = np.argwhere(ye == digit)[:, 0]
    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(4.8, 6.4))

    for row in range(3):
        i = index[row]
        axes[row, 0].imshow(xe[i, ...], **imshow_param)
        axes[row, 1].imshow(xe2[i, ...], **imshow_param)

    for ax in axes.reshape(-1):
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    axes[0, 0].set_title("encoder input")
    axes[0, 1].set_title("decoder output")

    fig.tight_layout()
    plt.savefig(figure_fp / f"digit_{digit}.png")
    plt.close()
