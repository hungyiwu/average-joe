import json
from pathlib import Path

import numpy as np

import autoencoder


# paths
data_fp = "./saved_data"
checkpoint_fp = "./checkpoints"

# restore model
checkpoint_fp = Path(checkpoint_fp)
with open(checkpoint_fp / "config.json", "r") as f:
    config = json.load(f)

model = autoencoder.conv_ae(
    input_shape=config["input_shape"], latent_dim=config["latent_dim"],
)
model.load_weights(checkpoint_fp / "weights")
model.compile(optimizer=config["optimizer"])

# load data
data_fp = Path(data_fp)
xt = np.load(data_fp / "x_train.npy")
xt2 = np.load(data_fp / "x_train_shuffled.npy")
xe = np.load(data_fp / "x_test.npy")
xe2 = np.load(data_fp / "x_test_shuffled.npy")

# test
print("restored eval")
model.evaluate(x=xe, y=xe2, batch_size=config["train_params"]["batch_size"])
