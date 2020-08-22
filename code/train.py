import json
import shutil
from pathlib import Path

import numpy as np

import autoencoder


# paths
data_fp = "./saved_data"
checkpoint_fp = "./checkpoints"

# params
latent_dim = 2
train_params = dict(batch_size=128, epochs=1, validation_split=0.3, shuffle=True)
opt = "adam"

# load data
data_fp = Path(data_fp)
xt = np.load(data_fp / "x_train.npy")
xt2 = np.load(data_fp / "x_train_shuffled.npy")
xe = np.load(data_fp / "x_test.npy")
xe2 = np.load(data_fp / "x_test_shuffled.npy")

# prep model
input_shape = xt.shape[1:]
model = autoencoder.conv_ae(input_shape=input_shape, latent_dim=latent_dim)
model.compile(optimizer=opt)

# train, evaluate
print("pre-train eval")
model.evaluate(x=xe, y=xe2, batch_size=train_params["batch_size"])

print("train")
model.fit(x=xt, y=xt2, **train_params)

print("post-train eval")
model.evaluate(x=xe, y=xe2, batch_size=train_params["batch_size"])

# save model and metadata
checkpoint_fp = Path(checkpoint_fp)
if checkpoint_fp.exists():
    shutil.rmtree(checkpoint_fp)

ne = train_params["epochs"]
model.save_weights(checkpoint_fp / "weights")

config = dict(
    input_shape=input_shape,
    latent_dim=latent_dim,
    train_params=train_params,
    optimizer=opt,
)
with open(checkpoint_fp / "config.json", "w") as f:
    json.dump(config, f)
