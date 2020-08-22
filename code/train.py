import json
import shutil
from pathlib import Path

from tensorflow.keras.datasets import mnist

import autoencoder
import data_util


# paths
checkpoint_fp = "./checkpoints"

# params
latent_dim = 2
num_shuffle = 10
train_params = dict(batch_size=128, epochs=1, validation_split=0.3, shuffle=True)
opt = "adam"

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = data_util.preprocess(x_train)
x_test = data_util.preprocess(x_test)

# prep model
input_shape = x_train.shape[1:]
model = autoencoder.conv_ae(input_shape=input_shape, latent_dim=latent_dim)
model.compile(optimizer=opt)

# train, evaluate
print("pre-train eval")
model.evaluate(
    x=x_test, y=data_util.shuffle_label(x_test, y_test), batch_size=train_params["batch_size"],
)

print("train")
for _ in range(num_shuffle):
    model.fit(
        x=x_train, y=data_util.shuffle_label(x_train, y_train), **train_params,
    )

print("post-train eval")
model.evaluate(
    x=x_test, y=data_util.shuffle_label(x_test, y_test), batch_size=train_params["batch_size"],
)

# save model and metadata
checkpoint_fp = Path(checkpoint_fp)
if checkpoint_fp.exists():
    shutil.rmtree(checkpoint_fp)

model.save_weights(checkpoint_fp / "weights")

config = dict(
    input_shape=input_shape,
    latent_dim=latent_dim,
    num_shuffle=num_shuffle,
    train_params=train_params,
    optimizer=opt,
)
with open(checkpoint_fp / "config.json", "w") as f:
    json.dump(config, f)
