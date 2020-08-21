import numpy as np
import tensorflow.keras as tfk

import autoencoder


# params
latent_dim = 2
train_params = dict(batch_size=128, epochs=1, validation_split=0.3, shuffle=True)

# load data
(x_train, _), (x_test, _) = tfk.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
a_max = np.iinfo(mnist_digits.dtype).max
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / a_max

# prep model
model = autoencoder.conv_ae(input_shape=mnist_digits.shape[1:], latent_dim=latent_dim)
model.compile(optimizer=tfk.optimizers.Adam())
model.fit(
        x=mnist_digits,
        y=mnist_digits,
        **train_params,
        )
model.evaluate(
        x=mnist_digits,
        y=mnist_digits,
        batch_size=train_params["batch_size"],
        )
