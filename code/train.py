import numpy as np
import tensorflow.keras as tfk

import autoencoder


# params
latent_dim = 2
train_params = dict(batch_size=128, epochs=1, validation_split=0.3, shuffle=True)

# load data
(x_train, y_train), (x_test, y_test) = tfk.datasets.mnist.load_data()
x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

# normalize data
a_max = np.iinfo(x.dtype).max
x = np.expand_dims(x, -1).astype("float32") / a_max

# shuffle, preserve label
x2 = np.empty_like(x)
for u in np.unique(y):
    i = np.argwhere(y == u)[:, 0]
    j = i.copy()
    np.random.shuffle(j)
    x2[i, ...] = x[j, ...]

# prep model
model = autoencoder.conv_ae(input_shape=x.shape[1:], latent_dim=latent_dim)
model.compile(optimizer=tfk.optimizers.Adam())

# train, evaluate
model.fit(x=x, y=x2, **train_params)
model.evaluate(x=x, y=x2, batch_size=train_params["batch_size"])
