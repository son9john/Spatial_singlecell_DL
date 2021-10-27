# %% codecell
inChannel = 1
x=dataset_train[0]
x, y = x.shape[1], x.shape[2]
input_img = tf.keras.layers.Input(shape = (x, y, inChannel))

# %% codecell
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))

        self.ct1 = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.ct2 = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.ct3 = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.ct3 = layers.Conv2DTranspose(1, (3, 3), activation='relu', padding='same', strides=(2, 2))

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.c3(x)

        x = self.ct1(x)
        x = self.ct2(x)
        x = self.ct3(x)
        return x

# %%
def autoencoder(cropped):
    #encoder
    h = Conv2D(8, (3, 3), activation='relu', padding='same', strides=(2, 2))(cropped)
    h = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)
    h = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)

    # decoder
    h = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)
    h = Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)
    h = Conv2DTranspose(1, (3, 3), activation='relu', padding='same', strides=(2, 2))(h)
    return h

# %% codecell
autoencoder = tf.keras.Model(input_img, autoencoder(input_img))
autoencoder = AutoEncoder()
autoencoder.compile(loss='mean_squared_error', optimizer = keras.optimizers.Adam(learning_rate=0.0003))
# %% codecell
autoencoder.summary()
# %% codecell
train_X=[d.numpy() for d in dataset_train]
train_X = np.stack(train_X, axis=0)
test_X=[d.numpy() for d in dataset_test]
test_X = np.stack(test_X, axis=0)

# %%
autoencoder_train = autoencoder.fit(train_X,
                                    train_X,
                                    batch_size=cfg.train.batch_size,
                                    epochs=cfg.train.epoch,
                                    verbose=1,
                                    validation_data=(test_X, test_X))
# %%
autoencoder.summary()
