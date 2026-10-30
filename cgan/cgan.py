import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation, Flatten,
    LeakyReLU, Dropout, Concatenate, Input
)
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# Settings for visualization
sns.set_theme(rc={"axes.facecolor": "#EDE9DE", "figure.facecolor": "#D8CA7E"})

# Preprocessing CPU threading settings
total_cores = tf.config.threading.get_inter_op_parallelism_threads()
max_threads = int(total_cores * 0.8)
tf.config.threading.set_intra_op_parallelism_threads(max_threads)
tf.config.threading.set_inter_op_parallelism_threads(max_threads)

# Dataset path
data_path = "./DATA2/"
labels_path = os.path.join(data_path, "labels.csv")

# Load labels.csv
labels_df = pd.read_csv(labels_path)
file_paths = labels_df['pth'].values
labels = labels_df['label'].values

# Encode labels and convert to one-hot encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = to_categorical(encoded_labels)

# Prepare TensorFlow dataset
def load_image(file_path, label):
    full_path = tf.strings.join([data_path, file_path], separator='/')
    img = tf.io.read_file(full_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [48, 48])  # Change to 48x48 resolution
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    return img, label

# TensorFlow Dataset oluşturma
dataset = tf.data.Dataset.from_tensor_slices((file_paths, one_hot_labels))
dataset = dataset.map(lambda x, y: load_image(x, y), num_parallel_calls=tf.data.AUTOTUNE).batch(256)

# Generator Model
def build_generator(latent_dim, num_classes):
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(num_classes,))

    # Concatenate noise and label input
    x = Concatenate()([noise_input, label_input])
    x = Dense(3 * 3 * 128, activation="relu")(x)  # Reduced filters for speed
    x = Reshape((3, 3, 128))(x)
    x = UpSampling2D()(x)  # 3x3 -> 6x6
    x = Conv2D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)
    x = UpSampling2D()(x)  # 6x6 -> 12x12
    x = Conv2D(64, kernel_size=3, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)
    x = UpSampling2D()(x)  # 12x12 -> 24x24
    x = Conv2D(32, kernel_size=3, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)
    x = UpSampling2D()(x)  # 24x24 -> 48x48
    x = Conv2D(3, kernel_size=3, padding="same")(x)
    x = Activation("tanh")(x)  # Use tanh for [-1, 1] output

    return Model([noise_input, label_input], x, name="generator")

# Discriminator Model
def build_discriminator(img_shape, num_classes):
    img_input = Input(shape=img_shape)
    label_input = Input(shape=(num_classes,))

    # Embed and reshape labels
    label_embedding = Dense(48 * 48 * 1)(label_input)
    label_embedding = Reshape((48, 48, 1))(label_embedding)

    # Concatenate image and label
    x = Concatenate(axis=-1)([img_input, label_embedding])
    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)  # Added Dropout
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)  # Added Dropout
    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)  # Added Dropout
    x = Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)  # Added Dropout
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model([img_input, label_input], x, name="discriminator")

# Conditional GAN Class
class ConditionalGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        real_images, real_labels = data
        batch_size = tf.shape(real_images)[0]

        # Generate fake images
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator([random_latent_vectors, real_labels])

        # Combine real and fake images
        combined_images = tf.concat([real_images, fake_images], axis=0)
        combined_labels = tf.concat([real_labels, real_labels], axis=0)
        labels = tf.concat([
            tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))
        ], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))  # Add noise

        # Train discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator([combined_images, combined_labels])
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Train generator
        misleading_labels = tf.ones((batch_size, 1))
        with tf.GradientTape() as tape:
            predictions = self.discriminator([
                self.generator([random_latent_vectors, real_labels]), real_labels
            ])
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

# Hyperparameters
latent_dim = 128
num_classes = len(label_encoder.classes_)
generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator((48, 48, 3), num_classes)

# Compile and Train Conditional GAN
cgan = ConditionalGAN(generator, discriminator, latent_dim)
cgan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="g_loss",
    patience=20,
    restore_best_weights=True,
    mode="min"
)


# Train the model
cgan.fit(dataset, epochs=50, callbacks=[early_stopping])

# Generate Images
def generate_images(generator, label_encoder, num_images=5):
    for label_idx in range(num_classes):
        random_latent_vectors = tf.random.normal(shape=(num_images, latent_dim))
        one_hot_labels = tf.convert_to_tensor(to_categorical([label_idx] * num_images, num_classes=num_classes), dtype=tf.float32)

        generated_images = generator([random_latent_vectors, one_hot_labels])
        generated_images = ((generated_images + 1.0) * 127.5).numpy().astype(np.uint8)  # Rescale to [0, 255]

        plt.figure(figsize=(15, 15))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(generated_images[i])
            plt.title(label_encoder.inverse_transform([label_idx])[0])
            plt.axis("off")
        plt.show()

# Generate and display images
generate_images(generator, label_encoder, num_images=5)
