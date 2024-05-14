import tensorflow as tf
from tensorflow.keras import layers, models , Input

# Assuming you have your data loaded in `train_images`, `train_labels`, `test_images`, `test_labels`
# For demonstration, let's create dummy data
train_images = tf.random.normal([1000, 28, 28, 1])
train_labels = tf.random.uniform([1000], maxval=10, dtype=tf.int32)
test_images = tf.random.normal([100, 28, 28, 1])
test_labels = tf.random.uniform([100], maxval=10, dtype=tf.int32)

# Assuming your input images are 28x28 pixels with 1 channel (grayscale)
input_layer = Input(shape=(28, 28, 1))

# Then, use input_layer as the first layer in your model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
