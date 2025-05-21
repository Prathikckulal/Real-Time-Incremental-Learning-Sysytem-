# import tensorflow as tf

# dataset = tf.data.Dataset.load('preprocessed_data_path')
# model = tf.keras.Sequential([...])

# for batch in dataset:

#     model.fit(batch)

import tensorflow as tf

dataset = tf.data.Dataset.load('preprocessed_data_path')
val_dataset = tf.data.Dataset.load('preprocessed_val_data_path')


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


batch_size = 32
dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=10000).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# Define the model (layers unspecified in original code, placeholder used)
model = tf.keras.Sequential([
    # Add your layers here, e.g.,
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model with optimizer, loss, and metrics
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training loop with multiple epochs and validation
num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    # Train on batches
    for batch in dataset:
        model.train_on_batch(batch)  # Corrected from model.fit(batch)
    
    # Evaluate on validation dataset
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
