import tensorflow as tf
 
# Load preprocessed data

dataset = tf.data.Dataset.load('preprocessed_data_path')
 
# Define the model

model = tf.keras.Sequential([...])
 
# Train the model incrementally

for batch in dataset:

    model.fit(batch)

 