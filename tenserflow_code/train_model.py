import tensorflow as tf

dataset = tf.data.Dataset.load('preprocessed_data_path')
model = tf.keras.Sequential([...])

for batch in dataset:

    model.fit(batch)

 
