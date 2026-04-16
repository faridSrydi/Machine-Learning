import tensorflow as tf
from tensorflow.keras import layers

# Bangun Deep Q-Network (DQN)
model = tf.keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(4,)),
    layers.Dense(24, activation='relu'),
    layers.Dense(2, activation='linear')  # Output: Q-values untuk 2 actions
])

# Loss function (Huber loss) dan optimizer
model.compile(optimizer='adam', loss=tf.keras.losses.Huber())

# Contoh: ringkasan model
model.summary()
