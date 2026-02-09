import tensorflow as tf
import numpy as np

# Load text
with open("data.txt", "r") as f:
    text = f.read()

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create sequences
sequence = tokenizer.texts_to_sequences([text])[0]
input_sequences = []
for i in range(1, len(sequence)):
    input_sequences.append(sequence[:i+1])

max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=max_seq_len, padding='pre'
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 64, input_length=max_seq_len - 1),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
model.fit(X, y, epochs=200, verbose=1)

# Save the model
model.save("genai_text_model.h5")

print("✅ Model training completed and saved as genai_text_model.h5")
