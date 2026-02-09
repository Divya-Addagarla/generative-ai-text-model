import tensorflow as tf
import numpy as np

# Load text
with open("data.txt", "r") as f:
    text = f.read()

# Tokenize
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Convert to sequence
sequence = tokenizer.texts_to_sequences([text])[0]

# Create input sequences
input_sequences = []
for i in range(1, len(sequence)):
    input_sequences.append(sequence[:i+1])

# Pad sequences
max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    input_sequences, maxlen=max_seq_len, padding='pre'
)

# Split into X and y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# One-hot encode output
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print("Input shape:", X.shape)
print("Output shape:", y.shape)
print("Vocabulary size:", total_words)
