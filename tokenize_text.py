import tensorflow as tf

# Read the dataset
with open("data.txt", "r") as f:
    text = f.read()

print("Dataset text:")
print(text)

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])

# Convert text to sequence
sequence = tokenizer.texts_to_sequences([text])[0]

print("\nTokenized sequence:")
print(sequence)

print("\nVocabulary size:", len(tokenizer.word_index) + 1)
