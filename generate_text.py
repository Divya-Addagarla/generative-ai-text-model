import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("genai_text_model.h5")

# Load and tokenize text again (same as training)
with open("data.txt", "r") as f:
    text = f.read()

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# Max sequence length (same logic as training)
sequence = tokenizer.texts_to_sequences([text])[0]
max_seq_len = len(sequence)

# Function to generate text
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list], maxlen=max_seq_len - 1, padding='pre'
        )

        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text

# Generate new text
seed = "artificial intelligence"
generated_text = generate_text(seed, 10)

print("\n🧠 Generated Text:")
print(generated_text)
