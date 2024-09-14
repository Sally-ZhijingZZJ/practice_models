import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
model = tf.keras.models.load_model("poem_word.h5")


file_path = "poem_texts.txt"
temperature = 0.5
seed_text = "Once upon a time"
num_words = 100

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read().lower()
tokenizer = Tokenizer(filters='!"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index
word_counts = len(word_index) + 1

input_sequence = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequence.append(n_gram_sequence)
max_seq_len = max([len(seq) for seq in input_sequence])

def generate_text(model, tokenizer, max_sequence_len, seed_text, num_words, temperature=1.0):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]

        preds = np.log(predicted_probs + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(len(preds), p=preds)
        next_word = tokenizer.index_word[next_index]

        seed_text += " " + next_word

    return seed_text


generated_text = generate_text(model, tokenizer, max_seq_len, seed_text, num_words,
                               temperature)

print(generated_text)