import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback, ModelCheckpoint



file_path = "poem_texts.txt"
seed_text = "Once upon a time"

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read().lower()

tokenizer = Tokenizer(filters='!"#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index
print(word_index)
word_counts = len(word_index) + 1

input_sequence = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequence.append(n_gram_sequence)

max_seq_len = max([len(seq) for seq in input_sequence])
input_sequences = np.array(
    tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_seq_len, padding='pre'))

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=word_counts)

model = Sequential()
model.add(Embedding(word_counts, 100, input_length=max_seq_len-1))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(word_counts, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


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
class TextGenerationCallback(Callback):
    def __init__(self, tokenizer, max_sequence_len, seed_text, num_words, temperature):
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.seed_text = seed_text
        self.num_words = num_words
        self.temperature = temperature

    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch: {epoch}')
        generated_text = generate_text(self.model, self.tokenizer, self.max_sequence_len, self.seed_text, self.num_words, self.temperature)
        print(f'Generated text: {generated_text}')

text_gen_callback = TextGenerationCallback(tokenizer, max_seq_len, seed_text, num_words=50, temperature=0.7)

model.fit(X, y, epochs=60, verbose=1, callbacks=[text_gen_callback])
tf.keras.models.save_model(model, "poem_word.h5")
