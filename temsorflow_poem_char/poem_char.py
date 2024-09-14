import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import random

with open('poem_texts.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

print(f'Corpus length: {len(text)} characters')


chars = sorted(list(set(text)))
print(f'Total unique characters: {len(chars)}')
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

LEN_SEQ = 100
step = 3
n_epoch = 5
batch_size = 128
temperature = 0.5

sentences = []
next_chars = []

for i in range(0, len(text) - LEN_SEQ, step):
    sentences.append(text[i: i + LEN_SEQ])
    next_chars.append(text[i + LEN_SEQ])

print(f'Number of sequences: {len(sentences)}')

X = np.zeros((len(sentences), LEN_SEQ, len(chars)), dtype=np.float32)
y = np.zeros((len(sentences), len(chars)), dtype=np.float32)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_index[char]] = 1.0
    y[i, char_to_index[next_chars[i]]] = 1.0

model = Sequential()
model.add(LSTM(128, input_shape=(LEN_SEQ, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, text, maxlen, chars, char_to_index, index_to_char, seed=None, length=1000, temperature=1.0):
    if seed is None:
        start_index = random.randint(0, len(text) - maxlen - 1)
        seed = text[start_index: start_index + maxlen]
    else:
        if len(seed) < maxlen:
            seed = seed.rjust(maxlen)
        else:
            seed = seed[-maxlen:]
    generated = seed
    print('----- Generating with seed: "' + seed + '"')
    for _ in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)), dtype=np.float64)
        for t, char in enumerate(seed):
            if char in char_to_index:
                x_pred[0, t, char_to_index[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        generated += next_char
        seed = seed[1:] + next_char
    return generated

class TextGenerationCallback(tf.keras.callbacks.Callback):
    def __init__(self, text, LEN_SEQ, chars, char_to_index, index_to_char, temperature):
        self.text = text
        self.LEN_SEQ = LEN_SEQ
        self.chars = chars
        self.char_to_index = char_to_index
        self.index_to_char = index_to_char
        self.temperature = temperature

    def on_epoch_end(self, epoch, log=None):
        print()
        print('Epoch: %d\n' % epoch)
        start_index = random.randint(0, len(self.text) - self.LEN_SEQ - 1)
        seed_sentence = self.text[start_index: start_index + self.LEN_SEQ]
        print('----- Seed: "' + seed_sentence + '"\n')
        generated = seed_sentence

        for _ in range(400):
            x_pred = np.zeros((1, self.LEN_SEQ, len(self.chars)), dtype=np.float64)
            for t, char in enumerate(seed_sentence):
                if char in self.char_to_index:
                    x_pred[0, t, self.char_to_index[char]] = 1.0
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature=self.temperature)
            next_char = self.index_to_char[next_index]
            generated += next_char
            seed_sentence = seed_sentence[1:] + next_char
        print(generated)
        print()

text_gen_callback = TextGenerationCallback(text, LEN_SEQ, chars, char_to_index, index_to_char, temperature)

model.fit(X, y,
          batch_size=batch_size,
          epochs=n_epoch,
          callbacks=[text_gen_callback])

tf.keras.models.save_model(model, "poem_char_e5.h5")

