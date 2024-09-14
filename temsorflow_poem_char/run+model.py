import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model("poem_char.h5")

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, seqlen, chars, char_indices, indices_char, seed=None, length=1000, temperature=1.0):
    seed = seed.rjust(seqlen)
    generated = seed

    for _ in range(length):
        x_pred = np.zeros((1, seqlen, len(chars)), dtype=np.float32)
        for t, char in enumerate(seed):
            if char in char_indices:
                x_pred[0, t, char_indices[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = indices_char[next_index]
        generated += next_char
        seed = seed[1:] + next_char

    return generated


with open('poem_texts.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

chars = sorted(list(set(text)))
char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}
seqlen = 100

seed_text = "Once upon a time"

generated_text = generate_text(model, seqlen, chars, char_indices, indices_char, seed=seed_text, length=1000, temperature=0.5)
print(generated_text)
