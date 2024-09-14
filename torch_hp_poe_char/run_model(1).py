import numpy as np
import torch
import torch.nn as nn

best_model, char_to_int = torch.load("single-char-short.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())


class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(self.dropout(x))
        return x
# class CharModel(nn.Module):
#     def __init__(self, embedding_dim = 1, hidden_dim=256, num_layers=1):
#         super().__init__()
#         self.num_chars = 10000
#         self.embedding = nn.Embedding(self.num_chars, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, self.num_chars)
#
#     def forward(self, x):
#         x = x.long()  # Convert float tensor to long for embedding
#         x = self.embedding(x.squeeze(-1))  # Squeeze the last dimension for embedding
#         x, _ = self.lstm(x)  # LSTM output
#         x = self.fc(x[:, -1, :])  # Take output of the last time step
#         return x

model = CharModel()
model.load_state_dict(best_model)

filename = "poe_hp.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text) - seq_length)
prompt = raw_text[start:start + seq_length]
pattern = [char_to_int[c] for c in prompt]

model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        prediction = model(x)
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")
        pattern.append(index)
        pattern = pattern[1:]
