import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

filename = "poe_hp.txt"
text = open(filename, 'r', encoding='utf-8').read().lower()
text = text[:55000]

chars = list(set(text))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

seq_length = 150
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
#         self.num_chars = n_chars
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

n_epochs = 40
batch_size = 128
model = CharModel().to(device)  # Move model to the device

optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)

best_model = None
best_loss = np.inf

for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Move data to device
            y_pred = model(X_batch)
            loss += loss_fn(y_pred, y_batch)
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))


torch.save([best_model, char_to_int], "single-char-short.pth")
