import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PeptideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Encoder(nn.Module):
    def __init__(self, input_dim=20000, hidden1=1024, hidden2=512, context_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, context_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size=23, embed_dim=64, hidden_dim=256, context_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embed_dim + context_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, trg, context):
        # trg: [batch_size, seq_len]
        # context: [batch_size, context_dim]
        embedded = self.embedding(trg) # [batch_size, seq_len, embed_dim]
        
        seq_len = trg.shape[1]
        context_repeated = context.unsqueeze(1).repeat(1, seq_len, 1) # [batch_size, seq_len, context_dim]
        
        rnn_input = torch.cat((embedded, context_repeated), dim=2)
        outputs, _ = self.rnn(rnn_input)
        
        predictions = self.fc_out(outputs) # [batch_size, seq_len, vocab_size]
        return predictions

class BaselineModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        context = self.encoder(src)
        out = self.decoder(trg, context)
        return out
