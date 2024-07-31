import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Hyperparameters
vocab_size = 10000  # Example vocabulary size
embedding_dim = 512
num_heads = 8
num_layers = 6
dropout = 0.1
max_seq_length = 512
batch_size = 32
learning_rate = 1e-4
num_epochs = 10

# Sample Dataset
class SampleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Transformer Decoder-Only Model
class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout):
        super(TransformerDecoderOnly, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embedding_dim))
        decoder_layer = nn.TransformerDecoderLayer(embedding_dim, num_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        # 保持输入为长整型
        x = x.long()
        memory = memory.long()

        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # 在这里将 x 和 memory 转换为浮点型
        x = x.float()
        memory = self.embedding(memory).float()
        
        # 调整 x 和 memory 的维度
        x = x.transpose(0, 1)  # (batch_size, seq_len, embedding_dim) -> (seq_len, batch_size, embedding_dim)
        memory = memory.transpose(0, 1)  # (batch_size, seq_len, embedding_dim) -> (seq_len, batch_size, embedding_dim)
        
        x = self.transformer_decoder(x, memory)
        x = x.transpose(0, 1)  # 转换回 (batch_size, seq_len, embedding_dim)
        x = self.fc_out(x)
        return x

# Generate some random data for demonstration
data = np.random.randint(0, vocab_size, (1000, max_seq_length))
targets = np.random.randint(0, vocab_size, (1000, max_seq_length))

# Create DataLoader
dataset = SampleDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = TransformerDecoderOnly(vocab_size, embedding_dim, num_heads, num_layers, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.long()
        targets = targets.long()

        optimizer.zero_grad()
        output = model(data, data)  # Using data as memory for simplicity
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), '/mnt/data/lifengyu/transformer_test')
print("Model saved as /mnt/data/lifengyu/transformer_test")
