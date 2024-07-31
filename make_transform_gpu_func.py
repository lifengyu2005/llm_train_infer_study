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

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#添加了 MultiHeadAttention 类来实现多头注意力机制。
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out(out)
#创建了自定义的 TransformerDecoderLayer 类，使用基础组件（如 MultiHeadAttention 和前馈网络）构建。
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # Self Attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Cross Attention
        if memory is not None:
            attn_output = self.cross_attn(x, memory, memory, memory_mask)
            x = x + self.dropout(attn_output)
            x = self.norm2(x)
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        return x
#3. 实现了 TransformerDecoder 类，它堆叠多个 TransformerDecoderLayer。
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x
    


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
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        # 保持输入为长整型
        x = x.long().to(device)
        memory = memory.long()
        memory = memory.long().to(device)

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
model = TransformerDecoderOnly(vocab_size, embedding_dim, num_heads, num_layers, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.long().to(device)
        targets = targets.long().to(device)

        optimizer.zero_grad()
        output = model(data, data)  # Using data as memory for simplicity
        loss = criterion(output.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

# Save the model
# Create the directory if it doesn't exist
import os

save_dir = '/mnt/data/lifengyu/transformer_test'
os.makedirs(save_dir, exist_ok=True)

# Save the model with a .pt extension
save_path = os.path.join(save_dir, 'transformer_test.pt')
torch.save(model.state_dict(), save_path)
print(f"Model saved as {save_path}")

