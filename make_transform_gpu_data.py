"""

很好的问题！嵌入向量的查找过程实际上非常直观。让我详细解释一下这个过程：
嵌入矩阵结构：
假设我们有一个词汇表大小为 vocab_size 的嵌入矩阵，每个嵌入向量的维度是 embedding_dim。
这个矩阵可以表示为 E ，形状为 (vocab_size, embedding_dim)。
索引映射：
每个词的整数 ID 直接对应于这个矩阵中的一行。
例如，ID 为 0 的词对应第 0 行，ID 为 1 的词对应第 1 行，以此类推。
查找过程：
对于输入 x 中的每个整数 i，查找过程就是简单地选择矩阵 E 的第 i 行。
在 PyTorch 中，这个操作通过高效的索引机制自动完成。
"""


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
        
        #self.q_linear：用于生成查询（Query）向量  输入是embed_dim维度，输出是embed_dim维度
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        
        #self.k_linear：用于生成键（Key）向量
        self.k_linear = nn.Linear(embed_dim, embed_dim)

        #self.v_linear：用于生成值（Value）向量
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.q_linear(query) #  每个线性层执行以下操作：  output = input * weight.T + bias
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
##最基础的网络结构一堆attension 加一些FNN网络
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
        ##是的，您理解得很正确。这行代码确实是在创建一个嵌入矩阵，其中包含 vocab_size 个 embedding_dim 维的向量。具体来说：
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        ###这个感觉定义了一个全0的tensor
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embedding_dim))

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, dim_feedforward=2048, dropout=dropout)
        ###这个是多层放在一个类中
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)
        
        #这行代码定义了一个全连接层（fully connected layer），也称为线性层。它的功能是将transformer decoder的输出映射回词汇表大小的空间。具体解释如下：
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        """
        这行代码的功能是创建一个 dropout 层。用中文解释如下：
这行代码创建了一个 dropout 层，它是一种正则化技术，用于防止神经网络过拟合。具体来说：
1. nn.Dropout(dropout) 创建了一个 dropout 层，其中 dropout 是一个浮点数，表示在训练过程中随机丢弃（设置为零）神经元的概率。
2. 在前向传播过程中，这个 dropout 层会随机"关闭"一部分神经元，迫使网络学习更加鲁棒的特征，不过分依赖于任何特定的神经元。
在测试阶段，dropout 层通常会被自动禁用，所有神经元都会参与计算。
使用 dropout 可以帮助模型更好地泛化，减少过拟合的风险，特别是在处理大型、复杂的神经网络时非常有效。
总的来说，这行代码为模型添加了一个重要的正则化机制，有助于提高模型的泛化能力和鲁棒性。
        """
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory):
        # 数据预处理
        # 保持输入为长整型
        x = x.long().to(device)
        memory = memory.long()
        memory = memory.long().to(device)
        
        # 1. 嵌入层 + 位置编码
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        
        # 类型转换
        # 在这里将 x 和 memory 转换为浮点型
        x = x.float()
        memory = self.embedding(memory).float()
        
        # 2. 维度调整
        # 调整 x 和 memory 的维度
        x = x.transpose(0, 1)  # (batch_size, seq_len, embedding_dim) -> (seq_len, batch_size, embedding_dim)
        memory = memory.transpose(0, 1)  # (batch_size, seq_len, embedding_dim) -> (seq_len, batch_size, embedding_dim)
        
        # 3. Transformer 解码器
        x = self.transformer_decoder(x, memory)
        
        # 4. 维度调整回原始形状
        #将 Transformer 解码器的输出调整回 (batch_size, seq_len, embedding_dim) 形状。
        x = x.transpose(0, 1)  # 转换回 (batch_size, seq_len, embedding_dim)
        
        # 5. 输出层
        #最后通过一个全连接层（self.fc_out）将解码器的输出映射到词汇表大小的空间。
        #这一步生成每个位置上所有可能词的概率分布。
        x = self.fc_out(x)
        return x

# Generate some random data for demonstration
data = np.random.randint(0, vocab_size, (1000, max_seq_length))
targets = np.random.randint(0, vocab_size, (1000, max_seq_length))

# Create DataLoader
dataset = SampleDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
"""
vocab_size = 10000  # Example vocabulary size 词汇表大小
embedding_dim = 512
num_heads = 8
num_layers = 6
dropout = 0.1
max_seq_length = 512
batch_size = 32
learning_rate = 1e-4
num_epochs = 10
"""
# Initialize model, loss function, and optimizer
model = TransformerDecoderOnly(vocab_size, embedding_dim, num_heads, num_layers, dropout).to(device)

"""
nn.CrossEntropyLoss() 是 PyTorch 中的一个损失函数，通常用于多类分类问题。它结合了 nn.LogSoftmax() 和 nn.NLLLoss() 在一个单独的类中。这个损失函数在训练神经网络时非常常用，特别是在处理分类任务时。
主要特点：
1. 它为每个类别计算损失。
它期望原始的、未归一化的分数（logits），而不是 softmax 之后的概率。
3. 它在内部应用 softmax 函数。
它可以自动处理多维输入。
在你的代码中，criterion = nn.CrossEntropyLoss() 创建了一个交叉熵损失函数的实例，用于计算模型输出和目标之间的损失。这个损失函数在训练循环中被用来计算每个批次的损失：
"""
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    """
    model.train() 是一个非常重要的方法调用，用于将模型设置为训练模式。这个调用的主要功能包括：
    启用梯度计算：确保在前向传播过程中计算并存储梯度，这对于反向传播和参数更新是必要的。
    激活 Dropout 层：如果模型中有 Dropout 层，它们会在训练模式下正常工作，随机"丢弃"一些神经元。
    启用 Batch Normalization 的学习：如果模型使用了 Batch Normalization，它会在训练模式下更新运行时的均值和方差统计信息。
    4. 可能影响其他层的行为：某些自定义层可能在训练和评估模式下有不同的行为。
    总的来说，model.train() 确保模型处于正确的状态以进行训练。这与 model.eval() 相对，后者用于将模型设置为评估模式，通常在验证或测试时使用。
    在您的训练循环中，每个 epoch 开始时调用 model.train() 是一个好习惯，确保模型始终处于正确的训练状态，特别是如果在同一脚本中混合了训练和评估代码的情况下
    """
    model.train()
    #因此，这个循环将执行大约32次。最后一个批次可能会小于32，因为1000不能被32整除。
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.long().to(device)
        targets = targets.long().to(device)

        optimizer.zero_grad()
        #每个批次的 data 是一个形状为 (32, 512) 的张量，其中每个元素是一个范围在 [0, vocab_size-1] 之间的整数，代表词汇表中的一个词。
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

