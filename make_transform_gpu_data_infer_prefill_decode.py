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
# 修改 MultiHeadAttention 类
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
        
        self.stored_k = None
        self.stored_v = None
        
    def forward(self, query, key, value, mask=None, use_stored_kv=False):
        batch_size = query.size(0)
        
        if use_stored_kv and self.stored_k is not None and self.stored_v is not None:
            K = self.stored_k
            V = self.stored_v
            Q = self.q_linear(query)
        else:
            Q = self.q_linear(query)
            K = self.k_linear(key)
            V = self.v_linear(value)
            self.stored_k = K
            self.stored_v = V
        
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

    def get_key_values(self):
        return self.stored_k, self.stored_v

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
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None, use_stored_kv=False):
        attn_output = self.self_attn(x, x, x, tgt_mask, use_stored_kv=use_stored_kv)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        if memory is not None:
            attn_output = self.cross_attn(x, memory, memory, memory_mask)
            x = x + self.dropout(attn_output)
            x = self.norm2(x)
        
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        return x

    def get_key_values(self):
        return self.self_attn.get_key_values()
        
#3. 实现了 TransformerDecoder 类，它堆叠多个 TransformerDecoderLayer。
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        
    def forward(self, x, memory, tgt_mask=None, memory_mask=None, past_key_values=None):
        use_stored_kv = past_key_values is not None
        new_key_values = []
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if use_stored_kv else None
            x = layer(x, memory, tgt_mask, memory_mask, use_stored_kv=use_stored_kv)
            new_key_values.append(layer.get_key_values())
        return x, new_key_values

    def get_key_values(self):
        return [layer.get_key_values() for layer in self.layers]
    

# Transformer Decoder-Only Model
# 定义模型架构（与训练时使用的架构相同）
class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout):
        super(TransformerDecoderOnly, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, embedding_dim))
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_key_values=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        
        # 如果有 past_key_values，说明是 decode 阶段
        if past_key_values is not None:
            x, new_key_values = self.transformer_decoder(x[:, -1:], x, past_key_values=past_key_values)
        else:
            x, new_key_values = self.transformer_decoder(x, x)
        
        x = x.transpose(0, 1)
        x = self.fc_out(x)
        return x, new_key_values


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型参数
vocab_size = 10000
embedding_dim = 512
num_heads = 8
num_layers = 6
dropout = 0.1

# 初始化模型
model = TransformerDecoderOnly(vocab_size, embedding_dim, num_heads, num_layers, dropout).to(device)

# 加载模型权重
model.load_state_dict(torch.load('/mnt/data/lifengyu/transformer_test/transformer_test.pt'))
model.eval()

# 修改 Prefill 阶段
def prefill(model, input_sequence):
    #with torch.no_grad(): 是 PyTorch 中的一个上下文管理器，它的作用是暂时关闭梯度计算。这个语句有以下几个重要的含义和用途：
    with torch.no_grad():
        output, past_key_values = model(input_sequence)
    return output, past_key_values

# 修改 Decode 阶段
def decode(model, input_sequence, past_key_values):
    with torch.no_grad():
        output, new_key_values = model(input_sequence, past_key_values)
    return output, new_key_values

# 执行推理
input_sequence = torch.randint(0, vocab_size, (1, 10)).to(device)
max_new_tokens = 20

# Prefill 阶段
output, past_key_values = prefill(model, input_sequence)
predicted_sequence = input_sequence

# Decode 阶段
for _ in range(max_new_tokens):
    next_token_logits = output[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
    predicted_sequence = torch.cat([predicted_sequence, next_token], dim=-1)
    
    output, past_key_values = decode(model, next_token, past_key_values)

print("Input sequence:", input_sequence)
print("Predicted sequence:", predicted_sequence)
