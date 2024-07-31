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

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer_decoder(x, None)  # Set memory to None for decoder-only model
        x = x.transpose(0, 1)
        x = self.fc_out(x)
        return x


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
model.eval()  # 设置为评估模式

# # 准备输入数据
# input_sequence = torch.randint(0, vocab_size, (1, 10)).to(device)  # 假设输入是一个长度为10的序列

# # 执行推理
# with torch.no_grad():
#     output = model(input_sequence)

# # 处理输出
# predicted_tokens = torch.argmax(output, dim=-1)

# print("Input sequence:", input_sequence)
# print("Predicted tokens:", predicted_tokens)

def generate_multiple_sequences(model, input_sequence, max_length, num_beams=3):
    model.eval()
    with torch.no_grad():
        # 初始化 num_beams 个相同的起始序列
        current_seqs = input_sequence.repeat(num_beams, 1)
        seq_scores = torch.zeros(num_beams, 1).to(input_sequence.device)
        
        for _ in range(max_length - input_sequence.size(1)):
            output = model(current_seqs)
            vocab_size = output.size(-1)
            
            # 计算下一个词的 log 概率
            next_token_logits = output[:, -1, :]
            next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
            
            # 计算每个可能序列的得分
            next_scores = seq_scores.unsqueeze(-1) + next_token_logprobs
            next_scores = next_scores.view(-1)
            
            # 选择 top num_beams 个得分最高的序列
            top_scores, top_indices = next_scores.topk(num_beams, dim=0, largest=True, sorted=True)
            
            # 更新序列和得分
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            new_seqs = []
            for i, beam_idx in enumerate(beam_indices):
                new_seq = torch.cat([current_seqs[beam_idx], token_indices[i].unsqueeze(0)], dim=-1)
                new_seqs.append(new_seq)
            
            current_seqs = torch.stack(new_seqs)
            seq_scores = top_scores.unsqueeze(-1)
        
        # 选择得分最高的序列
        # 修改这里：确保返回的是二维张量
        best_seq = current_seqs[seq_scores.argmax()].unsqueeze(0)
        return best_seq

# 准备输入数据
input_sequence = torch.randint(0, vocab_size, (1, 10)).to(device)  # 输入长度为10

# 执行推理，生成30个token 包含输入的token
output_sequence = generate_multiple_sequences(model, input_sequence, max_length=30, num_beams=3)

print("Input sequence:", input_sequence)
print("Generated sequence:", output_sequence)
print("Input length:", input_sequence.size(1))
print("Output length:", output_sequence.size(1))

# 如果你想打印生成的序列内容，可以这样做：
##print("Generated tokens:", output_sequence.squeeze().tolist())