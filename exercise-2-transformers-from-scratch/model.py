import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_size)


class PositionalEncoding(nn.Module):

    def __init__(self, embed_size, seq_len=5000, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, embed_size) to hold the positional encodings
        pe = torch.zeros(seq_len, embed_size)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        # Apply the sine function to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cosine function to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        pe = pe.unsqueeze(0)  # (1, seq_len, embed_size)

        # Register pe as a buffer so that it is not considered a model parameter, but will be saved and loaded with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape (batch_size, seq_len, embed_size)
        seq_len = x.size(1)
        x = x + (self.pe[:, :seq_len]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, embed_size, hidden_size, dropout):
        super().__init__()
        self.linear1 = nn.Linear(embed_size, hidden_size) # W1 and B1
        self.linear2 = nn.Linear(hidden_size, embed_size) # W2 and B2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size 
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        self.query_linear = nn.Linear(embed_size, embed_size) # Wq and Bq
        self.key_linear = nn.Linear(embed_size, embed_size) # Wk and Bk
        self.value_linear = nn.Linear(embed_size, embed_size) # Wv and Bv
        self.fc_out = nn.Linear(embed_size, embed_size) # Wo and Bo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def scaled_dot_product_attention(Q, K, V, mask=None, dropout=nn.Dropout):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (N, num_heads, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)  # (N, num_heads, seq_len_q, seq_len_k)

        if dropout is not None:
            attention_weights = dropout(attention_weights)  # Apply dropout to attention weights

        output = torch.matmul(attention_weights, V)  # (N, num_heads, seq_len_q, head_dim)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        N = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]

        # Linear projections
        Q = self.query_linear(query)  # (N, seq_len_q, embed_size)
        K = self.key_linear(key)      # (N, seq_len_k, embed_size)
        V = self.value_linear(value)  # (N, seq_len_v, embed_size)

        # Split into multiple heads
        Q = Q.view(N, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_len_q, head_dim)
        K = K.view(N, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_len_k, head_dim)
        V = V.view(N, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)  # (N, num_heads, seq_len_v, head_dim)

        # Scaled dot-product attention
        x, self.attention_scores = self.scaled_dot_product_attention(Q, K, V, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(N, -1, self.embed_size)  # (N, seq_len_q, embed_size)

        return self.fc_out(x)  # (N, seq_len_q, embed_size)


class ResidualConnection(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Self-attention sublayer
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        # Feed-forward sublayer
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, self_attention, cross_attention, feed_forward, dropout):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self-attention sublayer
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        # Cross-attention sublayer
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, enc_output, enc_output, src_mask))
        # Feed-forward sublayer
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, embed_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)  # Apply log softmax for numerical stability


class Transformer(nn.Module):
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos_enc, tgt_pos_enc, projection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos_enc = src_pos_enc
        self.tgt_pos_enc = tgt_pos_enc
        self.projection = projection

    def encode(self, src, src_mask):
        src_embedded = self.src_embed(src)
        src_pos_encoded = self.src_pos_enc(src_embedded)
        return self.encoder(src_pos_encoded, src_mask)

    def decode(self, enc_output, src_mask, tgt, tgt_mask):
        tgt_embedded = self.tgt_embed(tgt)
        tgt_pos_encoded = self.tgt_pos_enc(tgt_embedded)
        return self.decoder(tgt_pos_encoded, enc_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection(x)


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1):
    # Create the embedding layers
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    # Create the positional encoding layers
    src_pos_enc = PositionalEncoding(d_model, seq_len=src_seq_len, dropout=dropout)
    tgt_pos_enc = PositionalEncoding(d_model, seq_len=tgt_seq_len, dropout=dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(num_encoder_layers):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(num_decoder_layers):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer model
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_enc, tgt_pos_enc, projection)

    # Initialize the parameters of the model
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer