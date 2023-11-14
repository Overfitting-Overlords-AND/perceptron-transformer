import math
import torch

emb_dim = 40
seq_len = 1024  # irrelevant
num_heads = 5
drop = 0.1
vocab = 16000


class PositionalEncoding(torch.nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(seq_len, emb_dim)
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # unsqueeze makes it [s, 1] instead of [s]
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, S, E]

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_dim, seq_len):
        super(MultiHeadAttention, self).__init__()

        self.qkv_proj = torch.nn.Linear(emb_dim, emb_dim * 3)
        self.wo_proj = torch.nn.Linear(emb_dim, emb_dim)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len).view(1, 1, seq_len, seq_len)),
        )
        # view makes it change from a 2d tensor [3,3] to a 4d one, [1,1,3,3]..a batch of size 1, channel of size 1?

    def forward(self, x):
        B, S, E = x.shape
        EMBD_HEAD = int(emb_dim / num_heads)

        qry, key, val = self.qkv_proj(x).split(
            emb_dim, dim=-1
        )  # at this point shapes (B, S, E) =
        # (B,S,E,embedhead*num_heads)
        # e.g. if E is 30 and num_heads = 5 then emb_head = 6
        qry = qry.reshape(B, S, num_heads, EMBD_HEAD).transpose(
            1, 2
        )  # split into (B, S, num_heads, emb_head)
        # after transpose, B batches, num_heads per batch, S x emb_head matrices
        key = key.reshape(B, S, num_heads, EMBD_HEAD).transpose(1, 2)
        val = val.reshape(B, S, num_heads, EMBD_HEAD).transpose(1, 2)

        msk = (
            self.mask[:, :, :S, :S] == 0
        )  # take all elts in batch and channel(only one) and S of the seq_len x "
        att = qry @ key.transpose(-1, -2) / torch.sqrt(torch.tensor(EMBD_HEAD))
        # qry = S x embhead, key = embhead x S and scale with the size of each head
        att = att.masked_fill(msk, float("-inf"))  # (B, num_heads, S, S)
        att = torch.nn.functional.softmax(
            att, dim=-1
        )  # softmax along columns..(B, num_heads, S, S)
        out = (
            (att @ val).transpose(1, 2).reshape(B, S, E)
        )  # (B, numheads, S, embhead) to #(B, S, numheads, embhead)
        # to (B,S,E) so per batch, S numhead x embhead matrices, one per sequence word to one S x E matrix
        return self.wo_proj(out)


class FeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = torch.nn.Linear(
            emb_dim, emb_dim * 4
        )  # By using a higher-dimensional space (e.g., *4),
        # the model can potentially learn more intricate and non-linear relationships within the data
        self.relu = torch.nn.ReLU()
        self.c_proj = torch.nn.Linear(emb_dim * 4, emb_dim)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, seq_len)
        self.ln_2 = torch.nn.LayerNorm(emb_dim)
        self.ffww = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffww(self.ln_2(x))
        return x


class GPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(vocab, emb_dim)
        self.drop = torch.nn.Dropout(drop)
        self.blocks = torch.nn.ModuleList([AttentionBlock() for _ in range(4)])
        self.norm = torch.nn.LayerNorm(emb_dim)
        self.vocab = torch.nn.Linear(emb_dim, vocab)
        self.pos = PositionalEncoding()

    def forward(self, x):
        this_seq_len = min(x.size(1), seq_len)

        # Slice the input tensor to the desired sequence length
        x = x[:, :this_seq_len]
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos(x)
        x = tok_emb + pos_emb
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.vocab(x)
        return x  # 4 seperate attention blocks, output (B, S, E)

    def num_params(self):
        gpt_params = sum(p.numel() for p in self.parameters())  # no. of parameters
        emb_params = (
            self.tok_emb.weight.numel()
        )  # no of parameters (weights) in the token embeddings
        print(f"Total Parameters: {gpt_params} | Embedding: {emb_params}")
        return {"gpt_params": gpt_params, "emb_params": emb_params}

    def generate(self, x, temp=1.0, num=10):
        self.eval()
        # Generate 'num' tokens
        for _ in range(num):
            # Turn off gradient computation for efficiency during generation
            with torch.no_grad():
                # Forward pass through the model to get logits

                logits = self(x)
                # Take the logits for the last position in the sequence (final row)
                logits = logits[:, -1, :] / temp
                # Apply softmax to each entry in row (along the columns) to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, E)
                # Sample the next token from the probability distribution,
                # In a multinomial distribution, each element's probability represents the likelihood of being chosen
                next = torch.multinomial(probs, num_samples=1)
                # If the sampled token is an end-of-sequence token (e.g., EOS), stop generation
                if next.item() == 1:
                    break
                # Concatenate the sampled token to the input sequence
                x = torch.cat([x, next], dim=1)
        self.train()
        return x
