import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpectrumEncoder(nn.Module):
    def __init__(self, input_dim=20000, hidden1=1024, context_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, context_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DiffusionTransformer(nn.Module):
    def __init__(self, vocab_size=23, seq_len=32, embed_dim=128, context_dim=256, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # We project the context to sequence length so we can concat or add it.
        # Alternatively, add context to the embeddings.
        self.context_proj = nn.Linear(context_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len
        
    def forward(self, x_t, t, context):
        # x_t: [B, S]
        # t: [B] context: [B, context_dim]
        B, S = x_t.shape
        
        # Token & Positional embeddings
        positions = torch.arange(S, device=x_t.device).unsqueeze(0).expand(B, S)
        emb = self.token_emb(x_t) + self.pos_emb(positions)
        
        # Time and Context
        t_emb = self.time_embed(t.unsqueeze(-1).float()).unsqueeze(1) # [B, 1, E]
        c_emb = self.context_proj(context).unsqueeze(1) # [B, 1, E]
        
        # Add context and time to every token's embedding
        x = emb + t_emb + c_emb
        
        # Transformer
        out = self.transformer(x)
        
        # Project to vocab
        logits = self.head(out) # [B, S, V]
        return logits


class CategoricalDiffusion(nn.Module):
    def __init__(self, encoder, model, total_timesteps=100, vocab_size=23):
        super().__init__()
        self.encoder = encoder
        self.model = model
        self.num_timesteps = total_timesteps
        self.vocab_size = vocab_size

    def q_sample(self, x_0, t):
        """
        Forward process: uniform diffusion.
        With probability alpha_t, keep the token. With 1-alpha_t, resample uniformly.
        Here we use a simple linear schedule for alpha_bar.
        """
        B, S = x_0.shape
        alpha_bar = 1.0 - (t.float() / self.num_timesteps) # [B]
        alpha_bar = alpha_bar.view(B, 1)
        
        # Sample uniform tokens
        noise = torch.randint_like(x_0, low=0, high=self.vocab_size)
        
        # Mask for which tokens to replace
        replace_mask = torch.rand(B, S, device=x_0.device) > alpha_bar
        
        x_t = torch.where(replace_mask, noise, x_0)
        return x_t

    def compute_loss(self, x_0, spectrum):
        """
        Sample t, compute context, corrupt to x_t, predict x_0, compute CrossEntropy.
        """
        B = x_0.shape[0]
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=x_0.device)
        
        # Corrupt
        x_t = self.q_sample(x_0, t)
        
        # Encode
        context = self.encoder(spectrum)
        
        # Predict logits
        logits = self.model(x_t, t, context)
        
        # Compute loss (cross entropy to predict original x_0)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), x_0.view(-1), ignore_index=0)
        return loss

    @torch.no_grad()
    def sample(self, spectrum):
        # Simplistic sampling: start from random noise, denoise iteratively.
        B = spectrum.shape[0]
        x_t = torch.randint(0, self.vocab_size, (B, self.model.seq_len), device=spectrum.device)
        context = self.encoder(spectrum)
        
        for t in reversed(range(1, self.num_timesteps + 1)):
            t_tensor = torch.full((B,), t, device=spectrum.device, dtype=torch.long)
            logits = self.model(x_t, t_tensor, context)
            
            # Very basic greedy sampling of the predicted x_0 for the next step.
            # In a true discrete diffusion, we'd sample from the posterior q(x_{t-1} | x_t, x_0)
            # But for baseline preliminary results Checkpoint 2, this is sufficient structure to show progress.
            x_0_pred = torch.argmax(logits, dim=-1)
            
            # Simple heuristic: at t=1, we just take the prediction. 
            # Otherwise we mix it with some random noise.
            alpha_bar_t_minus_1 = 1.0 - ((t - 1) / self.num_timesteps)
            replace_mask = torch.rand(B, self.model.seq_len, device=spectrum.device) > alpha_bar_t_minus_1
            noise = torch.randint_like(x_0_pred, low=0, high=self.vocab_size)
            x_t = torch.where(replace_mask, noise, x_0_pred)
            
        return x_t
