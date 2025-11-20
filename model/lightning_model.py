import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        dr = 0.1
        self.mha = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            batch_first=True,
            dropout=dr
        )
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(dr)
        self.dropout2 = nn.Dropout(dr)
    
    def forward(self, x, padding_mask=None):
        x_norm = self.ln1(x)
        
        # Create causal attention mask
        attn_mask = torch.triu(
            torch.ones(x_norm.size(1), x_norm.size(1)),
            diagonal=1
        ).bool().to(x_norm.device)
        
        if padding_mask is None:
            x_attn, _ = self.mha(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                need_weights=False
            )
        else:
            x_attn, _ = self.mha(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                need_weights=False,
                key_padding_mask=padding_mask
            )
        
        x = x + self.dropout1(x_attn)
        x_norm = self.ln2(x)
        x_ffwd = self.ffwd(x_norm)
        x = x + self.dropout2(x_ffwd)
        
        return x


class GPTModel(nn.Module):
    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None, padding_mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B * T, C)
            targets_view = targets.view(B * T)
            loss = F.cross_entropy(logits_view, targets_view, ignore_index=-100)
        
        return logits, loss


class GPTLightningModule(pl.LightningModule):
    def __init__(
        self,
        block_size,
        vocab_size,
        n_embd,
        n_head,
        n_layer,
        learning_rate=3e-4,
        weight_decay=0.1,
        max_epochs=100,
        use_qat=False,
        qat_backend='fbgemm'
    ):
        """
        PyTorch Lightning wrapper for GPT model with optional QAT.
        
        Args:
            block_size: Maximum sequence length
            vocab_size: Size of vocabulary
            n_embd: Embedding dimension
            n_head: Number of attention heads
            n_layer: Number of transformer blocks
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            max_epochs: Maximum number of training epochs (for scheduler)
            use_qat: Whether to use Quantization-Aware Training
            qat_backend: Backend for quantization ('fbgemm' for x86, 'qnnpack' for ARM)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create the model
        self.model = GPTModel(block_size, vocab_size, n_embd, n_head, n_layer)
        
        # QAT setup
        self.use_qat = use_qat
        if self.use_qat:
            self.model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qat_backend)
            # Prepare model for QAT
            torch.ao.quantization.prepare_qat(self.model, inplace=True)
    
    def forward(self, idx, targets=None, padding_mask=None):
        return self.model(idx, targets, padding_mask)
    
    def training_step(self, batch, batch_idx):
        # Assuming batch is a tuple of (input_ids, targets) or dict
        idx = batch['x']
        targets = batch['y']
        padding_mask = batch.get('padding_mask', None)
        
        logits, loss = self(idx, targets, padding_mask)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        idx = batch['x']
        targets = batch['y']
        padding_mask = batch.get('padding_mask', None)

        logits, loss = self(idx, targets, padding_mask)
        
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        idx = batch['x']
        targets = batch['y']
        padding_mask = batch.get('padding_mask', None)
        
        logits, loss = self(idx, targets, padding_mask)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        # Create optimizer with weight decay
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.learning_rate * 0.1
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def convert_to_quantized(self):
        """
        Convert the QAT model to a fully quantized model.
        Call this after training is complete.
        """
        if not self.use_qat:
            raise ValueError("Model was not trained with QAT. Set use_qat=True during initialization.")
        
        self.model.eval()
        torch.ao.quantization.convert(self.model, inplace=True)
        return self.model
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, padding_mask=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: (B, T) tensor of token indices
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k logits
            padding_mask: Optional padding mask
        
        Returns:
            Generated token indices
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if it exceeds block_size
                idx_cond = idx if idx.size(1) <= self.model.block_size else idx[:, -self.model.block_size:]
                
                # Get predictions
                logits, _ = self(idx_cond, padding_mask=padding_mask)
                logits = logits[:, -1, :] / temperature
                
                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# Example usage
if __name__ == "__main__":
    # Model configuration
    config = {
        'block_size': 128,
        'vocab_size': 50257,
        'n_embd': 384,
        'n_head': 6,
        'n_layer': 6,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'max_epochs': 100,
        'use_qat': False  # Set to True to enable QAT
    }
    
    # Create model
    model = GPTLightningModule(**config)
    
    # For training with QAT, simply set use_qat=True
    model_with_qat = GPTLightningModule(**{**config, 'use_qat': True})
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"QAT enabled: {model.use_qat}")
