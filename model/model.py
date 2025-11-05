from heapq import merge
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization

class FeedFoward(nn.Module): #piccolo MLP per ogni token
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x) # Converte in "falso-quantizzato"
        
        x = self.linear1(x)
        x = nn.ReLU(x)
        x = self.linear2(x)
        
        x = self.dequant(x)
        return x

class Block(nn.Module):
    """ Transformer block fedele al paper originale (Post-Norm) """

    def __init__(self, n_embd, n_head):
        super().__init__()
        dr = 0.1
        # Usiamo MultiheadAttention di PyTorch che è efficiente
        self.mha = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, batch_first=True, dropout=dr)
        self.ffwd = FeedFoward(n_embd)
        # Due layer di normalizzazione
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # Dropout, come nel paper originale
        self.dropout1 = nn.Dropout(dr)
        self.dropout2 = nn.Dropout(dr)

    def forward(self, x, padding_mask = None):

        x = self.ln1(x)
        # --- Primo sotto-livello: Multi-Head Attention ---
        # Calcoliamo l'attention. La maschera serve per non far "vedere" il futuro.
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        
        if padding_mask is None:
            x_attn, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        else:
            x_attn, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False, key_padding_mask = padding_mask)

        # 1. Connessione residua (Add) e Dropout
        x = x + self.dropout1(x_attn)
        # 2. Normalizzazione (Norm)

        x = self.ln2(x)

        # --- Secondo sotto-livello: Feed Forward ---
        x_ffwd = self.ffwd(x)

        # 1. Connessione residua (Add) e Dropout
        x = x + self.dropout2(x_ffwd)
        # 2. Normalizzazione (Norm)

        return x

class GPTModel(nn.Module):
    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer):      ##non era definito block_size
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # layer norm finale
        self.lm_head = nn.Linear(n_embd, vocab_size)  # predice i token

    def forward(self, idx, targets=None, padding_mask = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)       # (B,T,n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,n_embd)
        x = tok_emb + pos_emb                           # somma embedding + posizioni
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask)                            # passa attraverso tutti i Block
        x = self.ln_f(x)                                # normalizzazione finale
        logits = self.lm_head(x)                        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Per calcolare la loss, Pytorch ha bisogno di un formato (Batch * T, n_classes)
            B, T, C = logits.shape
            logits_view = logits.view(B*T, C)
            targets_view = targets.view(B*T)
            loss = F.cross_entropy(logits_view, targets_view, ignore_index=-100)

        return logits, loss

    def generate(self, idx, max_new_tokens):
      for _ in range(max_new_tokens):
          # taglia il contesto agli ultimi block_size token
          idx_cond = idx[:, -self.block_size:]
          # ottieni le predizioni
          logits = self(idx_cond)
          # prendiamo solo l'ultimo passo temporale
          logits = logits[:, -1, :]  # (B, vocab_size)
          probs = F.softmax(logits, dim=-1)
          idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
          # aggiungiamo il nuovo token alla sequenza
          idx = torch.cat((idx, idx_next), dim=1)
      return idx

    def fuse_model(self):
        """
        Unisce (Linear, ReLU) in un unico modulo ottimizzato
        """
        for block in self.blocks:
            # 'ffwd.net' è il nn.Sequential dentro FeedFoward
            # ['0', '1'] sono gli indici di nn.Linear (0) e nn.ReLU (1)
            torch.ao.quantization.fuse_modules(
                block.ffwd.net, ['0', '1'], inplace=True
            )
    
@torch.no_grad() # Fondamentale: disabilita il calcolo dei gradienti per risparmiare memoria e velocizzare
def generate(model, start_text, max_new_tokens, stoi, itos, merges, block_size, conversation = False, temperature=1.0, top_k=None):
    """
    Genera testo autoregressivamente a partire da un contesto iniziale (stringa).

    Args:
        model: Il modello GPT addestrato.
        start_text (str): La stringa di partenza da cui iniziare la generazione.
        max_new_tokens (int): Il numero massimo di nuovi token da generare.
        stoi (dict): Dizionario da carattere a intero (token).
        itos (dict): Dizionario da intero (token) a carattere.
        merges (dict): Regole di merge per il tokenizer BPE.
        block_size (int): La dimensione del contesto del modello.
        temperature (float): Controlla la casualità. Valori > 1.0 aumentano la casualità,
                             valori < 1.0 la diminuiscono. 1.0 è il default.
        top_k (int, optional): Se specificato, considera solo i 'k' token più probabili
                               ad ogni passo.
    """
    model.eval() # Mette il modello in modalità di valutazione (disattiva dropout, etc.)

    from tokenizer.tokenizer import encode, decode

    # unsqueeze(0) aggiunge la dimensione del batch (B=1)
    context = torch.tensor(encode(start_text, merges, stoi, len(stoi), len(merges)), dtype=torch.long, device=model.lm_head.weight.device).unsqueeze(0)
    context = context.reshape(1, -1)

    # --- 4. Loop di generazione ---
    for _ in range(max_new_tokens):
        # Se il contesto è più lungo di block_size, lo tagliamo
        context_cond = context[:, -block_size:]

        # Otteniamo i logits dal modello
        logits, _ = model(context_cond)

        # Prendiamo solo i logits dell'ultimo token, che ci servono per predire il successivo
        logits = logits[:, -1, :] # -> (B, vocab_size)

        # Applica la temperatura per modulare la distribuzione di probabilità
        logits = logits / temperature

        # (Opzionale) Applica il top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf') # Mette a -infinito tutti i logits non nella top k

        # Calcola le probabilità con softmax
        probs = F.softmax(logits, dim=-1)

        # Campiona il prossimo token dalla distribuzione di probabilità
        next_token = torch.multinomial(probs, num_samples=1) # -> (B, 1)

        # Aggiunge il nuovo token al contesto per il prossimo ciclo
        context = torch.cat([context, next_token], dim=1)
        
        if conversation and ('@' in itos[next_token.item()]):
            break

    model.train() # Riporta il modello in modalità training

    # --- 5. Decodifica e restituisce il risultato ---
    generated_text = decode(context[0].tolist(), itos)

    return generated_text







