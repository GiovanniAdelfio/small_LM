import torch
import torch.optim as optim
from tqdm import tqdm

import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

def run_training(
    train_loader,
    val_loader,
    save_dir: str,
    experiment_name: str,
    model: pl.LightningModule, # Accetta il modello già creato
    max_epochs: int = 100,
    patience: int = 5
):
    """
    Esegue il training con Early Stopping e salvataggio dei checkpoint.
    Restituisce il percorso (path) del miglior modello salvato.
    """
    
    # 1. Creazione cartella per l'esperimento
    full_save_path = os.path.join(save_dir, experiment_name)
    os.makedirs(full_save_path, exist_ok=True)

    # 2. Configurazione Callbacks
    
    # EarlyStopping: ferma se la val_loss non scende per 'patience' epoche
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=True,
        mode='min'
    )

    # ModelCheckpoint: salva i migliori modelli in base alla val_loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=full_save_path,
        filename='gpt-{epoch:02d}-{val_loss:.4f}', # es: gpt-05-0.4512.ckpt
        save_top_k=1,         # Teniamo solo il migliore in assoluto per risparmiare spazio
        monitor='val_loss',
        mode='min',
        save_last=True        # Salva anche l'ultimo step (utile per resume)
    )

    # 3. Setup del Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",   # Usa GPU se disponibile, altrimenti CPU
        devices="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # 4. Avvio Training
    print(f"--- Starting Experiment: {experiment_name} ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"Training completato. Il miglior modello è salvato in:\n{checkpoint_callback.best_model_path}")
    
    # Restituisce la stringa col percorso del file .ckpt
    return checkpoint_callback.best_model_path
