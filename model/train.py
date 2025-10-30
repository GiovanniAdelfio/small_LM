import torch
import torch.optim as optim
import tqdm

def train(model, train_dataloader, val_dataloader=None, lr=1e-3, weight_decay=1e-4, epochs=10, opt_name="adam", device="cpu", checkpoint_path=None):
    model = model.to(device)

    #Optimizer
    if opt_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{opt_name}' non supportato.")

    #Train
    loss_train = []
    loss_val = []
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            xb, yb = batch["x"].to(device), batch["y"].to(device)
            logits, loss = model(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()


        avg_train_loss = ep_loss / len(train_dataloader)
        loss_train.append(avg_train_loss)

        #Evaluation
        avg_val_loss = 0.0
        if val_dataloader:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    xb, yb = batch["x"].to(device), batch["y"].to(device)
                    _, val_loss = model(xb, yb)
                    val_loss_sum += val_loss.item()
            avg_val_loss = val_loss_sum / len(val_dataloader)
            loss_val.append(avg_val_loss)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        #Checkpoint
        if checkpoint_path:
            torch.save(model.state_dict(), f"{checkpoint_path}_epoch{epoch+1}.pt")

    print("Training completato!")
    return model, loss_train, loss_val
