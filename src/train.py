import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from .model import ChessNet
from .dataset import ChessDataset
import argparse
import tqdm
import os

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    cache_path = args.cache if args.cache else None
    dataset = ChessDataset([args.pgn], max_samples=args.max_samples, cache_path=cache_path)
    if len(dataset) == 0:
        print("Dataset is empty. Check PGN file path.")
        return
        
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    model = ChessNet(num_residual_blocks=args.res_blocks, channels=args.channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for tensors, move_indices, outcomes in pbar:
            tensors, move_indices, outcomes = tensors.to(device), move_indices.to(device), outcomes.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                policy_logits, value_output = model(tensors)
                loss_policy = criterion_policy(policy_logits, move_indices)
                loss_value = criterion_value(value_output, outcomes)
                loss = loss_policy + 0.5 * loss_value
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        correct_policy = 0
        total = 0
        with torch.no_grad():
            for tensors, move_indices, outcomes in val_loader:
                tensors, move_indices, outcomes = tensors.to(device), move_indices.to(device), outcomes.to(device)
                policy_logits, value_output = model(tensors)
                
                loss_policy = criterion_policy(policy_logits, move_indices)
                loss_value = criterion_value(value_output, outcomes)
                loss = loss_policy + 0.5 * loss_value
                val_loss += loss.item()
                
                _, predicted = torch.max(policy_logits, 1)
                total += move_indices.size(0)
                correct_policy += (predicted == move_indices).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        policy_acc = correct_policy / total
        print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Policy Acc: {policy_acc:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best.pt")
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--res_blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs("checkpoints", exist_ok=True)
    train(args)
