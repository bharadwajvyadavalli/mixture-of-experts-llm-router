"""
Training module for MoE model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json
from pathlib import Path


class Trainer:
    """Trainer for MoE model"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        Path('checkpoints').mkdir(exist_ok=True)
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc='Training'):
            input_ids = batch['input_ids'].to(self.device)
            
            # Shift for language modeling
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            
            # Forward pass
            logits, aux_loss = self.model(input_ids)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            # Calculate loss
            task_loss = self.criterion(logits, targets)
            loss = task_loss + 0.01 * aux_loss  # Auxiliary loss weight
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validating'):
                input_ids = batch['input_ids'].to(self.device)
                
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                
                logits, aux_loss = self.model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=10):
        for epoch in range(1, epochs + 1):
            print(f'\nEpoch {epoch}/{epochs}')
            
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, 'checkpoints/best_model.pt')
                print('✓ Saved best model')
        
        # Save history
        with open('checkpoints/history.json', 'w') as f:
            json.dump(self.history, f)
        
        return self.history