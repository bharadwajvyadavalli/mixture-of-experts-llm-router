"""
Main script for MoE model training and evaluation
Usage: python main.py [--evaluate] [--quick]
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.model import MixtureOfExperts
from src.trainer import Trainer
from src.data_generator import create_dataloaders
from src.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(description='Train MoE model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--quick', action='store_true', help='Quick training (2 epochs)')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    Path('checkpoints').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    print("Initializing model...")
    model = MixtureOfExperts(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        hidden_dim=1024,
        num_experts=4,
        num_layers=2,
        top_k=2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.1f}M")
    
    if args.evaluate:
        # Load checkpoint
        checkpoint_path = Path('checkpoints/best_model.pt')
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded checkpoint")
        else:
            print("No checkpoint found, using random weights")
        
        # Evaluate
        print("\nEvaluating model...")
        _, val_loader = create_dataloaders(tokenizer, batch_size=32)
        evaluator = Evaluator(model, device)
        results = evaluator.evaluate(val_loader)
        
    else:
        # Create data loaders
        print("\nPreparing data...")
        train_loader, val_loader = create_dataloaders(tokenizer, batch_size=32)
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Train model
        print("\nTraining model...")
        trainer = Trainer(model, device)
        epochs = 2 if args.quick else 10
        history = trainer.train(train_loader, val_loader, epochs=epochs)
        
        # Evaluate
        print("\nEvaluating model...")
        evaluator = Evaluator(model, device)
        results = evaluator.evaluate(val_loader)
    
    print("\n" + "="*50)
    print("COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. View dashboard: streamlit run visualization/dashboard.py")
    print("2. Generate plots: python visualization/plots.py")


if __name__ == "__main__":
    main()