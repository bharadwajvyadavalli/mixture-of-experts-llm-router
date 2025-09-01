"""
Evaluation metrics for MoE model
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time
import json


class Evaluator:
    """Evaluate MoE model performance"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def calculate_perplexity(self, dataloader):
        """Calculate perplexity on dataset"""
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Calculating perplexity'):
                input_ids = batch['input_ids'].to(self.device)
                
                targets = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                
                logits, _ = self.model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(logits, targets)
                total_loss += loss.item() * targets.size(0)
                total_tokens += targets.size(0)
        
        perplexity = np.exp(total_loss / total_tokens)
        return perplexity
    
    def measure_inference_speed(self, dataloader, num_batches=10):
        """Measure inference speed"""
        times = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                
                # Warmup
                if i == 0:
                    for _ in range(3):
                        _ = self.model(input_ids[:, :-1])
                
                # Time inference
                start = time.time()
                _ = self.model(input_ids[:, :-1])
                torch.cuda.synchronize() if self.device == 'cuda' else None
                times.append(time.time() - start)
        
        avg_time = np.mean(times[1:])  # Exclude warmup
        tokens_per_second = input_ids.shape[0] * input_ids.shape[1] / avg_time
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'tokens_per_second': tokens_per_second
        }
    
    def count_parameters(self):
        """Count model parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        active_per_token = total * (self.model.top_k / self.model.num_experts)
        
        return {
            'total_params': total,
            'active_params_per_token': int(active_per_token),
            'efficiency_ratio': 1 - (active_per_token / total)
        }
    
    def evaluate(self, dataloader):
        """Run complete evaluation"""
        print("\nEvaluating model...")
        
        results = {}
        
        # Perplexity
        results['perplexity'] = self.calculate_perplexity(dataloader)
        print(f"Perplexity: {results['perplexity']:.2f}")
        
        # Speed
        speed_metrics = self.measure_inference_speed(dataloader)
        results.update(speed_metrics)
        print(f"Inference: {speed_metrics['avg_inference_time_ms']:.1f}ms")
        print(f"Throughput: {speed_metrics['tokens_per_second']:.0f} tokens/s")
        
        # Parameters
        param_metrics = self.count_parameters()
        results.update(param_metrics)
        print(f"Total params: {param_metrics['total_params']/1e6:.1f}M")
        print(f"Active params: {param_metrics['active_params_per_token']/1e6:.1f}M per token")
        print(f"Efficiency: {param_metrics['efficiency_ratio']*100:.1f}% reduction")
        
        # Save results
        with open('results/evaluation.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results