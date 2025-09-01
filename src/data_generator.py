"""
Synthetic data generator for domain-specific training
"""

import random
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class DomainDataGenerator:
    """Generate synthetic domain-specific text"""
    def __init__(self):
        self.templates = {
            'finance': [
                "The stock price of {} {} by {}% to ${}",
                "Q{} earnings report shows revenue of ${} billion",
                "The {} index {} {} points following {}",
                "Investment in {} sector {} as {}"
            ],
            'legal': [
                "Pursuant to Section {}, the {} shall {} within {} days",
                "The court {} that {} under {}",
                "This agreement is between {} and {} dated {}",
                "The {} hereby {} all rights to {}"
            ],
            'code': [
                "def {}({}): return {}",
                "class {}: def __init__(self): self.{} = {}",
                "for {} in {}: {}",
                "if {}: {} else: {}"
            ],
            'general': [
                "The {} {} {} in {}",
                "Studies show that {} when {}",
                "According to {}, {} has {}",
                "{} announced that {} during {}"
            ]
        }
        
        self.vocab = {
            'finance': {
                'companies': ['Apple', 'Google', 'Tesla', 'Amazon'],
                'movements': ['rose', 'fell', 'surged', 'dropped'],
                'events': ['earnings', 'inflation data', 'Fed meeting']
            },
            'legal': {
                'parties': ['plaintiff', 'defendant', 'contractor'],
                'actions': ['ruled', 'determined', 'concluded'],
                'terms': ['negligence', 'breach', 'liability']
            },
            'code': {
                'functions': ['process', 'calculate', 'validate'],
                'variables': ['data', 'result', 'value'],
                'operations': ['append', 'return', 'print']
            },
            'general': {
                'subjects': ['researchers', 'scientists', 'experts'],
                'actions': ['discovered', 'developed', 'announced'],
                'objects': ['findings', 'technology', 'solution']
            }
        }
    
    def generate(self, domain, num_samples=100):
        samples = []
        for _ in range(num_samples):
            template = random.choice(self.templates[domain])
            # Simple fill - just use random words
            text = template.format(*[str(random.randint(1, 100)) for _ in range(template.count('{}'))])
            samples.append(text)
        return samples
    
    def generate_dataset(self, samples_per_domain=1000):
        Path('data').mkdir(exist_ok=True)
        dataset = {}
        
        for domain in self.templates.keys():
            dataset[domain] = self.generate(domain, samples_per_domain)
        
        with open('data/dataset.json', 'w') as f:
            json.dump(dataset, f)
        
        return dataset


class TextDataset(Dataset):
    """PyTorch dataset for text data"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding='max_length', 
            max_length=max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}


def create_dataloaders(tokenizer, batch_size=32, train_split=0.8):
    """Create train and validation dataloaders"""
    # Load or generate data
    data_path = Path('data/dataset.json')
    if data_path.exists():
        with open(data_path, 'r') as f:
            dataset = json.load(f)
    else:
        generator = DomainDataGenerator()
        dataset = generator.generate_dataset()
    
    # Combine all texts
    all_texts = []
    for domain_texts in dataset.values():
        all_texts.extend(domain_texts)
    
    # Split data
    split_idx = int(len(all_texts) * train_split)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader