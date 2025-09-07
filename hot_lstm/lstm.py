# Yiming Guo 08/31/2024
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict, Counter
import warnings
import os
import re
from typing import List, Tuple, Dict, Any, Optional
warnings.filterwarnings('ignore')

class KeySequenceDataset(Dataset):
    """Dataset for key sequence prediction with full sequences"""
    
    def __init__(self, sequences, key_to_idx, window_size, top_k_hot_keys, target_windows_size):
        self.sequences = sequences
        self.key_to_idx = key_to_idx
        self.window_size = window_size
        self.top_k_hot_keys = top_k_hot_keys
        self.vocab_size = len(key_to_idx)
        self.target_windows_size = target_windows_size
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        windows = self.sequences[idx] if idx < len(self.sequences) else []
        next_windows = self.sequences[idx + 1] if idx + 1 < len(self.sequences) else []

        input_windows = windows 
        target_windows = next_windows[-self.target_windows_size:]  
        
        seq_indices = []
        for window_keys in input_windows:
            indices = [self.key_to_idx.get(key, 0) for key in window_keys]
            if len(indices) < self.window_size:
                indices = indices + [0] * (self.window_size - len(indices))
            else:
                indices = indices[:self.window_size]
            seq_indices.append(indices)
        
        target_keys_flattened = []
        for window in target_windows:
            target_keys_flattened.extend(window)
        
        
        target_key_counter = Counter(target_keys_flattened)
        # print(f'training_key_counter{len(target_key_counter)}')
        target_hot_keys = [key for key, _ in target_key_counter.most_common(self.top_k_hot_keys)]
        

        target_indices = [self.key_to_idx.get(key, 0) for key in target_hot_keys]
        target_padded = target_indices + [0] * max(0, self.top_k_hot_keys - len(target_indices))
        target_padded = target_padded[:self.top_k_hot_keys]
        
        # print (f'torch.tensor(seq_indices, dtype=torch.long).shape{torch.tensor(seq_indices, dtype=torch.long).shape}')
        # print (f'torch.tensor(target_padded, dtype=torch.long.shape{torch.tensor(target_padded, dtype=torch.long).shape}')

        return torch.tensor(seq_indices, dtype=torch.long), torch.tensor(target_padded, dtype=torch.long)

class EfficientKeyPredictionLSTM(nn.Module):
    
    def __init__(self, vocab_size, window_size, embedding_dim=32, hidden_size=64,num_layers = 2,dropout_rate = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim * window_size,
            hidden_size=hidden_size,
            num_layers=num_layers,  
            dropout= dropout_rate,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x shape: [batch, seq_len, window_size]
        batch, seq_len, win_size = x.shape
        
        # 嵌入层
        x = x.view(batch * seq_len, win_size)
        emb = self.embedding(x)  # [batch*seq_len, win_size, emb_dim]
        emb = emb.view(batch, seq_len, -1)  # [batch, seq_len, win_size*emb_dim]
        lstm_out, _ = self.lstm(emb)
        last_out = lstm_out[:, -1, :]

        return self.fc(last_out)
 
 

class KeyAccuracyMetrics:
    
    def __init__(self, k_list=[10000]):
        self.k_list = k_list
        
    def __call__(self, predictions, targets):
        """
        predictions: (batch_size, vocab_size)
        targets: (batch_size, top_k) - indices of hot keys
        """
        batch_size, vocab_size = predictions.shape
        results = {}
        
        for k in self.k_list:
            _, pred_indices = predictions.topk(k, dim=-1)
            
            total_recall = 0.0
            total_precision = 0.0
            total_f1 = 0.0
            total_accuracy = 0.0
            total_samples = 0
            
            for i in range(batch_size):
                non_zero_targets = targets[i][targets[i] != 0]
                if len(non_zero_targets) == 0:
                    continue
                    
                target_set = set(non_zero_targets.tolist())
                pred_set = set(pred_indices[i].tolist())
                
                # print(f'target_set: {target_set}\n')
                # print(f'pred_set: {pred_set}')    
                
        
                intersection = pred_set & target_set
                tp = len(intersection)
                # print(f'len(intersection{tp}, target_set_len{len(target_set)}, pred_set{len(pred_set)})')
            
                recall = tp / len(target_set) if len(target_set) > 0 else 0.0
                precision = tp / k if k > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                accuracy = tp / min(len(target_set), k) if min(len(target_set), k) > 0 else 0.0
                
                total_recall += recall
                total_precision += precision
                total_f1 += f1
                total_accuracy += accuracy
                total_samples += 1
            
            if total_samples > 0:
                results[f'top_{k}_recall'] = total_recall / total_samples
                results[f'top_{k}_precision'] = total_precision / total_samples
                results[f'top_{k}_f1'] = total_f1 / total_samples
                results[f'top_{k}_accuracy'] = total_accuracy / total_samples
            else:
                for metric in ['recall', 'precision', 'f1', 'accuracy']:
                    results[f'top_{k}_{metric}'] = 0.0
        
        return results

class KeyPredictor:
    """Main class for key prediction using LSTM with complete sequences"""
    
    def __init__(self, window_size=20, sequence_length=10, top_k_hot_keys=5, prediction_win=2):
        self.window_size = window_size
        self.sequence_length = sequence_length
        self.top_k_hot_keys = top_k_hot_keys
        self.key_to_idx = {}
        self.idx_to_key = {}
        self.model = None
        self.prediction_win = prediction_win
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # For data statistics
        self.window_stats = {}
    
    def preprocess_data(self, df: pd.DataFrame) -> List:

        print("Starting data preprocessing with fixed-size windows...")
        start_time = time.time()

        
        windows = []
        total_records = len(df)
        # window_step = max(1, self.window_size // 10)不应该重复呀
        window_step = max(1, self.window_size)  
        
        print(f"Total records: {total_records}")
        print(f"Window size: {self.window_size} keys, Step: {window_step} keys")
        
        for i in range(0, total_records - self.window_size + 1, window_step):
            window_keys = df['key'].iloc[i:i+self.window_size].tolist()
            windows.append(window_keys)
        
        print(f"Created {len(windows)} windows with overlap")
        
        # Create key vocabulary from all keys
        all_keys = set()
        for key_list in windows:
            all_keys.update(key_list)
        
        # Create vocabulary mapping
        sorted_keys = sorted(all_keys)
        self.key_to_idx = {key: idx+1 for idx, key in enumerate(sorted_keys)}  # 0 reserved for padding
        self.key_to_idx["[PAD]"] = 0
        self.idx_to_key = {idx: key for key, idx in self.key_to_idx.items()}
        
        keys_per_window = [len(set(key_list)) for key_list in windows]  # unique keys per window
        self.window_stats = {
            'total_unique_keys': len(all_keys),
            'total_windows': len(windows),
            'avg_unique_keys_per_window': np.mean(keys_per_window),
            'max_unique_keys_per_window': np.max(keys_per_window),
            'min_unique_keys_per_window': np.min(keys_per_window),
            'window_size': self.window_size,
            'top_k_hot_keys': self.top_k_hot_keys
        }
        
        print(f"\nData Statistics:")
        print(f"  Total unique keys: {self.window_stats['total_unique_keys']}")
        print(f"  Total windows: {self.window_stats['total_windows']}")
        print(f"  Avg unique keys per window: {self.window_stats['avg_unique_keys_per_window']:.2f}")
        print(f"  Max unique keys per window: {self.window_stats['max_unique_keys_per_window']}")
        print(f"  Min unique keys per window: {self.window_stats['min_unique_keys_per_window']}")
        

        sequences = []
        
        print(f"\nCreating training sequences with {self.sequence_length} historical windows...")
     
        for i in range(len(windows) - self.sequence_length + 1):
            seq = windows[i:i + self.sequence_length]
            sequences.append(seq)
        
        print(f"Created {len(sequences)} training sequences in {time.time()-start_time:.2f} seconds")
        
        return sequences
    
    def create_data_loaders(self, sequences, batch_size=64, test_size=0.2, val_size=0.1):
        """Create train/val/test data loaders"""
        print("\nCreating data loaders...")
        start_time = time.time()
        total_samples = len(sequences)
        train_end = int(total_samples * (1 - test_size - val_size))
        val_end = int(total_samples * (1 - test_size))
        
        train_seq = sequences[:train_end]
        val_seq = sequences[train_end:val_end]
        test_seq = sequences[val_end:]
        
        print(f"  Train samples: {len(train_seq)}")
        print(f"  Validation samples: {len(val_seq)}")
        print(f"  Test samples: {len(test_seq)}")
        

        train_dataset = KeySequenceDataset(train_seq, self.key_to_idx, self.window_size, self.top_k_hot_keys, self.prediction_win)
        val_dataset = KeySequenceDataset(val_seq, self.key_to_idx, self.window_size, self.top_k_hot_keys, self.prediction_win)
        test_dataset = KeySequenceDataset(test_seq, self.key_to_idx, self.window_size, self.top_k_hot_keys, self.prediction_win)
 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"Data loaders created in {time.time()-start_time:.2f} seconds")
        return train_loader, val_loader, test_loader
    
   
    def set_based_loss(self, predictions, targets, base_emphasis=1.0, max_emphasis=3.0):
        """
        自适应热度增强损失，根据实际热键数量动态调整热度增强程度
        
        Args:
            predictions: 模型预测得分 (batch_size, vocab_size)
            targets: 目标热键索引 (batch_size, top_k)
            base_emphasis: 基础强调因子
            max_emphasis: 最大强调因子
        
        Returns:
            自适应热度增强损失值
        """
        batch_size, vocab_size = predictions.shape
        
        # 创建目标的多热编码并统计实际热键数量
        target_multihot = torch.zeros(batch_size, vocab_size, 
                                    device=predictions.device, 
                                    dtype=torch.float)
        actual_key_counts = torch.zeros(batch_size, device=predictions.device)
        
        for i in range(batch_size):
            actual_keys = targets[i][targets[i] != 0]
            if len(actual_keys) > 0:
                target_multihot[i, actual_keys] = 1.0
                actual_key_counts[i] = len(actual_keys)
        
        # 计算自适应强调因子
        # 热键数量越少，强调因子越大
        avg_key_count = actual_key_counts.float().mean()
        emphasis_factor = base_emphasis + (max_emphasis - base_emphasis) * torch.sigmoid(
            - (avg_key_count - 5) / 2  # 当平均热键数量为5时，强调因子为(base_emphasis + max_emphasis)/2
        )
        
        # 计算sigmoid概率
        pred_prob = torch.sigmoid(predictions)
        
        # 基础焦点损失
        alpha = 0.25
        gamma = 2.0
        bce_loss = nn.BCELoss(reduction='none')(pred_prob, target_multihot)
        p_t = target_multihot * pred_prob + (1 - target_multihot) * (1 - pred_prob)
        modulating_factor = (1 - p_t) ** gamma
        alpha_factor = target_multihot * alpha + (1 - target_multihot) * (1 - alpha)
        focal_loss = alpha_factor * modulating_factor * bce_loss
        
        # 热度增强项
        # 对真实热键的预测概率与1之间的差异进行惩罚，程度由强调因子控制
        heat_enhancement = target_multihot * (1 - pred_prob) ** emphasis_factor
        
        # 组合损失
        total_loss = focal_loss + heat_enhancement
        
        return total_loss.mean()

  
    def train_model(self, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
        """Train the LSTM model"""
        vocab_size = len(self.key_to_idx)
        print(f"\nInitializing efficient model with vocabulary size: {vocab_size}")
        start_time = time.time()
        
        self.model = EfficientKeyPredictionLSTM(
            vocab_size=vocab_size,
            window_size=self.window_size,
            embedding_dim = 16,
            hidden_size=32,
            num_layers=1,
            dropout_rate = 0.3 
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Loss function and optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        train_losses = []
        val_losses = []
        val_metrics_history = []  # Store all metrics for each epoch
        best_val_loss = float('inf')
        patience_counter = 0
     
        metrics_calculator = KeyAccuracyMetrics(k_list=[self.top_k_hot_keys])
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            self.model.train()
            train_loss = 0
            train_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                loss = self.set_based_loss(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_batches = 0
            val_metrics = defaultdict(float)
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    
                    loss = self.set_based_loss(outputs, batch_y)
                    val_loss += loss.item()
                    val_batches += 1
                    
            
                    batch_metrics = metrics_calculator(outputs, batch_y)
                    for key, value in batch_metrics.items():
                        val_metrics[key] += value
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            
            # Average metrics
            for key in val_metrics:
                val_metrics[key] /= val_batches if val_batches > 0 else 1
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_metrics_history.append(dict(val_metrics))
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_key_predictor.pth')
            else:
                patience_counter += 1
            
            # Print progress with metrics
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch [{epoch+1}/{epochs}] - {time.time()-epoch_start:.2f}s")
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
            print("  Validation Metrics:")
            for k in [self.top_k_hot_keys]:
                print(f"  Top-{k}: ", end="")
                for metric in ['recall', 'precision', 'f1', 'accuracy']:
                    key = f'top_{k}_{metric}'
                    value = val_metrics.get(key, 0)
                    print(f"{metric[:4]}: {value:.4f}  ", end="")
                print()
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
    
        if os.path.exists('best_key_predictor.pth'):
            self.model.load_state_dict(torch.load('best_key_predictor.pth'))
            print("Loaded best model weights")
        
        print(f"\nTraining completed in {time.time()-start_time:.2f} seconds")
        return train_losses, val_losses, val_metrics_history
    
    def evaluate_model(self, test_loader, k_list=[20000]):
        """Evaluate the model with various top-k metrics"""
        print("\nEvaluating model...")
        start_time = time.time()
        
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        
        self.model.eval()
        metrics_calculator = KeyAccuracyMetrics(k_list=k_list)
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                
                # Calculate metrics
                metrics_dict = metrics_calculator(outputs, batch_y)
                for key, value in metrics_dict.items():
                    all_metrics[key].append(value)
        
        # Average metrics
        results = {}
        for key, values in all_metrics.items():
            results[key] = np.mean(values)
        
        # Print evaluation results
        print("\nEvaluation Results:")
        print("-" * 60)
        for k in k_list:
            print(f"Top-{k} Metrics:")
            for metric in ['recall', 'precision', 'f1', 'accuracy']:
                key = f'top_{k}_{metric}'
                value = results.get(key, 0)
                print(f"  {metric.capitalize()}: {value:.4f}")
            print()
        
        print(f"Evaluation completed in {time.time()-start_time:.2f} seconds")
        return results
    
    def predict_next_window_hot_keys(self, recent_windows: List[List[str]], top_k: int = None) -> Tuple[List[str], np.ndarray]:

        if top_k is None:
            top_k = self.top_k_hot_keys
        
        if len(recent_windows) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} recent windows, got {len(recent_windows)}")
        
        # Take the last sequence_length windows
        input_windows = recent_windows[-self.sequence_length:]
        input_indices = []
        
        for window_keys in input_windows:
            indices = [self.key_to_idx.get(key, 0) for key in window_keys]
            if len(indices) < self.window_size:
                indices = indices + [0] * (self.window_size - len(indices))
            else:
                indices = indices[:self.window_size]
            input_indices.append(indices)
        
        # Create batch
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_tensor)  # (1, vocab_size)
            predictions = torch.softmax(predictions, dim=-1)
            scores, indices = predictions.topk(top_k, dim=-1) 
            predicted_keys = []
            prediction_scores = []
            
            for i in range(top_k):
                idx = indices[0, i].item()
                score = scores[0, i].item()
                predicted_keys.append(self.idx_to_key.get(idx, "[UNK]"))
                prediction_scores.append(score)
        
        return predicted_keys, np.array(prediction_scores)
    
    def save_model(self, filepath: str):
        """Save the trained model and metadata"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'key_to_idx': self.key_to_idx,
            'idx_to_key': self.idx_to_key,
            'window_size': self.window_size,
            'sequence_length': self.sequence_length,
            'top_k_hot_keys': self.top_k_hot_keys,
            'vocab_size': len(self.key_to_idx),
            'window_stats': self.window_stats
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
#     def load_model(self, filepath: str):
#         """Load a trained model and metadata"""
#         checkpoint = torch.load(filepath, map_location=self.device)
        
#         self.key_to_idx = checkpoint['key_to_idx']
#         self.idx_to_key = checkpoint['idx_to_key']
#         self.window_size = checkpoint['window_size']
#         self.sequence_length = checkpoint['sequence_length']
#         self.top_k_hot_keys = checkpoint['top_k_hot_keys']
#         self.window_stats = checkpoint.get('window_stats', {})
        
#         vocab_size = checkpoint['vocab_size']
#         self.model = EfficientKeyPredictionLSTM(
#             vocab_size=vocab_size,
#             window_size=self.window_size,
#             num_layers=2
#         ).to(self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
        
#         print(f"Model loaded from {filepath}")



def main():
    
    # 配置参数
    config = {
        # 'data_file': 'temporal_ycsb_workload_part1.csv',
        'data_file': 'processed_key.csv',    
        'window_size': 500,             
        'sequence_length': 300,         
        'top_k_hot_keys': 10000,   # in train 100000 ,the hottest is 20000  
        # 'top_k_hot_keys': 2000,
        'batch_size': 16,      
        'epochs': 5,          
        'learning_rate': 0.001,
        'patience': 10,         
        'prediction_win': 300
    }
    
    print("=== Efficient LSTM Hot Key Predictor ===")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nLoading workload data from {config['data_file']}...")
    start_time = time.time()
    df = pd.read_csv(config['data_file'])
    print(f"Loaded {len(df)} operations in {time.time()-start_time:.2f} seconds")

    predictor = KeyPredictor(
        window_size=config['window_size'],
        sequence_length=config['sequence_length'],
        top_k_hot_keys=config['top_k_hot_keys'],
        prediction_win=config['prediction_win']  
    )

    sequences = predictor.preprocess_data(df)
    
    if len(sequences) == 0:
        print("Error: No valid sequences created. Check your data and parameters.")
        return
    
    train_loader, val_loader, test_loader = predictor.create_data_loaders(
        sequences, batch_size=config['batch_size']
    )
    
    print("\n=== Training Model ===")
    train_losses, val_losses, val_metrics_history = predictor.train_model(
        train_loader, val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        patience=config['patience']
    )
    
    # visualize_training(train_losses, val_losses, val_metrics_history, 'training_progress.png')
    
    # print("\n=== Evaluating Model ===")
    metrics = predictor.evaluate_model(test_loader, k_list=[10000])
    
    # model_path = 'hot_key_prsedictor.pth'
    # predictor.save_model(model_path)
    
    # print(f"\nTraining completed! Model saved to {model_path}")

if __name__ == "__main__":
    main()