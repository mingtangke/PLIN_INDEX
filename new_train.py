import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import socket
import threading
import time
from collections import Counter, defaultdict
import select
import os
import re
import warnings
import json
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class DeviceAccessDataset(Dataset):
    """Dataset for device access prediction"""
    
    def __init__(self, sequences, targets, sequence_length, num_devices):
        self.sequences = sequences
        self.targets = targets
        self.sequence_length = sequence_length
        self.num_devices = num_devices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

class DeviceAccessPredictor:
    """Main class for device access prediction with communication capabilities"""
    
    def __init__(self, sequence_length=60, num_devices=20, top_k_hot_keys=100000):
        self.sequence_length = sequence_length
        self.num_devices = num_devices
        self.top_k_hot_keys = top_k_hot_keys
        self.model = None
        self.scaler = StandardScaler()
        self.device_hot_keys = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # For data statistics
        self.stats = {}
        
        # Communication setup
        self.sock = None
        self.conn = None
        self.addr = None
        self.running = False
        
        # Log file path
        self.log_file = "/home/ming/桌面/PLIN-N /PLIN-N/code_demo/workload_logs/workload_log.csv"
        
    def setup_communication(self, host='127.0.0.1', port=60002):
        """Set up socket communication with C++"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)
        print(f"Listening for C++ connection on {host}:{port}")
        
        self.conn, self.addr = self.sock.accept()
        print(f"Connected to C++ at {self.addr}")
        
    def listen_for_messages(self):
        """Listen for messages from C++"""
        self.running = True
        
        while self.running:
            try:
                # Check if data is available
                ready = select.select([self.conn], [], [], 0.1)
                if ready[0]:
                    data = self.conn.recv(1024).decode('utf-8')
                    
                    if not data:
                        continue
                        
                    if data.startswith("INDEX:"):
                        # Parse index range
                        parts = data.split(":")
                        if len(parts) >= 3:
                            start_idx = int(parts[1])
                            end_idx = int(parts[2])
                            
                            # Process new data
                            self.train_and_predict(start_idx, end_idx)
                        
            except Exception as e:
                print(f"Error in communication: {e}")
                time.sleep(1)
                
    def analyze_hot_keys(self, df, top_k=100):
        """
        分析日志并提取每个设备的热键
        
        参数:
        df: 原始数据DataFrame
        top_k: 每个设备提取的热键数量
        
        返回:
        每个设备的热键字典
        """
        device_hot_keys = {}
        
        # 获取所有设备ID
        device_ids = df["device_id"].unique()
        
        for device_id in device_ids:
            device_logs = df[df["device_id"] == device_id]
            key_counts = Counter(device_logs["key"])
            
            # 获取前top_k个最常访问的键
            hot_keys = [int(key) for key, count in key_counts.most_common(top_k)]
            device_hot_keys[int(device_id)] = hot_keys
        
        return device_hot_keys
    
    def save_hot_keys(self, device_hot_keys, filename="hot_keys.json"):
        """保存热键信息到JSON文件"""
        # 确保所有键都是基本Python类型
        serializable_hot_keys = {}
        for device_id, keys in device_hot_keys.items():
            serializable_hot_keys[str(device_id)] = [str(key) for key in keys]
        
        with open(filename, 'w') as f:
            json.dump(serializable_hot_keys, f, indent=4)
        print(f"热键信息已保存到: {filename}")
        return filename
        
    def preprocess_data(self, df: pd.DataFrame):
        """预处理数据并提取热键信息"""
        print("Starting data preprocessing...")
        start_time = time.time()

        # 确保时间戳是整数
        df['timestamp'] = df['timestamp'].astype(int)
        
        # 分析热键
        self.device_hot_keys = self.analyze_hot_keys(df, top_k=self.top_k_hot_keys)
        self.save_hot_keys(self.device_hot_keys, "device_hot_keys.json")
        
        # 聚合每秒的访问计数
        print("聚合访问计数...")
        agg_df = df.groupby(['timestamp', 'device_id']).size().unstack(fill_value=0)
        
        # 确保所有时间戳都有记录
        max_time = df['timestamp'].max()
        agg_df = agg_df.reindex(range(int(max_time) + 1), fill_value=0)
        
        # 确保所有设备都有列
        for device_id in range(1, self.num_devices + 1):
            if device_id not in agg_df.columns:
                agg_df[device_id] = 0
        
        # 按设备ID排序列
        agg_df = agg_df[sorted(agg_df.columns)]
        agg_df.columns = [f'count_dev{i}' for i in range(1, self.num_devices + 1)]
        
        # 标准化数据
        scaled_data = self.scaler.fit_transform(agg_df.values)
        
        # 创建序列
        X, y = self.create_sequences(scaled_data, self.sequence_length)
        y_labels = np.argmax(agg_df.values[self.sequence_length:], axis=1)
        
        print(f"Data preprocessing completed in {time.time()-start_time:.2f} seconds")
        return X, y_labels, agg_df
    
    def create_sequences(self, data, seq_length):
        """
        创建用于LSTM训练的序列数据
        """
        xs, ys = [], []
        
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
            
        return np.array(xs), np.array(ys)
    
    def create_data_loaders(self, X, y, batch_size=128, test_size=0.2, val_size=0.1):
        print("\nCreating data loaders...")
        start_time = time.time()
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"Data loaders created in {time.time()-start_time:.2f} seconds")
        
        return train_loader, test_loader
    
    def train_model(self, train_loader, test_loader, epochs=20, lr=0.001, patience=5):
        """Train the LSTM model"""
        print(f"\nInitializing LSTM model with input size: {self.num_devices}")
        start_time = time.time()
        
        input_size = self.num_devices
        hidden_size = 128
        num_layers = 2
        num_classes = self.num_devices
        
        self.model = LSTMPredictor(input_size, hidden_size, num_layers, num_classes).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose=True
        )
        
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        best_test_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs} Loss: {loss.item():.6f}')
            
            train_accuracy = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
        
            # 测试阶段
            self.model.eval()
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
            
            test_accuracy = 100. * test_correct / test_total
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            test_accuracies.append(test_accuracy)
            
            # 学习率调度
            scheduler.step(avg_test_loss)
            
            # 早停
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'scaler': self.scaler,
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'num_classes': num_classes,
                    'sequence_length': self.sequence_length,
                    'device_hot_keys': self.device_hot_keys
                }, 'best_device_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch [{epoch+1}/{epochs}] - {time.time()-epoch_start:.2f}s")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, LR: {current_lr:.6f}")
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # 加载最佳模型
        if os.path.exists('best_device_model.pth'):
            checkpoint = torch.load('best_device_model.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.device_hot_keys = checkpoint.get('device_hot_keys', {})
            print("Loaded best model weights")
        
        print(f"\nTraining completed in {time.time()-start_time:.2f} seconds")
        return train_losses, test_losses, train_accuracies, test_accuracies
    
    def evaluate_model(self, test_loader):
        """Evaluate the model"""
        print("\nEvaluating model...")
        start_time = time.time()
        
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Evaluation completed in {time.time()-start_time:.2f} seconds")
        
        return avg_test_loss, test_accuracy
    
    def predict_next_device(self, recent_sequence):
        """Predict the next device to be accessed"""
        if len(recent_sequence) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} recent sequences, got {len(recent_sequence)}")
        
        input_sequence = recent_sequence[-self.sequence_length:]
        input_tensor = torch.tensor([input_sequence], dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(input_tensor)
            predicted_device = torch.argmax(predictions, dim=1).item() + 1  # 设备ID从1开始
        
        return predicted_device
    
    def get_recent_sequence(self):
        """Get the most recent sequence from the log file"""
        try:
            df = pd.read_csv(self.log_file)
            
            # 确保时间戳是整数
            df['timestamp'] = df['timestamp'].astype(int)
            
            # 聚合每秒的访问计数
            agg_df = df.groupby(['timestamp', 'device_id']).size().unstack(fill_value=0)
            
            # 确保所有设备都有列
            for device_id in range(1, self.num_devices + 1):
                if device_id not in agg_df.columns:
                    agg_df[device_id] = 0
            
            # 按设备ID排序列
            agg_df = agg_df[sorted(agg_df.columns)]
            
            # 标准化数据
            scaled_data = self.scaler.transform(agg_df.values)
            
            # 获取最近的序列
            if len(scaled_data) >= self.sequence_length:
                return scaled_data[-self.sequence_length:]
            else:
                return scaled_data
                
        except Exception as e:
            print(f"Error getting recent sequence: {e}")
            return []
    
    def train_and_predict(self, start_idx, end_idx):
        """Process new data from the log file"""
        print(f"Processing new data from index {start_idx} to {end_idx}")
        
        # 读取新数据
        try:
            df = pd.read_csv(self.log_file)
            if end_idx > len(df):
                end_idx = len(df)
            if start_idx < 0:
                start_idx = 0
                
            new_df = df.iloc[start_idx:end_idx]
            
            if len(new_df) == 0:
                print("No new data to process")
                return
                
            X, y, agg_df = self.preprocess_data(new_df)
            
            if self.model is None:
                print("Training initial model...")
                train_loader, test_loader = self.create_data_loaders(X, y)
                self.train_model(train_loader, test_loader, epochs=5)
                self.evaluate_model(test_loader)
                self.save_model('initial_device_model.pth')
            else:
                print("Updating model with new data...")
                # 这里可以添加模型更新逻辑
                
            # 预测下一个设备
            recent_sequence = self.get_recent_sequence()
            print(f"Recent sequence available for prediction: {len(recent_sequence)}")
            
            if len(recent_sequence) >= self.sequence_length:
                predicted_device = self.predict_next_device(recent_sequence)
                hot_keys = self.device_hot_keys.get(predicted_device, [])
                hot_keys_str = ",".join(map(str, hot_keys[:100]))  # 只发送前100个热键
                
                message = f"DEVICE:{predicted_device}:HOT_KEYS:{hot_keys_str}END"
                try:
                    self.conn.send(message.encode('utf-8'))
                    print(f"Sent prediction for device {predicted_device}")
                except Exception as e:
                    print(f"Error sending prediction: {e}")
       
        except Exception as e:
            print(f"Error processing new data: {e}")
    
    def save_model(self, filepath: str):
        """Save the trained model and metadata"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'input_size': self.num_devices,
            'hidden_size': 128,
            'num_layers': 2,
            'num_classes': self.num_devices,
            'sequence_length': self.sequence_length,
            'device_hot_keys': self.device_hot_keys
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and metadata"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        num_classes = checkpoint['num_classes']
        
        self.model = LSTMPredictor(input_size, hidden_size, num_layers, num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.device_hot_keys = checkpoint.get('device_hot_keys', {})
        
        print(f"Model loaded from {filepath}")
    
    def stop(self):
        """Stop the communication thread"""
        self.running = False
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()

def main():
    # Create predictor instance
    predictor = DeviceAccessPredictor(
        sequence_length=60,
        num_devices=20,
        top_k_hot_keys=100000
    )
    
    # # Check if we have a pre-trained model
    # if os.path.exists('best_device_model.pth'):
    #     print("Loading pre-trained model...")
    #     predictor.load_model('best_device_model.pth')
    
    # Set up communication with C++
    print("Setting up communication with C++...")
    predictor.setup_communication("127.0.0.1", 60001)  
    
    # Start listening for messages in a separate thread
    comm_thread = threading.Thread(target=predictor.listen_for_messages)
    comm_thread.daemon = True
    comm_thread.start()
    
    try:
        # Keep the main thread alive
        print("Python device predictor is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping predictor...")
        predictor.stop()

if __name__ == "__main__":
    main()