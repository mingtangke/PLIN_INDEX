import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
from datetime import datetime, timedelta
import json
import random
from typing import List, Dict, Tuple
import argparse
import os

class TemporalYCSBGenerator:
    def __init__(self, 
                 num_keys: int = 100000,
                 num_operations: int = 1000000,
                 sampling_interval: int = 60, 
                 zipf_theta: float = 3.5,  # Zipfian distribution parameter
                 weight_update_hours: float = 1.0,  # 权重更新间隔（小时）
                 smooth_transition: bool = True):   # 是否平滑过渡
        """
        Initialize time-aware YCSB workload generator
        
        Args:
            num_keys: Total number of keys
            num_operations: Total number of operations
            sampling_interval: Sampling interval in seconds
            zipf_theta: Zipfian distribution parameter
            weight_update_hours: Hours between weight updates
            smooth_transition: Whether to use smooth transitions between weights
        """
        self.num_keys = num_keys
        self.num_operations = num_operations
        self.sampling_interval = sampling_interval
        self.zipf_theta = zipf_theta
        self.weight_update_hours = weight_update_hours
        self.smooth_transition = smooth_transition
        self.keys = [f"user{i:08d}" for i in range(num_keys)]
        self.hotspot_patterns = self._define_hotspot_patterns()
        self.trend_patterns = self._define_trend_patterns()
        
        # 缓存权重以避免频繁重新计算
        self.cached_weights = {}
        self.weight_update_interval = int(weight_update_hours * 3600 / sampling_interval)
        
    def _define_hotspot_patterns(self) -> List[Dict]:
        """Define time-based hotspot access patterns with stability periods"""
        patterns = [
            {
                'name': 'morning_peak',
                'description': 'Morning peak pattern',
                'time_ranges': [(6, 10)],  # 延长到4小时
                'hot_keys': list(range(0, 10)),  
                'intensity': 0.8,
                'stability_hours': 4  # 热键保持稳定的小时数
            },
            {
                'name': 'lunch_time',
                'description': 'Noon peak pattern',
                'time_ranges': [(11, 14)],  # 3小时窗口
                'hot_keys': list(range(50, 60)), 
                'intensity': 0.6,
                'stability_hours': 3
            },
            {
                'name': 'evening_peak',
                'description': 'Evening peak pattern', 
                'time_ranges': [(18, 22)],  # 4小时窗口
                'hot_keys': list(range(20, 40)),  
                'intensity': 0.9,
                'stability_hours': 4
            },
            {
                'name': 'night_batch',
                'description': 'Night batch processing',
                'time_ranges': [(0, 6)],  # 6小时窗口
                'hot_keys': list(range(60, 70)), 
                'intensity': 0.4,
                'stability_hours': 6
            },
            {
                'name': 'weekend_pattern',
                'description': 'Weekend access pattern',
                'time_ranges': [(10, 20)],  
                'hot_keys': list(range(80, 120)),  
                'intensity': 0.5,
                'stability_hours': 10,
                'weekdays_only': False
            }
        ]
        return patterns
    
    def _define_trend_patterns(self) -> List[Dict]:
        """Define trend patterns - keys whose popularity changes over time"""
        patterns = [
            {
                'name': 'gradual_rise',
                'hot_keys': list(range(30, 35)),  # 增加键数量
                'start_weight': 0.1,
                'end_weight': 2.0,
                'description': 'Gradual rising trend'
            },
            {
                'name': 'gradual_decline', 
                'hot_keys': list(range(40, 45)),
                'start_weight': 2.0,
                'end_weight': 0.1,
                'description': 'Gradual declining trend'
            },
            {
                'name': 'periodic_wave',
                'hot_keys': list(range(90,95)),
                'amplitude': 1.5,
                'period_hours': 24,
                'description': 'Periodic wave pattern'
            }
        ]
        return patterns
    
    def _get_base_zipfian_weights(self) -> np.ndarray:
        """Generate base Zipfian distribution weights"""
        # 使用固定的随机种子确保基础分布稳定
        np.random.seed(42)
        base_weights = np.random.zipf(self.zipf_theta, self.num_keys)
        np.random.seed()  # 重置随机种子
        
        # 归一化
        base_weights = base_weights.astype(np.float64)
        base_weights = np.maximum(base_weights, 1e-10)
        base_weights = base_weights / np.sum(base_weights)
        
        return base_weights
    
    def _calculate_time_based_adjustment(self, timestamp: datetime) -> np.ndarray:
        """Calculate time-based adjustment factors for weights"""
        hour = timestamp.hour
        weekday = timestamp.weekday()
        is_weekend = weekday >= 5
        
        # 初始化调整因子（1.0表示不调整）
        adjustment_factors = np.ones(self.num_keys, dtype=np.float64)
        
        # 应用热点模式
        for pattern in self.hotspot_patterns:
            # 检查是否在时间范围内
            in_time_range = any(start <= hour < end for start, end in pattern['time_ranges'])
            
            # 检查工作日/周末限制
            weekdays_only = pattern.get('weekdays_only', True)
            if weekdays_only and is_weekend:
                continue
            
            if in_time_range:
                intensity = pattern['intensity']
                hot_keys = pattern['hot_keys']
                
                # 使用高斯分布来平滑热键权重，避免突变
                for key_idx in hot_keys:
                    if key_idx < self.num_keys:
                        # 基础增强
                        base_boost = 1 + intensity * 2.0
                        # 添加一些随机性，但保持相对稳定
                        random_factor = np.random.normal(1.0, 0.05)  # 减小标准差
                        adjustment_factors[key_idx] *= base_boost * random_factor
        
        return adjustment_factors
    
    def _calculate_trend_adjustment(self, progress: float) -> np.ndarray:
        """Calculate trend-based adjustment factors"""
        adjustment_factors = np.ones(self.num_keys, dtype=np.float64)
        
        for pattern in self.trend_patterns:
            pattern_name = pattern['name']
            
            if pattern_name == 'periodic_wave':
                # 正弦波模式
                wave_value = pattern['amplitude'] * np.sin(
                    2 * np.pi * progress * 24 / pattern['period_hours']
                ) + 1.0
                factor = max(wave_value, 0.1)
            elif pattern_name == 'gradual_rise':
                # 渐进上升
                start_w = pattern['start_weight']
                end_w = pattern['end_weight']
                factor = start_w + (end_w - start_w) * progress
            else:  # gradual_decline
                # 渐进下降
                start_w = pattern['start_weight']
                end_w = pattern['end_weight']
                factor = start_w + (end_w - start_w) * progress
            
            # 应用到对应的键
            for key_idx in pattern['hot_keys']:
                if key_idx < self.num_keys:
                    adjustment_factors[key_idx] *= factor
        
        return adjustment_factors
    
    def _interpolate_weights(self, weights1: np.ndarray, weights2: np.ndarray, 
                            alpha: float) -> np.ndarray:
        """Interpolate between two weight arrays for smooth transition"""
        # 使用线性插值实现平滑过渡
        return weights1 * (1 - alpha) + weights2 * alpha
    
    def _get_weights_for_timestamp(self, timestamp: datetime, 
                                  total_duration_hours: float,
                                  current_index: int,
                                  total_indices: int) -> np.ndarray:
        """Get weights for a specific timestamp with caching and smooth transitions"""
        
        # 计算当前进度（0到1）
        progress = current_index / max(total_indices - 1, 1)
        
        # 确定权重更新点
        update_point = (current_index // self.weight_update_interval) * self.weight_update_interval
        next_update_point = update_point + self.weight_update_interval
        
        # 获取或计算基准权重
        if update_point not in self.cached_weights:
            # 获取基础Zipfian权重
            base_weights = self._get_base_zipfian_weights()
            
            # 计算时间调整
            time_adjustment = self._calculate_time_based_adjustment(timestamp)
            
            # 计算趋势调整
            update_progress = update_point / max(total_indices - 1, 1)
            trend_adjustment = self._calculate_trend_adjustment(update_progress)
            
            # 组合所有调整
            combined_weights = base_weights * time_adjustment * trend_adjustment
            
            # 添加轻微噪声保持真实性
            noise = np.random.normal(1.0, 0.02, self.num_keys)  # 减小噪声
            combined_weights *= noise
            
            # 归一化
            combined_weights = np.maximum(combined_weights, 1e-10)
            combined_weights = combined_weights / np.sum(combined_weights)
            
            self.cached_weights[update_point] = combined_weights
        
        current_weights = self.cached_weights[update_point]
        
        # 如果启用平滑过渡，计算下一个权重并插值
        if self.smooth_transition and next_update_point < total_indices:
            if next_update_point not in self.cached_weights:
                # 预计算下一个时间点的权重
                next_timestamp = timestamp + timedelta(hours=self.weight_update_hours)
                base_weights = self._get_base_zipfian_weights()
                time_adjustment = self._calculate_time_based_adjustment(next_timestamp)
                next_progress = next_update_point / max(total_indices - 1, 1)
                trend_adjustment = self._calculate_trend_adjustment(next_progress)
                
                next_weights = base_weights * time_adjustment * trend_adjustment
                noise = np.random.normal(1.0, 0.02, self.num_keys)
                next_weights *= noise
                next_weights = np.maximum(next_weights, 1e-10)
                next_weights = next_weights / np.sum(next_weights)
                
                self.cached_weights[next_update_point] = next_weights
            
            # 计算插值系数
            alpha = (current_index - update_point) / self.weight_update_interval
            return self._interpolate_weights(current_weights, 
                                           self.cached_weights[next_update_point], 
                                           alpha)
        
        return current_weights
    
    def generate_workload(self, start_time: str = "2024-01-01 00:00:00",
                         duration_hours: int = 168) -> pd.DataFrame:
        """
        Generate workload data with stable hot keys
        
        Args:
            start_time: Start time string
            duration_hours: Duration in hours
        
        Returns:
            DataFrame containing operation records
        """
        print(f"Generating workload: {self.num_operations} operations over {duration_hours} hours")
        print(f"Weight update interval: {self.weight_update_hours} hours")
        print(f"Smooth transitions: {self.smooth_transition}")
        
        # Generate time series
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        timestamps = []
        current_time = start_dt
        
        while current_time < start_dt + timedelta(hours=duration_hours):
            timestamps.append(current_time)
            current_time += timedelta(seconds=self.sampling_interval)
        
        # Clear weight cache
        self.cached_weights = {}
        
        operations = []
        operation_id = 0
        
        # Calculate average operations per interval
        avg_ops_per_interval = self.num_operations / len(timestamps)
        
        # Progress tracking
        last_printed_progress = 0
        
        for i, timestamp in enumerate(timestamps):
            # Print progress
            progress = int((i / len(timestamps)) * 100)
            if progress > last_printed_progress and progress % 10 == 0:
                print(f"Progress: {progress}%")
                last_printed_progress = progress
            
            # Get weights for current timestamp
            weights = self._get_weights_for_timestamp(
                timestamp, duration_hours, i, len(timestamps)
            )
            
            # Generate operations for this interval
            ops_this_interval = np.random.poisson(avg_ops_per_interval)
            
            for _ in range(ops_this_interval):
                # Select key based on weights
                key_idx = np.random.choice(self.num_keys, p=weights)
                key_name = self.keys[key_idx]
                
                # Select operation type
                operation_type = np.random.choice(['read', 'UPDATE', 'INSERT'], 
                                                p=[0.7, 0.25, 0.05])
                
                operations.append({
                    'operation_id': operation_id,
                    'timestamp': timestamp,
                    'operation_type': operation_type,
                    'key': key_name,
                    'key_index': key_idx,
                    'hour': timestamp.hour,
                    'weekday': timestamp.weekday(),
                    'is_weekend': timestamp.weekday() >= 5
                })
                
                operation_id += 1
        
        df = pd.DataFrame(operations)
        print(f"Completed: {len(df)} records generated")
        
        # Clear cache after generation
        self.cached_weights = {}
        
        return df
    
    def analyze_workload(self, df: pd.DataFrame) -> Dict:
        """Analyze generated workload"""
        analysis = {}
        
        # Basic statistics
        analysis['total_operations'] = len(df)
        analysis['unique_keys'] = df['key'].nunique()
        analysis['time_range'] = {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max(),
            'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        }
        
        # Hourly distribution 
        df['hour_bucket'] = df['timestamp'].dt.floor('1H')
        hourly_ops = df.groupby('hour_bucket').size()
        analysis['hourly_distribution'] = hourly_ops.to_dict()
        
        # Hot keys
        key_counts = df['key'].value_counts()
        analysis['top_hot_keys'] = key_counts.head(1200).to_dict()
        
        # Operation type distribution
        op_type_dist = df['operation_type'].value_counts()
        analysis['operation_type_distribution'] = op_type_dist.to_dict()
        
        # Analyze hot key stability (how long keys remain hot)
        analysis['hot_key_stability'] = self._analyze_hot_key_stability(df)
        
        return analysis
    
    def _analyze_hot_key_stability(self, df: pd.DataFrame) -> Dict:
        """Analyze how stable hot keys are over time"""
        df['hour_bucket'] = df['timestamp'].dt.floor('1H')
        hourly_buckets = sorted(df['hour_bucket'].unique())
        
        stability_info = {
            'top_keys_per_hour': {},
            'key_persistence': {}
        }
        
        # Track top keys for each hour
        for hour in hourly_buckets:
            hour_df = df[df['hour_bucket'] == hour]
            top_keys = hour_df['key'].value_counts().head(10).index.tolist()
            stability_info['top_keys_per_hour'][hour.isoformat()] = top_keys
        
        # Calculate persistence: how many consecutive hours a key remains in top 10
        all_top_keys = set()
        for keys in stability_info['top_keys_per_hour'].values():
            all_top_keys.update(keys)
        
        for key in list(all_top_keys)[:20]:  # Analyze top 20 keys
            consecutive_hours = 0
            max_consecutive = 0
            
            for hour in hourly_buckets:
                if key in stability_info['top_keys_per_hour'][hour.isoformat()]:
                    consecutive_hours += 1
                    max_consecutive = max(max_consecutive, consecutive_hours)
                else:
                    consecutive_hours = 0
            
            stability_info['key_persistence'][key] = max_consecutive
        
        return stability_info
    
    def save_workload(self, df: pd.DataFrame, output_path: str):
        """Save workload to file"""
        # Save as CSV
        csv_path = f"{output_path}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Workload saved to: {csv_path}")
        
        # Save analysis results
        analysis = self.analyze_workload(df)
        analysis_path = f"{output_path}_analysis.json"
        
        # Convert datetime objects to strings for JSON serialization
        analysis['time_range']['start'] = analysis['time_range']['start'].isoformat()
        analysis['time_range']['end'] = analysis['time_range']['end'].isoformat()
        
        # Convert hourly_distribution datetime keys to strings
        hourly_dist_str = {}
        for k, v in analysis['hourly_distribution'].items():
            if isinstance(k, pd.Timestamp):
                hourly_dist_str[k.isoformat()] = v
            else:
                hourly_dist_str[str(k)] = v
        analysis['hourly_distribution'] = hourly_dist_str
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"Analysis saved to: {analysis_path}")
    
    def visualize_sampled_keys_pattern(self, df: pd.DataFrame, save_dir: str = 'figure'):
        """
        Visualize access patterns for sampled keys
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")
        
        # Sample keys: every 50th key
        sampled_key_indices = list(range(0, min(self.num_keys, 100), 5))[:20]
        
        # Prepare data
        df['hour_bucket'] = df['timestamp'].dt.floor('1H')
        all_time_buckets = sorted(df['hour_bucket'].unique())
        
        # Collect access counts
        key_access_data = {}
        for key_idx in sampled_key_indices:
            key_name = self.keys[key_idx]
            key_df = df[df['key'] == key_name]
            hourly_counts = key_df.groupby('hour_bucket').size()
            
            access_counts = []
            for time_bucket in all_time_buckets:
                if time_bucket in hourly_counts.index:
                    access_counts.append(hourly_counts[time_bucket])
                else:
                    access_counts.append(0)
            
            key_access_data[key_idx] = access_counts
        
        # Create plots
        keys_per_plot = 6
        num_plots = 4
        
        for plot_idx in range(num_plots):
            fig, ax = plt.subplots(figsize=(14, 6))
            
            start_idx = plot_idx * keys_per_plot
            end_idx = min(start_idx + keys_per_plot, len(sampled_key_indices))
            plot_keys = sampled_key_indices[start_idx:end_idx]
            
            colors = plt.cm.tab10(np.linspace(0, 1, keys_per_plot))
            for i, key_idx in enumerate(plot_keys):
                ax.plot(range(len(all_time_buckets)), 
                       key_access_data[key_idx], 
                       label=f'Key {key_idx}',
                       color=colors[i],
                       linewidth=1.5,
                       marker='.',
                       markersize=3,
                       alpha=0.8)
            
            ax.set_title(f'Key Access Pattern Over Time (Keys {plot_keys[0]}-{plot_keys[-1]})', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Access Count', fontsize=12)
            
            # Set x-axis ticks
            if len(all_time_buckets) <= 24:
                ax.set_xticks(range(len(all_time_buckets)))
                ax.set_xticklabels([t.strftime('%m-%d %H:00') for t in all_time_buckets], 
                                  rotation=45, ha='right')
            else:
                step = max(1, len(all_time_buckets) // 24)
                tick_positions = range(0, len(all_time_buckets), step)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([all_time_buckets[i].strftime('%m-%d %H:00') 
                                   for i in tick_positions], 
                                  rotation=45, ha='right')
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper right', fontsize=9, ncol=2)
            
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f'sampled_keys_pattern_{plot_idx+1}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot {plot_idx+1} to: {save_path}")
            
            plt.close()
        
        print(f"\nAll sampled key pattern plots saved to '{save_dir}' directory")
    
    def visualize_patterns(self, df: pd.DataFrame, save_path: str = None):
        """Visualize access patterns including stability analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Hourly operation distribution
        df['hour_bucket'] = df['timestamp'].dt.floor('1H')
        hourly_ops = df.groupby('hour_bucket').size()
        
        x_labels = [ts.strftime('%m-%d %H:00') for ts in hourly_ops.index]
        
        axes[0, 0].bar(range(len(hourly_ops)), hourly_ops.values, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Hourly Operation Distribution')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Number of Operations')
        
        if len(hourly_ops) <= 24:
            axes[0, 0].set_xticks(range(len(hourly_ops)))
            axes[0, 0].set_xticklabels(x_labels, rotation=45, ha='right')
        else:
            step = max(1, len(hourly_ops) // 12)
            tick_positions = range(0, len(hourly_ops), step)
            axes[0, 0].set_xticks(tick_positions)
            axes[0, 0].set_xticklabels([x_labels[i] for i in tick_positions], rotation=45, ha='right')
        
        # 2. Hot key access frequency distribution
        top_keys = df['key'].value_counts().head(100)
        axes[0, 1].plot(range(len(top_keys)), top_keys.values, linewidth=2, color='darkred')
        axes[0, 1].set_title('Top 100 Keys Access Frequency')
        axes[0, 1].set_xlabel('Key Rank')
        axes[0, 1].set_ylabel('Access Count')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Hot key stability over time
        hourly_buckets = sorted(df['hour_bucket'].unique())
        top_k = 5
        
        # Track top-k keys for each hour
        key_ranks = {}
        for hour in hourly_buckets:
            hour_df = df[df['hour_bucket'] == hour]
            top_keys_hour = hour_df['key'].value_counts().head(top_k).index.tolist()
            for rank, key in enumerate(top_keys_hour):
                if key not in key_ranks:
                    key_ranks[key] = []
                key_ranks[key].append((hour, rank))
        
        # Plot top 5 most frequent keys' ranks over time
        global_top_keys = df['key'].value_counts().head(5).index
        colors = plt.cm.tab10(np.linspace(0, 1, 5))
        
        for i, key in enumerate(global_top_keys):
            if key in key_ranks:
                hours = [h for h, r in key_ranks[key]]
                ranks = [r for h, r in key_ranks[key]]
                hour_indices = [hourly_buckets.index(h) for h in hours]
                axes[1, 0].scatter(hour_indices, ranks, label=f'Key {key[-4:]}', 
                                 color=colors[i], s=30, alpha=0.7)
                axes[1, 0].plot(hour_indices, ranks, color=colors[i], alpha=0.3)
        
        axes[1, 0].set_title(f'Top 5 Keys Rank Evolution (Lower is Better)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Rank in Top Keys')
        axes[1, 0].invert_yaxis()
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Operation type distribution
        op_types = df['operation_type'].value_counts()
        axes[1, 1].pie(op_types.values, labels=op_types.index, autopct='%1.1f%%', 
                      startangle=90, colors=['#2ecc71', '#3498db', '#e74c3c'])
        axes[1, 1].set_title('Operation Type Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate time-aware YCSB workload with stable hot keys')
    parser.add_argument('--num_keys', type=int, default = 200, help='Total number of keys')
    parser.add_argument('--num_operations', type=int, default=80000, help='Total number of operations')
    parser.add_argument('--duration_hours', type=int, default=24, help='Duration in hours')
    parser.add_argument('--weight_update_hours', type=float, default=1.0, 
                       help='Hours between weight updates ')
    parser.add_argument('--smooth_transition', type=bool, default=True,
                       help='Use smooth transitions between weights (default: True)')
    parser.add_argument('--output', type=str, default='temporal_ycsb_workload', help='Output file prefix')
    parser.add_argument('--start_time', type=str, default='2024-01-01 00:00:00', help='Start time')
    parser.add_argument('--visualize', action='store_true', default=True, help='Generate visualization charts')
    
    args = parser.parse_args()
    
    # Create generator
    generator = TemporalYCSBGenerator(
        num_keys=args.num_keys,
        num_operations=args.num_operations,
        weight_update_hours=args.weight_update_hours,
        smooth_transition=args.smooth_transition
    )
    
    # Generate workload
    df = generator.generate_workload(
        start_time=args.start_time,
        duration_hours=args.duration_hours
    )
    
    # Save workload
    generator.save_workload(df, args.output)
    
    # Visualize if requested
    if args.visualize:
        generator.visualize_patterns(df, f"{args.output}_patterns.png")
        generator.visualize_sampled_keys_pattern(df)
    
    print("\nWorkload generation completed!")
    
    # Print basic statistics
    analysis = generator.analyze_workload(df)
    print(f"Total operations: {analysis['total_operations']}")
    print(f"Unique keys: {analysis['unique_keys']}")
    print(f"Time range: {analysis['time_range']['duration_hours']:.2f} hours")
    print(f"Top 5 hot keys: {list(analysis['top_hot_keys'].keys())[:5]}")
    
    # Print hot key stability info
    if 'hot_key_stability' in analysis:
        print("\nHot Key Stability Analysis:")
        persistence = analysis['hot_key_stability']['key_persistence']
        if persistence:
            sorted_persistence = sorted(persistence.items(), key=lambda x: x[1], reverse=True)[:5]
            for key, hours in sorted_persistence:
                print(f"  {key}: remained in top-10 for {hours} consecutive hours")

if __name__ == "__main__":
    main()