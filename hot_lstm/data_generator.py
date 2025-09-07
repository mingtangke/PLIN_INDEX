# Yiming Guo 08/14/2025
"""
./bin/ycsb run basic -p recordcount=100 -p operationcount=500000 -p workload=site.ycsb.workloads.CoreWorkload -p requestdistribution=hotspot -p hotspotdatafraction=0.2 -p hotspotopnfraction=0.8 -p readproportion=1.0 -p insertorder=ordered -p updateproportion=0 -s > data/tracea_load.txt
"""
import os
import subprocess
import time
import csv
import pandas as pd
import numpy as np
from collections import Counter
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import subprocess
import os

def run_ycsb_command():
    ycsb_dir = "/home/ming/桌面/PLIN-N /PLIN-N/YCSB"
    original_dir = os.getcwd()
    os.chdir(ycsb_dir)

    # load
    load_command = [
        "./bin/ycsb", "load", "basic",
        "-p", "recordcount=10000000",    # total 10000000 10000000
        "-p", "operationcount=3000000",
        "-p", "workload=site.ycsb.workloads.CoreWorkload",
        "-p", "requestdistribution=zipfian",
        # "-p", "hotspotdatafraction=0.2",
        # "-p", "hotspotopnfraction=0.8",
        "-p", "readproportion=1.0",
        "-p", "insertorder=ordered",
        "-p", "updateproportion=0",
        "-p","readallfields=true",
        "-s"
    ]
    
    with open("/home/ming/桌面/PLIN-N /PLIN-N/data/tracea_load.txt", "w") as f:
        subprocess.run(load_command, stdout=f, stderr=subprocess.PIPE, text=True)

    # run
    run_command = [
        "./bin/ycsb", "run", "basic",
        "-p", "recordcount=10000000",  
        "-p", "operationcount=3000000", 
        "-p", "workload=site.ycsb.workloads.CoreWorkload",
        "-p", "requestdistribution=zipfian",
        # "-p", "hotspotdatafraction=0.2",
        # "-p", "hotspotopnfraction=0.8",
        "-p", "readproportion=1.0",
        "-p", "insertorder=ordered",
        "-p", "updateproportion=0",
        "-p","readallfields=true",
        "-s"
    ]
    f.close()
    
    with open("/home/ming/桌面/PLIN-N /PLIN-N/data/tracea_run.txt", "w") as f1:
        subprocess.run(run_command, stdout=f1, stderr=subprocess.PIPE, text=True)

    os.chdir(original_dir)


def extract_keys_from_ycsb_log(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            log_content = f.readlines()
        
        key_counts = defaultdict(int)
        total_keys = 0
        keys = []
        
        for line in log_content:
            match = None
            if 'READ usertable' in line:
                match = re.search(r'READ usertable (\w+)', line)
            elif 'UPDATE usertable' in line:
                match = re.search(r'UPDATE usertable (\w+)', line)
            elif 'DELETE usertable' in line:
                match = re.search(r'DELETE usertable (\w+)', line)
            
            if match:
                key = match.group(1)
                keys.append(match.group(1)[4:])
                if key.startswith("user"):
                    key = key[4:]
                key_counts[key] += 1
                total_keys += 1
        

        sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("key\n")
            for key in keys:
                f.write(f"{key}\n")
        
        print(f"extract {len(keys)} keys to {output_file}")
        
        with open(output_file_summary, 'w', encoding='utf-8') as f:
            f.write(f"总键访问次数: {total_keys}\n")
            f.write(f"唯一键数量: {len(key_counts)}\n\n")
            f.write("键访问统计 (按访问次数降序):\n")
            for key, count in sorted_keys:
                f.write(f"{key}: {count}\n")
        
        print(f"成功处理 {total_keys} 次键访问，涉及 {len(key_counts)} 个唯一键")
        print(f"结果已保存到: {output_file}")
        print("\n前10个最常访问的键:")
        for i, (key, count) in enumerate(sorted_keys[:10], 1):
            print(f"{i}. {key}: {count} 次")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"处理文件时出错: {e}")

# extract_keys_from_ycsb_log(input_file, output_file)
if __name__ == "__main__":
    input_file = "/home/ming/桌面/PLIN-N /PLIN-N/data/tracea_run.txt"
    output_file = "/home/ming/桌面/PLIN-N /PLIN-N/data/processed_key.csv"
    output_file_summary = "/home/ming/桌面/PLIN-N /PLIN-N/data/processed_key_summary.txt"
    run_ycsb_command()
    extract_keys_from_ycsb_log(input_file, output_file)