// Copyright (c) Yiming Guo.
#ifndef HOT_MODEL_H
#define HOT_MODEL_H

#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>

// Assuming these are defined in your headers
#include "utils.h"
#include "parameters.h"

// LSTM Model Definition
struct EfficientKeyPredictionLSTMImpl : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear fc{nullptr};
    
    int64_t vocab_size;
    int64_t window_size;
    int64_t embedding_dim;
    int64_t hidden_size;
    
    EfficientKeyPredictionLSTMImpl(int64_t vocab_size, 
                                   int64_t window_size,
                                   int64_t embedding_dim = 8,
                                   int64_t hidden_size = 16,
                                   int64_t num_layers = 2,
                                   double dropout_rate = 0.3) 
        : vocab_size(vocab_size), 
          window_size(window_size),
          embedding_dim(embedding_dim),
          hidden_size(hidden_size) {
        
        // Vocab size already includes padding and unknown tokens
        embedding = register_module("embedding", 
            torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim).padding_idx(0)));
        
        lstm = register_module("lstm", 
            torch::nn::LSTM(torch::nn::LSTMOptions(embedding_dim * window_size, hidden_size)
                .num_layers(num_layers)
                .dropout(num_layers > 1 ? dropout_rate : 0.0)  // Only apply dropout if num_layers > 1
                .batch_first(true)));
        
        fc = register_module("fc", torch::nn::Linear(hidden_size, vocab_size));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // x shape: [batch, seq_len, window_size]
        auto sizes = x.sizes();
        int64_t batch = sizes[0];
        int64_t seq_len = sizes[1];
        int64_t win_size = sizes[2];
        
        // Reshape for embedding
        x = x.view({batch * seq_len, win_size});
        auto emb = embedding->forward(x);  // [batch*seq_len, win_size, emb_dim]
        emb = emb.view({batch, seq_len, -1});  // [batch, seq_len, win_size*emb_dim]
        
        // LSTM forward
        auto lstm_output = lstm->forward(emb);
        auto lstm_out = std::get<0>(lstm_output);
        
        // Get last output
        auto last_out = lstm_out.select(1, seq_len - 1);  // [batch, hidden_size]
        
        return fc->forward(last_out);
    }
};
TORCH_MODULE(EfficientKeyPredictionLSTM);

// Dataset class for key sequences
class KeySequenceDataset : public torch::data::Dataset<KeySequenceDataset> {
private:
    std::vector<std::vector<std::vector<int64_t>>> sequences;
    std::unordered_map<_key_t, int64_t> key_to_idx;
    int64_t window_size;
    int64_t top_k_hot_keys;
    int64_t target_windows_size;
    int64_t vocab_size;
    
public:
    KeySequenceDataset(const std::vector<std::vector<std::vector<int64_t>>>& seqs,
                       const std::unordered_map<_key_t, int64_t>& key_idx,
                       int64_t win_size,
                       int64_t top_k,
                       int64_t target_win_size)
        : sequences(seqs), key_to_idx(key_idx), window_size(win_size),
          top_k_hot_keys(top_k), target_windows_size(target_win_size) {
        vocab_size = key_to_idx.size();
    }
    
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override {
        return sequences.size() - 1;  // We need next window for target
    }
};

// Main Hot Model class
class Hot_Model {
public:
    std::unordered_map<_key_t, _payload_t> cache_map;
    std::vector<_key_t> logs;
    
    // Model parameters - THESE ARE THE KEY CONFIGURATION PARAMETERS
    static constexpr int64_t WINDOW_SIZE = 500;
    static constexpr int64_t SEQUENCE_LENGTH = 300;
    static constexpr int64_t TOP_K_HOT_KEYS = 10000;
    static constexpr int64_t BATCH_SIZE = 128;  // Reduced for memory efficiency
    static constexpr int64_t EPOCHS = 5;
    static constexpr int64_t PREDICTION_WIN = 300;
    static constexpr int64_t EMBEDDING_DIM = 16;  // Reduced from 16
    static constexpr int64_t HIDDEN_SIZE = 32;  // Reduced from 32
    static constexpr int64_t NUM_LAYERS = 1;  // Increased to fix dropout warning
    static constexpr double LEARNING_RATE = 0.001;
    static constexpr double DROPOUT_RATE = 0.3;
    
    // THIS IS THE CRITICAL PARAMETER TO LIMIT VOCABULARY SIZE
    static constexpr int64_t MAX_VOCAB_SIZE = 50000;  // Limit vocabulary size to prevent OOM
    
private:
    // Model and data structures
    EfficientKeyPredictionLSTM model{nullptr};
    std::unordered_map<_key_t, int64_t> key_to_idx;
    std::unordered_map<int64_t, _key_t> idx_to_key;
    std::vector<std::vector<_key_t>> windows;
    std::vector<std::vector<std::vector<int64_t>>> sequences;
    
    torch::Device device;
    
public:
    Hot_Model();
    ~Hot_Model() {}
    
    void train();
    void predict();
    std::vector<_key_t> predict_next_hot_keys(const std::vector<std::vector<_key_t>>& recent_windows, 
                                               int top_k = TOP_K_HOT_KEYS);
    
private:
    void get_test_data();
    int get_log_size() { return logs.size(); }
    void preprocess_data();
    void create_sequences();
    torch::Tensor set_based_loss(torch::Tensor predictions, torch::Tensor targets, 
                                 float base_emphasis = 1.0, float max_emphasis = 3.0);
    void save_model(const std::string& filepath);
    void load_model(const std::string& filepath);
};

#endif // HOT_MODEL_H