// Copyright (c) Yiming Guo.
#include "hot_model.h"
#include <sstream>
#include <set>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>  // For cudaMemGetInfo
#include <c10/cuda/CUDACachingAllocator.h>  // For c10::cuda::CUDACachingAllocator::emptyCache

// KeySequenceDataset Implementation
torch::data::Example<> KeySequenceDataset::get(size_t index) {
    auto input_windows = sequences[index];
    std::vector<int64_t> target_windows_flat;
    
    // Get next windows for target
    if (index + 1 < sequences.size()) {
        auto next_windows = sequences[index + 1];
        // Get last target_windows_size windows
        size_t start_idx = next_windows.size() > target_windows_size ? 
                          next_windows.size() - target_windows_size : 0;
        
        for (size_t i = start_idx; i < next_windows.size(); ++i) {
            for (auto key_idx : next_windows[i]) {
                target_windows_flat.push_back(key_idx);
            }
        }
    }
    
    // Count target key frequencies
    std::unordered_map<int64_t, int> target_key_counter;
    for (auto key : target_windows_flat) {
        target_key_counter[key]++;
    }
    
    // Get top k hot keys
    std::vector<std::pair<int64_t, int>> key_counts(target_key_counter.begin(), target_key_counter.end());
    std::sort(key_counts.begin(), key_counts.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<int64_t> target_indices;
    for (size_t i = 0; i < std::min(static_cast<size_t>(top_k_hot_keys), key_counts.size()); ++i) {
        target_indices.push_back(key_counts[i].first);
    }
    
    // Pad target if necessary
    while (target_indices.size() < top_k_hot_keys) {
        target_indices.push_back(0);
    }
    
    // Create input tensor
    torch::Tensor input_tensor = torch::zeros({static_cast<int64_t>(input_windows.size()), window_size}, 
                                              torch::kLong);
    for (size_t i = 0; i < input_windows.size(); ++i) {
        for (size_t j = 0; j < std::min(input_windows[i].size(), static_cast<size_t>(window_size)); ++j) {
            input_tensor[i][j] = input_windows[i][j];
        }
    }
    
    // Create target tensor
    torch::Tensor target_tensor = torch::tensor(target_indices, torch::kLong);
    
    return {input_tensor, target_tensor};
}

// Hot_Model Implementation
Hot_Model::Hot_Model() : device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
    if (device.is_cuda()) {
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    }
    std::cout << "MAX_VOCAB_SIZE is set to: " << MAX_VOCAB_SIZE << std::endl;
}

void Hot_Model::get_test_data() {
    std::string data_file = "//home//ming//桌面//PLIN-N //PLIN-N//data//processed_key.csv";
    std::ifstream infile(data_file);
    
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << data_file << std::endl;
        return;
    }
    
    std::string line;
    // Skip header if exists
    std::getline(infile, line);
    int maxline = 1000000;
    int count = 0;
    while (std::getline(infile, line) && count < maxline  ) {
        try {
            _key_t key = std::stoll(line);
            logs.push_back(key);
            count++;
            
        } catch (const std::exception& e) {
            // Skip invalid lines
            continue;
        }
    }
    infile.close();
    
    std::cout << "Loaded " << logs.size() << " keys from file" << std::endl;
}

void Hot_Model::preprocess_data() {
    std::cout << "Starting data preprocessing with MAX_VOCAB_SIZE = " << MAX_VOCAB_SIZE << std::endl;
    
    // Create windows
    windows.clear();
    int64_t total_records = logs.size();
    int64_t window_step = WINDOW_SIZE;  // No overlap
    
    for (int64_t i = 0; i <= total_records - WINDOW_SIZE; i += window_step) {
        std::vector<_key_t> window;
        for (int64_t j = i; j < i + WINDOW_SIZE; ++j) {
            window.push_back(logs[j]);
        }
        windows.push_back(window);
    }
    
    std::cout << "Created " << windows.size() << " windows" << std::endl;
    
    // Count key frequencies - THIS IS CRITICAL FOR REDUCING VOCABULARY SIZE
    std::unordered_map<_key_t, int64_t> key_freq;
    for (const auto& window : windows) {
        for (const auto& key : window) {
            key_freq[key]++;
        }
    }
    
    std::cout << "Original unique keys: " << key_freq.size() << std::endl;
    
    // Sort keys by frequency and take top MAX_VOCAB_SIZE
    std::vector<std::pair<_key_t, int64_t>> freq_pairs(key_freq.begin(), key_freq.end());
    std::sort(freq_pairs.begin(), freq_pairs.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Create vocabulary from top frequent keys ONLY
    key_to_idx.clear();
    idx_to_key.clear();
    key_to_idx[0] = 0;  // Padding token
    idx_to_key[0] = 0;
    key_to_idx[-1] = 1;  // Unknown token
    idx_to_key[1] = -1;
    
    int64_t idx = 2;
    // CRITICAL: Only take top MAX_VOCAB_SIZE-2 keys (accounting for padding and unknown)
    int64_t vocab_limit = std::min(static_cast<int64_t>(freq_pairs.size()), MAX_VOCAB_SIZE - 2);
    
    for (int64_t i = 0; i < vocab_limit; ++i) {
        key_to_idx[freq_pairs[i].first] = idx;
        idx_to_key[idx] = freq_pairs[i].first;
        idx++;
    }
    
    std::cout << "Limited vocabulary size: " << key_to_idx.size() 
              << " (max allowed: " << MAX_VOCAB_SIZE << ")" << std::endl;
    
    // Print top 10 most frequent keys
    std::cout << "Top 10 most frequent keys:" << std::endl;
    for (int i = 0; i < std::min(10, static_cast<int>(freq_pairs.size())); ++i) {
        std::cout << "  Key " << freq_pairs[i].first 
                  << ": " << freq_pairs[i].second << " occurrences" << std::endl;
    }
}

void Hot_Model::create_sequences() {
    sequences.clear();
    
    // Convert windows to sequences of indices
    std::vector<std::vector<int64_t>> indexed_windows;
    for (const auto& window : windows) {
        std::vector<int64_t> indexed_window;
        for (const auto& key : window) {
            // IMPORTANT: Use unknown token (1) for keys not in vocabulary
            auto it = key_to_idx.find(key);
            if (it != key_to_idx.end()) {
                indexed_window.push_back(it->second);
            } else {
                indexed_window.push_back(1);  // Unknown token
            }
        }
        indexed_windows.push_back(indexed_window);
    }
    
    // Create sequences
    for (size_t i = 0; i <= indexed_windows.size() - SEQUENCE_LENGTH; ++i) {
        std::vector<std::vector<int64_t>> seq;
        for (size_t j = i; j < i + SEQUENCE_LENGTH; ++j) {
            seq.push_back(indexed_windows[j]);
        }
        sequences.push_back(seq);
    }
    
    std::cout << "Created " << sequences.size() << " training sequences" << std::endl;
}

torch::Tensor Hot_Model::set_based_loss(torch::Tensor predictions, torch::Tensor targets, 
                                        float base_emphasis, float max_emphasis) {
    auto batch_size = predictions.size(0);
    auto vocab_size = predictions.size(1);
    
    // Create multi-hot encoding
    auto target_multihot = torch::zeros({batch_size, vocab_size}, 
                                        torch::TensorOptions().dtype(torch::kFloat32).device(predictions.device()));
    auto actual_key_counts = torch::zeros({batch_size}, 
                                          torch::TensorOptions().device(predictions.device()));
    
    for (int64_t i = 0; i < batch_size; ++i) {
        auto target_row = targets[i];
        auto non_zero_mask = target_row != 0;
        auto actual_keys = target_row.masked_select(non_zero_mask);
        
        if (actual_keys.numel() > 0) {
            target_multihot.index_put_({i, actual_keys}, 1.0);
            actual_key_counts[i] = actual_keys.numel();
        }
    }
    
    // Calculate adaptive emphasis factor
    auto avg_key_count = actual_key_counts.mean();
    auto emphasis_factor = base_emphasis + (max_emphasis - base_emphasis) * 
                          torch::sigmoid(-(avg_key_count - 5) / 2);
    
    // Calculate sigmoid probability
    auto pred_prob = torch::sigmoid(predictions);
    
    // Focal loss components
    float alpha = 0.25;
    float gamma = 2.0;
    
    auto bce_loss = torch::binary_cross_entropy(pred_prob, target_multihot, {}, torch::Reduction::None);
    auto p_t = target_multihot * pred_prob + (1 - target_multihot) * (1 - pred_prob);
    auto modulating_factor = torch::pow(1 - p_t, gamma);
    auto alpha_factor = target_multihot * alpha + (1 - target_multihot) * (1 - alpha);
    auto focal_loss = alpha_factor * modulating_factor * bce_loss;
    
    // Heat enhancement
    auto heat_enhancement = target_multihot * torch::pow(1 - pred_prob, emphasis_factor);
    
    // Combined loss
    auto total_loss = focal_loss + heat_enhancement;
    
    return total_loss.mean();
}

void Hot_Model::train() {
    std::cout << "\n=== Starting Training ===" << std::endl;
    std::cout << "Configuration: MAX_VOCAB_SIZE=" << MAX_VOCAB_SIZE 
              << ", BATCH_SIZE=" << BATCH_SIZE 
              << ", EMBEDDING_DIM=" << EMBEDDING_DIM 
              << ", HIDDEN_SIZE=" << HIDDEN_SIZE << std::endl;
    
    // Memory monitoring
    if (device.is_cuda()) {
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        std::cout << "GPU Memory: " << free_memory / (1024*1024) << " MB free / " 
                  << total_memory / (1024*1024) << " MB total" << std::endl;
    }
    
    // Load and preprocess data
    get_test_data();
    if (logs.empty()) {
        std::cerr << "No data loaded. Exiting training." << std::endl;
        return;
    }
    
    preprocess_data();
    create_sequences();
    
    // Initialize model with LIMITED vocabulary size
    int64_t vocab_size = key_to_idx.size();
    std::cout << "Initializing model with vocabulary size: " << vocab_size 
              << " (should be <= " << MAX_VOCAB_SIZE << ")" << std::endl;
    
    if (vocab_size > MAX_VOCAB_SIZE) {
        std::cerr << "ERROR: Vocabulary size exceeds MAX_VOCAB_SIZE!" << std::endl;
        return;
    }
    
    model = EfficientKeyPredictionLSTM(vocab_size, WINDOW_SIZE, EMBEDDING_DIM, 
                                       HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE);
    model->to(device);
    
    // Calculate model size
    int64_t total_params = 0;
    for (const auto& p : model->parameters()) {
        total_params += p.numel();
    }
    std::cout << "Total model parameters: " << total_params << " (" 
              << (total_params * 4) / (1024*1024) << " MB)" << std::endl;
    
    // Check memory again after model creation
    if (device.is_cuda()) {
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        std::cout << "GPU Memory after model init: " << free_memory / (1024*1024) << " MB free" << std::endl;
        
        if (free_memory < 1024*1024*1024) {  // Less than 1GB free
            std::cerr << "Warning: Low GPU memory! Consider reducing MAX_VOCAB_SIZE or BATCH_SIZE." << std::endl;
        }
    }
    
    // Split data into train/val/test
    size_t total_samples = sequences.size();
    size_t train_end = static_cast<size_t>(total_samples * 0.7);
    size_t val_end = static_cast<size_t>(total_samples * 0.9);
    
    std::vector<std::vector<std::vector<int64_t>>> train_seq(sequences.begin(), sequences.begin() + train_end);
    std::vector<std::vector<std::vector<int64_t>>> val_seq(sequences.begin() + train_end, sequences.begin() + val_end);
    std::vector<std::vector<std::vector<int64_t>>> test_seq(sequences.begin() + val_end, sequences.end());
    
    std::cout << "Train samples: " << train_seq.size() << std::endl;
    std::cout << "Val samples: " << val_seq.size() << std::endl;
    std::cout << "Test samples: " << test_seq.size() << std::endl;
    
    // Clear sequences to free memory
    sequences.clear();
    sequences.shrink_to_fit();
    
    // Create datasets and dataloaders
    auto train_dataset = KeySequenceDataset(train_seq, key_to_idx, WINDOW_SIZE, TOP_K_HOT_KEYS, PREDICTION_WIN)
        .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), BATCH_SIZE);
    
    auto val_dataset = KeySequenceDataset(val_seq, key_to_idx, WINDOW_SIZE, TOP_K_HOT_KEYS, PREDICTION_WIN)
        .map(torch::data::transforms::Stack<>());
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_dataset), BATCH_SIZE);
    
    // Optimizer
    torch::optim::AdamW optimizer(model->parameters(), 
                                  torch::optim::AdamWOptions(LEARNING_RATE).weight_decay(1e-4));
    
    // Training loop
    float best_val_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    const int patience = 10;
    
    for (int64_t epoch = 0; epoch < EPOCHS; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Training phase
        model->train();
        float train_loss = 0.0;
        int64_t train_batches = 0;
        int count = 0 ;
        for (auto& batch : *train_loader) {
              std::cout << count << std::endl;
              count++;
            try {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                
                optimizer.zero_grad();
                auto output = model->forward(data);
                auto loss = set_based_loss(output, target);
                
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                optimizer.step();
                
                train_loss += loss.item<float>();
                train_batches++;
                
                // // Clear cache periodically to prevent memory buildup
                // if (train_batches % 10 == 0 && device.is_cuda()) {
                //     c10::cuda::CUDACachingAllocator::emptyCache();
                // }
                
            } catch (const c10::Error& e) {
                std::cerr << "CUDA error in training batch: " << e.what() << std::endl;
                // if (device.is_cuda()) {
                //     c10::cuda::CUDACachingAllocator::emptyCache();
                // }
                continue;
            }
        }
        
        float avg_train_loss = train_loss / train_batches;
      

        // Validation phase
        model->eval();
        float val_loss = 0.0;
        int64_t val_batches = 0;
        
        torch::NoGradGuard no_grad;
        for (auto& batch : *val_loader) {
            try {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                
                auto output = model->forward(data);
                auto loss = set_based_loss(output, target);
                
                val_loss += loss.item<float>();
                val_batches++;
                
            } catch (const c10::Error& e) {
                std::cerr << "CUDA error in validation batch: " << e.what() << std::endl;
                continue;
            }
        }
        
        float avg_val_loss = val_loss / val_batches;
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        
        std::cout << "Epoch [" << epoch + 1 << "/" << EPOCHS << "] - " << duration.count() << "s" << std::endl;
        std::cout << "  Train Loss: " << avg_train_loss << ", Val Loss: " << avg_val_loss << std::endl;
        
        // Memory status
        if (device.is_cuda()) {
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            std::cout << "  GPU Memory: " << free_memory / (1024*1024) << " MB free" << std::endl;
        }
        
        // Early stopping
        if (avg_val_loss < best_val_loss) {
            best_val_loss = avg_val_loss;
            patience_counter = 0;
            save_model("best_hot_key_model.pt");
        } else {
            patience_counter++;
            if (patience_counter >= patience) {
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                break;
            }
        }
    }
    
    // Load best model
    load_model("best_hot_key_model.pt");
    std::cout << "Training completed!" << std::endl;
}

void Hot_Model::predict() {
    if (!model) {
        std::cerr << "Model not trained or loaded!" << std::endl;
        return;
    }
    
    // Example prediction using recent windows
    if (windows.size() >= SEQUENCE_LENGTH) {
        std::vector<std::vector<_key_t>> recent_windows(
            windows.end() - SEQUENCE_LENGTH, windows.end());
        
        auto predicted_keys = predict_next_hot_keys(recent_windows, 100);
        
        std::cout << "\nPredicted top 100 hot keys:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(10), predicted_keys.size()); ++i) {
            std::cout << "  Key " << i + 1 << ": " << predicted_keys[i] << std::endl;
        }
    }
}

std::vector<_key_t> Hot_Model::predict_next_hot_keys(const std::vector<std::vector<_key_t>>& recent_windows, 
                                                     int top_k) {
    if (recent_windows.size() < SEQUENCE_LENGTH) {
        std::cerr << "Need at least " << SEQUENCE_LENGTH << " windows for prediction" << std::endl;
        return {};
    }
    
    // Prepare input
    std::vector<std::vector<_key_t>> input_windows(
        recent_windows.end() - SEQUENCE_LENGTH, recent_windows.end());
    
    // Convert to indices
    torch::Tensor input_tensor = torch::zeros({1, SEQUENCE_LENGTH, WINDOW_SIZE}, torch::kLong);
    
    for (size_t i = 0; i < SEQUENCE_LENGTH; ++i) {
        for (size_t j = 0; j < std::min(input_windows[i].size(), size_t(WINDOW_SIZE)); ++j) {
            auto it = key_to_idx.find(input_windows[i][j]);
            if (it != key_to_idx.end()) {
                input_tensor[0][i][j] = it->second;
            } else {
                input_tensor[0][i][j] = 1;  // Unknown token
            }
        }
    }
    
    input_tensor = input_tensor.to(device);
    
    // Predict
    model->eval();
    torch::NoGradGuard no_grad;
    
    auto predictions = model->forward(input_tensor);
    predictions = torch::softmax(predictions, -1);
    
    auto [scores, indices] = predictions.topk(top_k, -1);
    
    // Convert indices back to keys
    std::vector<_key_t> predicted_keys;
    auto indices_accessor = indices.accessor<int64_t, 2>();
    
    for (int i = 0; i < top_k; ++i) {
        int64_t idx = indices_accessor[0][i];
        auto it = idx_to_key.find(idx);
        if (it != idx_to_key.end() && it->second != -1) {  // Skip unknown token
            predicted_keys.push_back(it->second);
        }
    }
    
    return predicted_keys;
}

void Hot_Model::save_model(const std::string& filepath) {
    torch::save(model, filepath);
    
    // Save additional metadata
    std::ofstream meta_file(filepath + ".meta");
    meta_file << key_to_idx.size() << std::endl;
    for (const auto& [key, idx] : key_to_idx) {
        meta_file << key << " " << idx << std::endl;
    }
    meta_file.close();
    
    std::cout << "Model saved to " << filepath << std::endl;
}

void Hot_Model::load_model(const std::string& filepath) {
    // Load metadata
    std::ifstream meta_file(filepath + ".meta");
    if (!meta_file.is_open()) {
        std::cerr << "Cannot open metadata file" << std::endl;
        return;
    }
    
    size_t vocab_size;
    meta_file >> vocab_size;
    
    key_to_idx.clear();
    idx_to_key.clear();
    
    _key_t key;
    int64_t idx;
    while (meta_file >> key >> idx) {
        key_to_idx[key] = idx;
        idx_to_key[idx] = key;
    }
    meta_file.close();
    
    // Initialize and load model
    model = EfficientKeyPredictionLSTM(vocab_size, WINDOW_SIZE, EMBEDDING_DIM, 
                                       HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE);
    torch::load(model, filepath);
    model->to(device);
    
    std::cout << "Model loaded from " << filepath << std::endl;
}
