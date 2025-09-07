// // database_logger.cpp
// #include "hot_key.h"
// #include <iostream>
// #include <chrono>

// DatabaseLogger::DatabaseLogger(const std::string& log_file, 
//                               const std::string& python_host, 
//                               int python_port)
//     : python_host_(python_host), python_port_(python_port) {
//         log_file_.open(log_file, std::ios::app);
//         if (!log_file_.is_open()) {
//         throw std::runtime_error("Failed to open log file");
//         }
// }

// DatabaseLogger::~DatabaseLogger() {
//     stop();
//     if (log_file_.is_open()) {
//         log_file_.close();
//     }
//     if (sockfd_ != -1) {
//         close(sockfd_);
//     }
// }

// void DatabaseLogger::start() {
//     running_ = true;
    
//     // Setup socket connection to Python
//     sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
//     if (sockfd_ < 0) {
//         throw std::runtime_error("Socket creation failed");
//     }
    
//     struct sockaddr_in serv_addr;
//     serv_addr.sin_family = AF_INET;
//     serv_addr.sin_port = htons(python_port_);
    
//     if (inet_pton(AF_INET, python_host_.c_str(), &serv_addr.sin_addr) <= 0) {
//         throw std::runtime_error("Invalid address/Address not supported");
//     }
    
//     if (connect(sockfd_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
//         throw std::runtime_error("Connection to Python failed");
//     }
    
//     // Start threads
//     logging_thread_ = std::thread(&DatabaseLogger::logging_thread, this);
//     comm_thread_ = std::thread(&DatabaseLogger::communication_thread, this);
// }

// void DatabaseLogger::stop() {
//     running_ = false;
//     if (logging_thread_.joinable()) {
//         logging_thread_.join();
//     }
//     if (comm_thread_.joinable()) {
//         comm_thread_.join();
//     }
// }

// void DatabaseLogger::log_query(const std::string& key) {
//     std::lock_guard<std::mutex> lock(log_mutex_);
//     log_queue_.push(key);
// }

// void DatabaseLogger::logging_thread() {
//     size_t current_index = 0;
    
//     while (running_) {
//         std::vector<std::string> batch;
        
//         {
//             std::lock_guard<std::mutex> lock(log_mutex_);
//             while (!log_queue_.empty()) {
//                 batch.push_back(log_queue_.front());
//                 log_queue_.pop();
//             }
//         }
        
//         if (!batch.empty()) {
//             // Write to log file
//             for (const auto& key : batch) {
//                 log_file_ << key << "\n";
//                 current_index++;
//             }
//             log_file_.flush();
            
//             // Send current index to Python
//             std::string message = "INDEX:" + std::to_string(current_index - batch.size()) + 
//                                  ":" + std::to_string(current_index);
//             send(sockfd_, message.c_str(), message.length(), 0);
//         }
        
//         std::this_thread::sleep_for(std::chrono::milliseconds(100));
//     }
// }

// void DatabaseLogger::communication_thread() {
//     char buffer[1024] = {0};
    
//     while (running_) {
//         // Receive hot keys from Python
//         int valread = read(sockfd_, buffer, 1024);
//         if (valread > 0) {
//             std::string message(buffer, valread);
            
//             if (message.find("HOT_KEYS:") == 0) {
//                 // Parse hot keys
//                 std::string keys_str = message.substr(9);
//                 size_t pos = 0;
//                 std::vector<std::string> hot_keys;
                
//                 while ((pos = keys_str.find(",")) != std::string::npos) {
//                     hot_keys.push_back(keys_str.substr(0, pos));
//                     keys_str.erase(0, pos + 1);
//                 }
//                 hot_keys.push_back(keys_str);
                
//                 // Process hot keys (save to file, use for optimization, etc.)
//                 std::ofstream hotkey_file("hot_keys.txt", std::ios::app);
//                 for (const auto& key : hot_keys) {
//                     hotkey_file << key << "\n";
//                 }
//                 hotkey_file.close();
//             }
//         }
        
//         std::this_thread::sleep_for(std::chrono::milliseconds(100));
//     }
// }