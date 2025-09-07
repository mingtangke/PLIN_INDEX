#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "parameters.h"
#include <unordered_map>
#include <map>

class DatabaseLogger {
public:
    DatabaseLogger(const std::string& log_file, const std::string& python_host, int python_port);
    ~DatabaseLogger();
    
    void start();
    void try_start();
    void stop();
    void log_query( _key_t & key,_payload_t & payload);
    bool prehot_cache = false;
    std::unordered_map<_key_t,_payload_t> hot_map_;
    std::unordered_map<_key_t,_payload_t> log_map_;

private:
    
    // std::thread logging_thread_;
    std::thread comm_thread_;
    std::atomic<bool> running_{false};

    bool retrain = true;
    int start_index = 0;
    int end_index = 0;
    int transform_count = 0;
    bool transfer_complete = false;

    
    
    
    std::mutex log_mutex_;
    std::queue<_key_t> log_queue_;
    // std::unordered_map<_key_t,_payload_t> log_map_;
    std::mutex hot_map_mutex_;
    
    size_t HOT_CACHE = 3000000;
    size_t MAX_BUFFER_SIZE = 1000000;
    size_t MAX_QUEUE_SIZE = 50000;
    std::ofstream log_file_;

    int sockfd_{-1};
    std::string python_host_;
    int python_port_;

    void communication_thread();
};


DatabaseLogger::DatabaseLogger(const std::string& log_file, const std::string& python_host, int python_port)
    : python_host_(python_host), python_port_(python_port) {
        log_file_.open(log_file, std::ios::app);
        if (!log_file_.is_open()) {
        throw std::runtime_error("Failed to open log file");
        }
}

DatabaseLogger::~DatabaseLogger() {
    stop();
    if (log_file_.is_open()) {
        log_file_.close();
    }
    if (sockfd_ != -1) {
        close(sockfd_);
    }
}


void DatabaseLogger::log_query( _key_t & key,_payload_t & payload){
    std::lock_guard<std::mutex> lock(log_mutex_);
    log_queue_.push(key);
    log_map_[key] = payload;

    if(log_queue_.size() > MAX_QUEUE_SIZE){
        while( !log_queue_.empty()){
            log_file_ << std::fixed << log_queue_.front() << "\n";
            log_queue_.pop();
        }
        log_file_.flush();
        end_index = end_index + MAX_QUEUE_SIZE - 1;
        std::cout<<"end_index: "<<end_index<<std::endl;
    }
}

// void DatabaseLogger::start(){
//     std::ifstream infile("hot_keys.csv");
//     if(infile.is_open()){
//         std::string line;
//         while (std::getline(infile, line)) {
//             size_t space_pos = line.find(' ');
//             if (space_pos != std::string::npos) {
//                 _key_t key = std::stod(line.substr(0, space_pos));
//                 _payload_t payload = std::stoi(line.substr(space_pos + 1));
//                 hot_map_[key] = payload;
//             }
//         }
//         infile.close();
//         prehot_cache = true;
//         std::cout<<"Load hot key finished! hot key size: "<<hot_map_.size()<<std::endl;
//     }
// }


void DatabaseLogger::start() {
    running_ = true;
    
    // Setup socket connection to lstm_server
    sockfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd_ < 0) {
        throw std::runtime_error("Socket creation failed");
    }
    
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(python_port_);
    
    if (inet_pton(AF_INET, python_host_.c_str(), &serv_addr.sin_addr) <= 0) {
        throw std::runtime_error("Invalid address/Address not supported");
    }
    
    if (connect(sockfd_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        throw std::runtime_error("Connection to Python failed");
    }
    comm_thread_ = std::thread(&DatabaseLogger::communication_thread, this);
}

void DatabaseLogger::stop() {
    running_ = false;
    if (comm_thread_.joinable()) {
        comm_thread_.join();
    }
}



void DatabaseLogger::communication_thread() {

    std::vector<char> buffer(MAX_BUFFER_SIZE, 0);
    std::ofstream hotkey_file("hot_keys.csv", std::ios::app);
    hotkey_file << 'hot_key_count' << transform_count << "\n";
    transform_count++;
    std::cout << "Communication thread started" << std::endl;

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000; // 100毫秒

    fd_set readfds;
    std::string hot_message = "";

    while (running_) {
        if (start_index == 0 && end_index >= HOT_CACHE && retrain) {
            retrain = false;
            std::string message = "INDEX:" + std::to_string(start_index) + ":" + std::to_string(end_index);
            std::cout << "Send to python: " << message << std::endl;
            ssize_t sent_bytes = send(sockfd_, message.c_str(), message.length(), 0);
            if (sent_bytes == -1) {
                perror("send failed");
            } else {
                start_index = end_index + 1;
            }
        }

        FD_ZERO(&readfds);
        FD_SET(sockfd_, &readfds);
        
        int activity = select(sockfd_ + 1, &readfds, NULL, NULL, &tv);
        
        if (activity < 0) {
            perror("select error");
            break;
        } else if (activity == 0) {
            continue;
        } else {
            std::cout << "C++ received python data" <<std::endl;
            int valread = read(sockfd_, buffer.data(), MAX_BUFFER_SIZE - 1);
            if (valread > 0) {
                buffer[valread] = '\0';
                std::string message(buffer.data());
                if (!transfer_complete) {
                    hot_message += message;
                }
                if (message.find("END") != std::string::npos) {
                    transfer_complete = true;
                }
            } else if (valread == 0) {
                std::cout << "Python connection closed" << std::endl;
                break;
            } else {
                perror("read error");
                break;
            }
        }

        if(hot_message.find("END") != std::string::npos && hot_message.find("HOT_KEYS:") == 0 ){
            std::lock_guard<std::mutex> lock(hot_map_mutex_);
            hot_map_.clear();
            std::string keys_str = hot_message.substr(9, hot_message.length() - 12);
            size_t pos = 0;

            while ((pos = keys_str.find(",")) != std::string::npos) {
                _key_t key = std::stod(keys_str.substr(0, pos));
                _payload_t payload = log_map_[key]; 
                keys_str.erase(0, pos + 1);
                hot_map_[key] = payload;
                hotkey_file << std::fixed << key << " "<<payload<<"\n";
            }

            _key_t key = std::stod(keys_str);
            _payload_t payload = log_map_[key];
             hot_map_[key] = payload;

            hotkey_file << std::fixed << key <<" "<<payload<<"\n";
            hotkey_file.close();
            prehot_cache = true;
            transfer_complete = false;
            hot_message = "";
        }
    }
}
