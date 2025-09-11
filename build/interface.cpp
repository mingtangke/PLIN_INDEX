#include <iostream>
#include <fstream>
#include <vector>
#include <string>
int main(void){
    std::ifstream infile_data("hot_keys.csv");
    std::ifstream infile_cache("Data.txt");
    if (!infile_data.is_open()) {
        std::cerr << "Failed to open file: hot_keys.csv" << std::endl;
    }
    if( !infile_cache.is_open()) {
        std::cerr << "Failed to open file: Data.txt" << std::endl;
    }
    std::vector<std::pair<double,long>> hot_keys;
    double key;
    long payload;
    while (infile_data >> key >> payload) {
        hot_keys.emplace_back(key, payload);
    }
    infile_data.close();
    std::vector<std::pair<double,long>> cache;
    while (infile_cache >> key >> payload) {
        cache.emplace_back(key, payload);
    }
    infile_cache.close();

    int count = 0;
    for(auto &i:hot_keys){
        bool found = false;
        for(auto &j:cache){
            if(i.first == j.first && i.second == j.second){
                found = true;
                count ++;
                break;
            }
        }
        if(!found){
            std::cout<<i.first<<" "<<i.second<<std::endl;
        }
    }


    std::cout<<"valid hot key: "<<count<<std::endl;
    return 0;
}