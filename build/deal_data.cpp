#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main() {
    std::string inputFileName = "//home//ming//桌面//PLIN-N //PLIN-N//build//command.txt";
    std::ifstream inputFile(inputFileName);

    if (!inputFile.is_open()) {
        std::cerr << "无法打开输入文件: " << inputFileName << std::endl;
        return 1;
    }

    // 读取所有行
    std::vector<std::string> lines;
    std::string line;
    int count = 0;
    while (std::getline(inputFile, line)) {
        lines.push_back(line);
        count++;
    }
    std::cout<<count<<std::endl;
    inputFile.close();

    for (int i = 0; i <= 9; i++) {
        std::string outputFileName = "command" + std::to_string(i) + ".txt";
        std::ofstream outputFile(outputFileName);
        for (int j = 0; j < 5000000; j++) {
            outputFile << lines[i*5000000+j] << "\n";
        }
        outputFile<<"bye";
        outputFile.close();
    }
    
}