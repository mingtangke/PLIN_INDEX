#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>



void serialize_vector(const std::vector<t_record>& table, std::ostream& os) {
    size_t size = table.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));  // 序列化容器大小
    for (const auto& record : table) {
        record.serialize(os);  // 序列化每个 t_record
    }
}

void deserialize_vector(std::vector<t_record>& table, std::istream& is) {
    size_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    table.resize(size);
    for (auto& record : table) {
        record.deserialize(is);  // 反序列化每个 t_record
    }
}

// 序列化 std::vector<Param>
void serialize_vector_of_params(const std::vector<Param>& vec, std::ostream& os) {
    size_t size = vec.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));  // 序列化 vector 大小
    for (const auto& param : vec) {
        param.serialize(os);  // 序列化每个 Param
    }
}

// 序列化 std::vector<std::vector<Param>>
void serialize_parameter(const std::vector<std::vector<Param>>& parameter, std::ostream& os) {
    size_t outer_size = parameter.size();
    std::cout<<"outer_size"<<outer_size<<std::endl; // 输出外层 vector 大小
    os.write(reinterpret_cast<const char*>(&outer_size), sizeof(outer_size));  // 序列化外层 vector 大小

    for (const auto& inner_vec : parameter) {
        serialize_vector_of_params(inner_vec, os);  // 序列化每个内层 vector<Param>
    }
}

void deserialize_vector_of_params(std::vector<Param>& vec, std::istream& is) {
    size_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));  // 读取 vector 大小
    vec.resize(size);
    for (auto& param : vec) {
        param.deserialize(is);  // 反序列化每个 Param
    }
}

void deserialize_parameter(std::vector<std::vector<Param>>& parameter, std::istream& is) {
    size_t outer_size;
    is.read(reinterpret_cast<char*>(&outer_size), sizeof(outer_size));  // 读取外层 vector 大小
    std::cout<<"outer_size"<<outer_size<<std::endl;
    if(outer_size > 100){
        exit(0);
    }
    parameter.resize(outer_size);

    for (auto& inner_vec : parameter) {
        deserialize_vector_of_params(inner_vec, is);  // 反序列化每个内层 std::vector<Param>
    }
}