# # cmake_minimum_required(VERSION 3.16)

# # project(Plin)

# # set(CMAKE_CXX_STANDARD 20)

# # set(CMAKE_CXX_FLAGS "-g -march=native") 
# # # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -g")
# # # set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")

# # # link_directories(/usr/lib/gcc/x86_64-linux-gnu/10/)

# # add_executable(c client.cpp)
# # add_executable(s server.cpp)
# # add_executable(t test_demo.cpp)

# # # add_compile_options(-fsanitize=thread)

# # # add_link_options(-fsanitize=thread)

# # # target_link_libraries(c -pthread -ltsan)
# # # target_link_libraries(s -pthread -ltsan)

# # target_link_libraries(c -pthread)
# # target_link_libraries(s -pthread)
# # target_link_libraries(t -pthread)

# # set(CMAKE_BUILD_TYPE Debug)
# cmake_minimum_required(VERSION 3.10)
# project(PLIN_N)

# # 设置 C++17 标准
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 手动设置 LibTorch 路径
# set(LIBTORCH_PATH /usr/local/libtorch-cuda)

# # 包含目录
# include_directories(
#     ${LIBTORCH_PATH}/include
#     ${LIBTORCH_PATH}/include/torch/csrc/api/include
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )

# # 链接目录
# link_directories(${LIBTORCH_PATH}/lib)

# # 添加可执行文件
# add_executable(plin_n main.cpp hot_model.cpp)

# # 链接库
# target_link_libraries(plin_n
#     torch
#     torch_cpu
#     torch_cuda
#     c10
#     c10_cuda
#     pthread
# }

# # 设置运行时库路径
# set_target_properties(plin_n PROPERTIES
#     INSTALL_RPATH "${LIBTORCH_PATH}/lib"
#     BUILD_WITH_INSTALL_RPATH TRUE
# )
cmake_minimum_required(VERSION 3.17)
project(HotKeyPredictor LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
# find_package(CUDA 12.1 REQUIRED)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Set LibTorch path (adjust this path to your LibTorch installation)
# Download LibTorch from: https://pytorch.org/get-started/locally/
# Select: Stable, Linux, LibTorch, C++/Java, CUDA 12.1
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libtorch")

# Find Torch
find_package(Torch REQUIRED)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# CUDA specific flags for better performance
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")  # Adjust based on your GPU architecture
# Common architectures:
# - sm_60: Pascal (GTX 10xx)
# - sm_70: Volta (V100)
# - sm_75: Turing (RTX 20xx, GTX 16xx)
# - sm_80: Ampere (A100)
# - sm_86: Ampere (RTX 30xx)
# - sm_89: Ada Lovelace (RTX 40xx)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    main.cpp
    hot_model.cpp
    # Add your utils.cpp and other source files here if needed
)

# Create executable
add_executable(hot_key_predictor ${SOURCES})

# Link libraries
target_link_libraries(hot_key_predictor
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_curand_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    cuda  # Add CUDA runtime library
    pthread
)

# Set properties for CUDA
set_property(TARGET hot_key_predictor PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Copy LibTorch DLLs/SOs to build directory (for runtime)
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET hot_key_predictor
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:hot_key_predictor>)
endif()

# Installation rules
install(TARGETS hot_key_predictor DESTINATION bin)

# Print configuration information
message(STATUS "")
message(STATUS "Configuration:")
message(STATUS "  CUDA version: ${CUDA_VERSION}")
message(STATUS "  CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "  LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "  C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "  CUDA compiler flags: ${CMAKE_CUDA_FLAGS}")
message(STATUS "")

