cmake_minimum_required(VERSION 3.14)
project(gnn)

set(CMAKE_CXX_STANDARD 11)

# 头文件目录
include_directories(include)
include_directories(src)

# 核心源码
file(GLOB CORE_SRC src/*.cpp)

# 后端源码
file(GLOB BACKEND_SRC src/backend/*.cpp)

# 编译库
add_library(gnn STATIC
    ${CORE_SRC}
    ${BACKEND_SRC}
)
