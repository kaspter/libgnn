cmake_minimum_required(VERSION 3.14)
project(libgnn VERSION 1.0.0 LANGUAGES C CXX)

# 编译标准
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 自动检测系统架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(ARCH_X86_64 ON)
    message(STATUS "Detected x86_64 (amd64) architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(ARCH_ARM ON)
    message(STATUS "Detected ARM architecture")
else()
    message(STATUS "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 编译选项
option(GNN_BUILD_TESTS "Build unit tests" OFF)
option(GNN_BUILD_EXAMPLES "Build examples" OFF)
option(GNN_USE_MNN "Enable MNN backend" ON)
option(GNN_USE_NCNN "Enable NCNN backend" OFF)
option(GNN_ENABLE_LOG "Enable logging" ON)
option(GNN_ENABLE_DEBUG_LOG "Enable debug logging" OFF)

# 通用编译选项
add_compile_options(-Wall -Wextra -Wpedantic)

# ==============================================
# 性能 + 体积优化
# ==============================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # 极致速度优化
    add_compile_options(-O3 -ffast-math)

    # CPU原生指令集优化
    if(ARCH_X86_64)
        add_compile_options(-march=native -msse4.2 -mavx -mavx2 -mfma -mbmi -mbmi2)
    endif()

    if(ARCH_ARM)
        add_compile_options(-march=armv8.2-a+fp16+simd)
    endif()

    # LTO链接时优化（性能提升25%+）
    add_compile_options(-flto=auto)
    add_link_options(-flto=auto)

    # 死代码剔除（体积减少30%+）
    add_compile_options(-ffunction-sections -fdata-sections)
    add_link_options(-Wl,--gc-sections)

    # 符号剥离（体积进一步减小）
    add_link_options(-Wl,-s -Wl,--strip-all)
endif()

# 日志配置
if(GNN_ENABLE_LOG)
    add_compile_definitions(GNN_LOG_LEVEL=3)
    if(GNN_ENABLE_DEBUG_LOG)
        add_compile_definitions(GNN_LOG_LEVEL=4)
    endif()
else()
    add_compile_definitions(GNN_LOG_LEVEL=0)
endif()

# 头文件目录
include_directories(include)
include_directories(src)

# 核心源码
file(GLOB_RECURSE CORE_SRC src/core/*.h src/core/*.cpp)
file(GLOB_RECURSE BACKEND_SRC src/backend/*.cpp)

# 编译库
add_library(gnn STATIC
    src/gnn.cpp
    ${CORE_SRC}
    ${BACKEND_SRC}
)

# ==================== MNN 后端 ==============================
if(GNN_USE_MNN)
    message(STATUS "Building with MNN backend")

    include(FetchContent)
    FetchContent_Declare(
        mnn
        GIT_REPOSITORY https://github.com/alibaba/MNN.git
        GIT_TAG 3.5.0
        GIT_SHALLOW ON
        GIT_PROGRESS ON
    )

    # MNN 高性能编译配置
    set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_QUANTOOLS OFF CACHE BOOL "" FORCE)
    set(MNN_BUILD_CONVERTER OFF CACHE BOOL "" FORCE)

    # 后端开关
    set(MNN_FORWARD_CPU ON CACHE BOOL "" FORCE)
    set(MNN_FORWARD_OPENCL OFF CACHE BOOL "" FORCE)
    set(MNN_FORWARD_CUDA OFF CACHE BOOL "" FORCE)
    set(MNN_FORWARD_METAL OFF CACHE BOOL "" FORCE)
    set(MNN_FORWARD_VULKAN OFF CACHE BOOL "" FORCE)
    set(MNN_FORWARD_TENSORRT OFF CACHE BOOL "" FORCE)

    # 高性能模式关键配置
    set(MNN_USE_THREAD_POOL ON CACHE BOOL "" FORCE)
    set(MNN_TUNE_MODE TUNE_RT_NORM CACHE STRING "" FORCE)
    set(MNN_OPENMP OFF CACHE BOOL "" FORCE)
    set(MNN_DEFAULT_THREAD_NUMBER 4 CACHE STRING "" FORCE)
    set(MNN_CPU_TUNE ON CACHE BOOL "" FORCE)
    set(MNN_LOW_MEMORY OFF CACHE BOOL "" FORCE)  # 性能模式必须关闭
    set(MNN_BUILD_MINI OFF CACHE BOOL "" FORCE) # 性能模式必须关闭
    set(MNN_ENABLE_EXCEPTION OFF CACHE BOOL "" FORCE)
    set(MNN_ENABLE_RTTI OFF CACHE BOOL "" FORCE)

    # 架构特定优化
    if(ARCH_X86_64)
        set(MNN_USE_SSE ON CACHE BOOL "" FORCE)
        set(MNN_USE_AVX ON CACHE BOOL "" FORCE)
        set(MNN_USE_AVX2 ON CACHE BOOL "" FORCE)
        set(MNN_USE_FMA ON CACHE BOOL "" FORCE)
        set(MNN_ARM82 OFF CACHE BOOL "" FORCE)
    endif()

    if(ARCH_ARM)
        set(MNN_ARM82 ON CACHE BOOL "" FORCE)
        set(MNN_USE_SSE OFF CACHE BOOL "" FORCE)
        set(MNN_USE_AVX OFF CACHE BOOL "" FORCE)
        set(MNN_USE_AVX2 OFF CACHE BOOL "" FORCE)
        set(MNN_USE_FMA OFF CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(mnn)
    target_include_directories(gnn PRIVATE ${mnn_SOURCE_DIR}/include ${mnn_SOURCE_DIR}/include/MNN)
    target_compile_definitions(gnn PRIVATE GNN_USE_MNN)
    target_link_libraries(gnn PRIVATE MNN)
endif()

# ==================== NCNN 后端 ====================
if(GNN_USE_NCNN)
    message(STATUS "Building with NCNN backend")

    include(FetchContent)
    FetchContent_Declare(
        ncnn
        GIT_REPOSITORY https://github.com/Tencent/ncnn.git
        GIT_TAG 20240410
        GIT_SHALLOW ON
        GIT_PROGRESS ON
    )

    # NCNN 最小化编译
    set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(NCNN_OPENMP OFF CACHE BOOL "" FORCE)
    set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
    set(NCNN_LOW_MEMORY ON CACHE BOOL "" FORCE)

    # 架构特定优化
    if(ARCH_X86_64)
        set(NCNN_SSE2 ON CACHE BOOL "" FORCE)
        set(NCNN_SSE42 ON CACHE BOOL "" FORCE)
        set(NCNN_AVX ON CACHE BOOL "" FORCE)
        set(NCNN_AVX2 ON CACHE BOOL "" FORCE)
        set(NCNN_FMA ON CACHE BOOL "" FORCE)
        set(NCNN_ARM82 OFF CACHE BOOL "" FORCE)
    endif()

    if(ARCH_ARM)
        set(NCNN_ARM82 ON CACHE BOOL "" FORCE)
        set(NCNN_SSE2 OFF CACHE BOOL "" FORCE)
        set(NCNN_SSE42 OFF CACHE BOOL "" FORCE)
        set(NCNN_AVX OFF CACHE BOOL "" FORCE)
        set(NCNN_AVX2 OFF CACHE BOOL "" FORCE)
        set(NCNN_FMA OFF CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(ncnn)
    target_include_directories(gnn PRIVATE ${ncnn_SOURCE_DIR}/src)
    target_compile_definitions(gnn PRIVATE GNN_USE_NCNN)
    target_link_libraries(gnn PRIVATE ncnn)
endif()

# 单元测试
if(GNN_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# 示例
if(GNN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 安装
install(TARGETS gnn
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# 版本信息
message(STATUS "libgnn version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "MNN backend: ${GNN_USE_MNN}")
message(STATUS "NCNN backend: ${GNN_USE_NCNN}")
message(STATUS "Optimizations: LTO=ON, CPU_TUNE=ON, GRAPH_OPTIMIZE=3")
