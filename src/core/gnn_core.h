#ifndef __GNN_CORE_H__
#define __GNN_CORE_H__

#include "gnn.h"
#include "gnn_log.h"
#include <map>
#include <cstring>
#include <vector>
#include <string>
#include <memory>

// 统一后端接口（所有后端必须实现）
class GNNBackend {
public:
    virtual ~GNNBackend() = default;

    virtual gnn_runtime_t* create_runtime(const char* param_path, const char* bin_path) = 0;
    virtual gnn_session_t* create_session(gnn_runtime_t* rt, const gnn_config_t* conf) = 0;

    virtual gnn_tensor_t* get_input(gnn_runtime_t* rt, gnn_session_t* sess, const char* name) = 0;
    virtual gnn_tensor_t* get_output(gnn_runtime_t* rt, gnn_session_t* sess, const char* name) = 0;

    virtual void resize_tensor(gnn_runtime_t* rt, gnn_tensor_t* tensor, const int* dims, int ndim) = 0;
    virtual void resize_session(gnn_runtime_t* rt, gnn_session_t* sess) = 0;

    virtual void fill_data(gnn_tensor_t* tensor, const void* data, size_t len) = 0;
    virtual int run_session(gnn_runtime_t* rt, gnn_session_t* sess) = 0;

    virtual const void* get_data(gnn_tensor_t* tensor, size_t* out_len) = 0;

    virtual void release_session(gnn_runtime_t* rt, gnn_session_t* sess) = 0;
    virtual void destroy_runtime(gnn_runtime_t* rt) = 0;
};

// 后端注册器（单例 + 自动注册）
class GNNBackendRegistry {
public:
    static GNNBackendRegistry& instance() {
        static GNNBackendRegistry inst;
        return inst;
    }

    void register_backend(gnn_backend_t type, std::unique_ptr<GNNBackend> backend) {
        backends_[type] = std::move(backend);
        GNN_LOG_INFO("Registered backend: %d", type);
    }

    GNNBackend* get_backend(gnn_backend_t type) {
        auto it = backends_.find(type);
        if (it == backends_.end()) {
            GNN_LOG_ERROR("Backend %d not supported", type);
            return nullptr;
        }
        return it->second.get();
    }

private:
    GNNBackendRegistry() = default;
    std::map<gnn_backend_t, std::unique_ptr<GNNBackend>> backends_;
};

// 后端自动注册宏（添加新后端只需要这一行）
#define GNN_REGISTER_BACKEND(type, cls) \
    static class cls##Register { \
    public: \
        cls##Register() { \
            GNNBackendRegistry::instance().register_backend(type, std::make_unique<cls>()); \
        } \
    } g_##cls##_register;

// 运行时内部结构（保存后端指针）
struct gnn_runtime_t {
    gnn_backend_t backend_type;
    GNNBackend* backend;
    void* impl; // 后端私有实现
};

// 会话内部结构
struct gnn_session_t {
    void* impl; // 后端私有实现
};

// 张量内部结构
struct gnn_tensor_t {
    GNNBackend* backend; // 所属后端接口指针
    void* impl; // 后端私有实现
};

#endif
