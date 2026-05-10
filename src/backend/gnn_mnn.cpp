#include "../core/gnn_core.h"

#ifdef GNN_USE_MNN
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"
#include <mutex>

using namespace MNN;

// MNN 等价优化配置
// 提前设置环境变量开启最高级别图优化
static void __attribute__((constructor)) mnn_optimize_init() {
    setenv("MNN_GRAPH_OPTIMIZE_LEVEL", "3", 1); // 最高级别图优化
    setenv("MNN_CPU_NUM_THREADS", "4", 1);      // 默认线程数
    setenv("MNN_USE_THREAD_POOL", "1", 1);      // 启用全局线程池
}

// MNN 私有实现结构
struct MNNRuntimeImpl {
    std::unique_ptr<Interpreter> interpreter;
};

struct MNNSessionImpl {
    Session* session;
    std::once_flag warmup_flag; // 推理预热标志
};

struct MNNTensorImpl {
    Tensor* tensor;
};

class MNNBackend : public GNNBackend {
public:
    gnn_runtime_t* create_runtime(const char* param_path, const char* bin_path) override {
        (void)param_path; // MNN 只需要 bin 路径

        auto impl = std::make_unique<MNNRuntimeImpl>();
        impl->interpreter.reset(Interpreter::createFromFile(bin_path));
        if (!impl->interpreter) {
            GNN_LOG_ERROR("Failed to load MNN model: %s", bin_path);
            return nullptr;
        }

        gnn_runtime_t* rt = new gnn_runtime_t();
        rt->backend_type = GNN_BACKEND_MNN;
        rt->backend = this;
        rt->impl = impl.release();
        GNN_LOG_INFO("MNN runtime created successfully");
        return rt;
    }

    gnn_session_t* create_session(gnn_runtime_t* rt, const gnn_config_t* conf) override {
        MNNRuntimeImpl* runtime_impl = static_cast<MNNRuntimeImpl*>(rt->impl);

        ScheduleConfig sc;
        sc.numThread = conf->thread_num;
        sc.type = static_cast<MNNForwardType>(conf->device);
        
        // MNN 全版本兼容处理（2.x-3.6.x）
#if MNN_VERSION_MAJOR >= 3
        if (conf->save_mid) {
            sc.saveTensors = std::vector<std::string>();
        }
#else
        sc.saveTensors = conf->save_mid;
#endif

        auto impl = std::make_unique<MNNSessionImpl>();
        impl->session = runtime_impl->interpreter->createSession(sc);
        if (!impl->session) {
            GNN_LOG_ERROR("Failed to create MNN session");
            return nullptr;
        }

        gnn_session_t* sess = new gnn_session_t();
        sess->impl = impl.release();
        GNN_LOG_DEBUG("MNN session created with %d threads", conf->thread_num);
        return sess;
    }

    gnn_tensor_t* get_input(gnn_runtime_t* rt, gnn_session_t* sess, const char* name) override {
        MNNRuntimeImpl* runtime_impl = static_cast<MNNRuntimeImpl*>(rt->impl);
        MNNSessionImpl* session_impl = static_cast<MNNSessionImpl*>(sess->impl);

        auto impl = std::make_unique<MNNTensorImpl>();
        impl->tensor = runtime_impl->interpreter->getSessionInput(session_impl->session, name);
        if (!impl->tensor) {
            GNN_LOG_ERROR("Input tensor not found: %s", name);
            return nullptr;
        }
        
        gnn_tensor_t* t = new gnn_tensor_t();
        t->backend = this;
        t->impl = impl.release();
        return t;
    }

    gnn_tensor_t* get_output(gnn_runtime_t* rt, gnn_session_t* sess, const char* name) override {
        MNNRuntimeImpl* runtime_impl = static_cast<MNNRuntimeImpl*>(rt->impl);
        MNNSessionImpl* session_impl = static_cast<MNNSessionImpl*>(sess->impl);

        auto impl = std::make_unique<MNNTensorImpl>();
        impl->tensor = runtime_impl->interpreter->getSessionOutput(session_impl->session, name);
        if (!impl->tensor) {
            GNN_LOG_ERROR("Output tensor not found: %s", name);
            return nullptr;
        }
        
        gnn_tensor_t* t = new gnn_tensor_t();
        t->backend = this;
        t->impl = impl.release();
        return t;
    }

    void resize_tensor(gnn_runtime_t* rt, gnn_tensor_t* tensor, const int* dims, int ndim) override {
        MNNRuntimeImpl* runtime_impl = static_cast<MNNRuntimeImpl*>(rt->impl);
        MNNTensorImpl* tensor_impl = static_cast<MNNTensorImpl*>(tensor->impl);

        std::vector<int> d(dims, dims + ndim);
        runtime_impl->interpreter->resizeTensor(tensor_impl->tensor, d);
    }

    void resize_session(gnn_runtime_t* rt, gnn_session_t* sess) override {
        MNNRuntimeImpl* runtime_impl = static_cast<MNNRuntimeImpl*>(rt->impl);
        MNNSessionImpl* session_impl = static_cast<MNNSessionImpl*>(sess->impl);

        runtime_impl->interpreter->resizeSession(session_impl->session);
    }

    void fill_data(gnn_tensor_t* tensor, const void* data, size_t len) override {
        MNNTensorImpl* tensor_impl = static_cast<MNNTensorImpl*>(tensor->impl);

        memcpy(tensor_impl->tensor->host<void>(), data, len);
    }

    int run_session(gnn_runtime_t* rt, gnn_session_t* sess) override {
        MNNRuntimeImpl* runtime_impl = static_cast<MNNRuntimeImpl*>(rt->impl);
        MNNSessionImpl* session_impl = static_cast<MNNSessionImpl*>(sess->impl);

        // 推理预热：第一次空跑，消除首次推理慢的问题
        std::call_once(session_impl->warmup_flag, [&](){
            GNN_LOG_DEBUG("Performing MNN session warmup...");
            runtime_impl->interpreter->runSession(session_impl->session);
        });

        ErrorCode ret = runtime_impl->interpreter->runSession(session_impl->session);
        if (ret != NO_ERROR) {
            GNN_LOG_ERROR("MNN inference failed with error: %d", ret);
            return GNN_ERROR_INFERENCE;
        }
        return GNN_OK;
    }

    const void* get_data(gnn_tensor_t* tensor, size_t* out_len) override {
        MNNTensorImpl* tensor_impl = static_cast<MNNTensorImpl*>(tensor->impl);
        *out_len = tensor_impl->tensor->size() * sizeof(float);
        return tensor_impl->tensor->host<void>();
    }

    void release_session(gnn_runtime_t* rt, gnn_session_t* sess) override {
        MNNRuntimeImpl* runtime_impl = static_cast<MNNRuntimeImpl*>(rt->impl);
        MNNSessionImpl* session_impl = static_cast<MNNSessionImpl*>(sess->impl);

        runtime_impl->interpreter->releaseSession(session_impl->session);
        delete session_impl;
        delete sess;
        GNN_LOG_DEBUG("MNN session released");
    }

    void destroy_runtime(gnn_runtime_t* rt) override {
        MNNRuntimeImpl* runtime_impl = static_cast<MNNRuntimeImpl*>(rt->impl);

        delete runtime_impl;
        delete rt;
        GNN_LOG_INFO("MNN runtime destroyed");
    }
};

// 自动注册 MNN 后端
GNN_REGISTER_BACKEND(GNN_BACKEND_MNN, MNNBackend)
#endif
