#include "gnn.h"
#include "core/gnn_core.h"

void gnn_config_init(gnn_config_t* conf)
{
    if (!conf) {
        GNN_LOG_ERROR("Null pointer passed to gnn_config_init");
        return;
    }
    memset(conf, 0, sizeof(gnn_config_t));
    conf->thread_num    = 1;
    conf->device        = GNN_DEV_CPU;
    conf->vulkan_enable = 0;
    conf->save_mid      = 0;
}

gnn_runtime_t* gnn_runtime_create(gnn_backend_t backend,
                                  const char* param_path,
                                  const char* bin_path)
{
    if (!param_path || !bin_path) {
        GNN_LOG_ERROR("Null path passed to gnn_runtime_create");
        return nullptr;
    }

    GNNBackend* b = GNNBackendRegistry::instance().get_backend(backend);
    if (!b) {
        return nullptr;
    }

    return b->create_runtime(param_path, bin_path);
}

gnn_session_t* gnn_session_create(gnn_runtime_t* rt, const gnn_config_t* conf)
{
    if (!rt || !conf) {
        GNN_LOG_ERROR("Null pointer passed to gnn_session_create");
        return nullptr;
    }
    return rt->backend->create_session(rt, conf);
}

gnn_tensor_t* gnn_tensor_get_input(gnn_runtime_t* rt, gnn_session_t* sess, const char* name)
{
    if (!rt || !sess || !name) {
        GNN_LOG_ERROR("Null pointer passed to gnn_tensor_get_input");
        return nullptr;
    }
    return rt->backend->get_input(rt, sess, name);
}

gnn_tensor_t* gnn_tensor_get_output(gnn_runtime_t* rt, gnn_session_t* sess, const char* name)
{
    if (!rt || !sess || !name) {
        GNN_LOG_ERROR("Null pointer passed to gnn_tensor_get_output");
        return nullptr;
    }
    return rt->backend->get_output(rt, sess, name);
}

void gnn_tensor_resize(gnn_runtime_t* rt, gnn_tensor_t* tensor, const int* dims, int ndim)
{
    if (!rt || !tensor || !dims || ndim <= 0) {
        GNN_LOG_ERROR("Invalid parameters passed to gnn_tensor_resize");
        return;
    }
    rt->backend->resize_tensor(rt, tensor, dims, ndim);
}

void gnn_session_resize(gnn_runtime_t* rt, gnn_session_t* sess)
{
    if (!rt || !sess) {
        GNN_LOG_ERROR("Null pointer passed to gnn_session_resize");
        return;
    }
    rt->backend->resize_session(rt, sess);
}

void gnn_tensor_fill_data(gnn_tensor_t* tensor, const void* data, size_t len)
{
    if (!tensor || !tensor->backend || !data || len == 0) {
        GNN_LOG_ERROR("Invalid parameters passed to gnn_tensor_fill_data");
        return;
    }
    tensor->backend->fill_data(tensor, data, len);
}

gnn_error_t gnn_session_run(gnn_runtime_t* rt, gnn_session_t* sess)
{
    if (!rt || !sess) {
        GNN_LOG_ERROR("Null pointer passed to gnn_session_run");
        return GNN_ERROR_NULL_PTR;
    }
    return (gnn_error_t)rt->backend->run_session(rt, sess);
}

const void* gnn_tensor_get_data(gnn_tensor_t* tensor, size_t* out_len)
{
    if (!tensor || !tensor->backend || !out_len) {
        GNN_LOG_ERROR("Invalid parameters passed to gnn_tensor_get_data");
        return nullptr;
    }
    *out_len = 0;
    return tensor->backend->get_data(tensor, out_len);
}

void gnn_session_release(gnn_runtime_t* rt, gnn_session_t* sess)
{
    if (!rt || !sess) {
        GNN_LOG_ERROR("Null pointer passed to gnn_session_release");
        return;
    }
    rt->backend->release_session(rt, sess);
}

void gnn_runtime_destroy(gnn_runtime_t* rt)
{
    if (!rt) {
        GNN_LOG_ERROR("Null pointer passed to gnn_runtime_destroy");
        return;
    }
    rt->backend->destroy_runtime(rt);
}

const char* gnn_version() {
    return "1.0.0";
}
