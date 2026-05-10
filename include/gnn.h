#ifndef __GNN_CAPI_H__
#define __GNN_CAPI_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 错误码定义
typedef enum {
    GNN_OK = 0,
    GNN_ERROR_NULL_PTR,
    GNN_ERROR_INVALID_PARAM,
    GNN_ERROR_MODEL_LOAD,
    GNN_ERROR_SESSION_CREATE,
    GNN_ERROR_TENSOR_NOT_FOUND,
    GNN_ERROR_INFERENCE,
    GNN_ERROR_BACKEND_NOT_SUPPORTED,
    GNN_ERROR_MEMORY_ALLOC
} gnn_error_t;

/* 推理后端枚举 */
typedef enum {
    GNN_BACKEND_MNN     = 0,
    GNN_BACKEND_NCNN    = 1,
    GNN_BACKEND_AMLNN   = 2,
    GNN_BACKEND_RKNN    = 3,
    GNN_BACKEND_TNN     = 4
} gnn_backend_t;

/* 运行设备枚举 */
typedef enum {
    GNN_DEV_CPU         = 0,
    GNN_DEV_GPU         = 1,
    GNN_DEV_NPU         = 2
} gnn_device_t;

/* 不透明句柄 */
typedef struct gnn_runtime_t  gnn_runtime_t;
typedef struct gnn_session_t  gnn_session_t;
typedef struct gnn_tensor_t   gnn_tensor_t;

/* 推理配置结构体 */
typedef struct {
    int             thread_num;
    gnn_device_t    device;
    int             vulkan_enable;
    int             save_mid;
    uint8_t reserve[24];
} gnn_config_t;

void gnn_config_init(gnn_config_t* conf);

gnn_runtime_t* gnn_runtime_create(gnn_backend_t backend,
                                  const char* param_path,
                                  const char* bin_path);

gnn_session_t* gnn_session_create(gnn_runtime_t* rt, const gnn_config_t* conf);

gnn_tensor_t* gnn_tensor_get_input(gnn_runtime_t* rt, gnn_session_t* sess, const char* name);
gnn_tensor_t* gnn_tensor_get_output(gnn_runtime_t* rt, gnn_session_t* sess, const char* name);

void gnn_tensor_resize(gnn_runtime_t* rt, gnn_tensor_t* tensor, const int* dims, int ndim);
void gnn_session_resize(gnn_runtime_t* rt, gnn_session_t* sess);

void gnn_tensor_fill_data(gnn_tensor_t* tensor, const void* data, size_t len);
gnn_error_t  gnn_session_run(gnn_runtime_t* rt, gnn_session_t* sess);

const void* gnn_tensor_get_data(gnn_tensor_t* tensor, size_t* out_len);

void gnn_session_release(gnn_runtime_t* rt, gnn_session_t* sess);
void gnn_runtime_destroy(gnn_runtime_t* rt);

#ifdef __cplusplus
}
#endif

#endif
