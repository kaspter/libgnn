#ifndef __GNN_LOG_H__
#define __GNN_LOG_H__

#include <cstdio>

#define GNN_LOG_LEVEL_ERROR 1
#define GNN_LOG_LEVEL_WARN  2
#define GNN_LOG_LEVEL_INFO  3
#define GNN_LOG_LEVEL_DEBUG 4

#ifndef GNN_LOG_LEVEL
#define GNN_LOG_LEVEL GNN_LOG_LEVEL_INFO
#endif

#define GNN_LOG_ERROR(fmt, ...) \
    if (GNN_LOG_LEVEL >= GNN_LOG_LEVEL_ERROR) \
        fprintf(stderr, "[GNN ERROR] " fmt "\n", ##__VA_ARGS__)

#define GNN_LOG_WARN(fmt, ...) \
    if (GNN_LOG_LEVEL >= GNN_LOG_LEVEL_WARN) \
        fprintf(stderr, "[GNN WARN] " fmt "\n", ##__VA_ARGS__)

#define GNN_LOG_INFO(fmt, ...) \
    if (GNN_LOG_LEVEL >= GNN_LOG_LEVEL_INFO) \
        fprintf(stdout, "[GNN INFO] " fmt "\n", ##__VA_ARGS__)

#define GNN_LOG_DEBUG(fmt, ...) \
    if (GNN_LOG_LEVEL >= GNN_LOG_LEVEL_DEBUG) \
        fprintf(stdout, "[GNN DEBUG] " fmt "\n", ##__VA_ARGS__)

#endif
