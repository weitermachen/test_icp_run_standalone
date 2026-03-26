// Author: weitermachen
// Time: 2026-03-24

#pragma once

#define SUCCESS                                         0

// 节点状态
#define NODE_STATUS_NOT_RUN                             1
#define NODE_STATUS_RUNNING                             2


// 实例化异常
#define ALGORITHM_INSTANCE_ERROR                        -1
#define INSTANCE_NOT_EXIST                              -2


// 分配异常
#define ALLOC_ID_ERROR                                  -10


// 算法运行异常
#define PARAMS_BIND_ERROR                               -20
#define ALGORITHM_RUN_ERROR                             -21

// 参数异常
#define INVALID_PARAMS_NUM                              -30

// 全局变量异常
#define GLOBAL_VAR_NOT_EXIST                            -40
#define GLOBAL_VAR_DUP_NAME                             -41
#define GLOBAL_VAR_TYPE_NOT_SUPPORTED                   -42
#define GLOBAL_VAR_TYPE_MISMATCH                        -43
#define GLOBAL_VAR_BIND_CONFLICT                        -44
#define GLOBAL_VAR_IO_ERROR                             -45
#define GLOBAL_VAR_ASSIGN_PARSE_ERROR                   -46

