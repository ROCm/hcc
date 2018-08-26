#pragma once

#include<string>

typedef void* (*PushArgImpl_t)(void *, int, size_t, const void *);
typedef void* (*PushArgPtrImpl_t)(void *, int, size_t, const void *);
typedef void* (*GetContextImpl_t)();

// Activity profiling routines
typedef void (*SetActivityRecordImpl_t)(unsigned long);
typedef void (*GetActivityCoordImpl_t)(int*, uint64_t*);
typedef bool (*SetActivityCallbackImpl_t)(unsigned, void*, void*);
typedef const char* (*GetCmdNameImpl_t)(unsigned);
