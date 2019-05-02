#pragma once

#include<string>

typedef void* (*PushArgImpl_t)(void *, int, size_t, const void *);
typedef void* (*PushArgPtrImpl_t)(void *, int, size_t, const void *);
typedef void* (*GetContextImpl_t)();
typedef void  (*ShutdownImpl_t)();

// Activity profiling routines
typedef void (*InitActivityCallbackImpl_t)(void*, void*, void*);
typedef bool (*EnableActivityCallbackImpl_t)(unsigned, bool);
typedef const char* (*GetCmdNameImpl_t)(unsigned);
