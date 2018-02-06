#pragma once

#include<string>

typedef void* (*PushArgImpl_t)(void *, int, size_t, const void *);
typedef void* (*PushArgPtrImpl_t)(void *, int, size_t, const void *);
typedef void* (*GetContextImpl_t)();
