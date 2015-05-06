#pragma once

typedef void* (*LaunchKernelAsyncImpl_t)(void *, size_t, size_t*, size_t*);
typedef void* (*MatchKernelNamesImpl_t)(char *);
typedef void* (*PushArgImpl_t)(void *, int, size_t, const void *);
typedef void* (*PushArgPtrImpl_t)(void *, int, size_t, const void *);
typedef void* (*GetContextImpl_t)();
