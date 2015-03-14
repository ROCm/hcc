#pragma once

typedef void (*EnumerateDevicesImpl_t)(int*, int*);
typedef void (*QueryDeviceInfoImpl_t)(const wchar_t*, bool*, size_t*, bool*, wchar_t*);
typedef void* (*CreateKernelImpl_t)(const char*, void*, void*);
typedef void (*LaunchKernelImpl_t)(void *, size_t, size_t*, size_t*);
typedef void* (*LaunchKernelAsyncImpl_t)(void *, size_t, size_t*, size_t*);
typedef void* (*MatchKernelNamesImpl_t)(char *);
typedef void* (*PushArgImpl_t)(void *, int, size_t, const void *);
typedef void* (*PushArgPtrImpl_t)(void *, int, size_t, const void *);
typedef void* (*GetAllocatorImpl_t)();

