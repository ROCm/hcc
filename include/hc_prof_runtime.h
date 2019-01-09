#pragma once

namespace Kalmar {
namespace CLAMP {
// Activity profiling primitives
extern bool SetActivityCallback(uint32_t op, void* callback, void* arg);
extern void SetActivityIdCallback(void* callback);
extern const char* GetCmdName(uint32_t id);
} // namespace CLAMP
} // namespace Kalmar

/** \endcond */
