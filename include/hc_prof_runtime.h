#pragma once

namespace Kalmar {
namespace CLAMP {
// Activity profiling primitives
// HCC activity record id type
typedef uint64_t record_id_t;
// Set/get methods
extern void SetActivityRecord(record_id_t record_id);
extern void GetActivityCoord(int* device_id, uint64_t* queue_id);
extern bool SetActivityCallback(uint32_t op, void* callback, void* arg);
extern const char* GetCmdName(uint32_t id);
} // namespace CLAMP
} // namespace Kalmar

/** \endcond */
