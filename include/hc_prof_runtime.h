#pragma once

namespace hc {
// HCC Op IDs enumeration
enum HSAOpId {
    HSA_OP_ID_DISPATCH = 0,
    HSA_OP_ID_COPY = 1,
    HSA_OP_ID_BARRIER = 2,
    HSA_OP_ID_NUMBER = 3
};
}; // namespace hc

namespace Kalmar {
namespace CLAMP {
// Activity profiling primitives
extern void InitActivityCallback(void* id_callback, void* op_callback, void* arg);
extern bool EnableActivityCallback(uint32_t op, bool enable);
extern const char* GetCmdName(uint32_t id);
} // namespace CLAMP
} // namespace Kalmar

/** \endcond */
