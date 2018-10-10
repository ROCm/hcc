//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "hc_aligned_alloc.hpp"
#include "hc_defines.hpp"

#include <hsa/hsa.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <future>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace hc
{
    namespace detail
    {
        namespace enums
        {
            /// access_type is used for accelerator that supports unified memory
            /// Such accelerator can use access_type to control whether can
            /// access data on it or not
            enum access_type {
                access_type_none = 0,
                access_type_read = (1 << 0),
                access_type_write = (1 << 1),
                access_type_read_write = access_type_read | access_type_write,
                access_type_auto = (1 << 31)
            };

            enum queuing_mode {
                queuing_mode_immediate,
                queuing_mode_automatic
            };

            enum execute_order {
                execute_in_order,
                execute_any_order
            };

            // Flags to specify visibility of previous commands after a marker
            // is executed.
            enum memory_scope {
                no_scope=0,           // No release operation applied
                accelerator_scope=1,  // Release to current accelerator
                system_scope=2,       // Release to system (CPU + all
                                      // accelerators)
            };

            static
            inline
            memory_scope greater_scope(memory_scope scope1, memory_scope scope2)
            {
                if ((scope1==system_scope) || (scope2 == system_scope)) {
                    return system_scope;
                }
                if ((scope1==accelerator_scope) ||
                    (scope2 == accelerator_scope)) {
                    return accelerator_scope;
                }
                return no_scope;
            }

            enum hcCommandKind {
                hcCommandInvalid= -1,

                hcMemcpyHostToHost = 0,
                hcMemcpyHostToDevice = 1,
                hcMemcpyDeviceToHost = 2,
                hcMemcpyDeviceToDevice = 3,
                hcCommandKernel = 4,
                hcCommandMarker = 5,
            };

            // Commands sent to copy queues:
            static
            inline
            bool isCopyCommand(hcCommandKind k)
            {
                switch (k) {
                    case hcMemcpyHostToHost:
                    case hcMemcpyHostToDevice:
                    case hcMemcpyDeviceToHost:
                    case hcMemcpyDeviceToDevice:
                        return true;
                    default:
                        return false;
                }
            }

            // Commands sent to compute queue:
            static
            inline
            bool isComputeQueueCommand(hcCommandKind k)
            {
                return (k == hcCommandKernel) || (k == hcCommandMarker);
            }

            enum hcWaitMode {
                hcWaitModeBlocked = 0,
                hcWaitModeActive = 1
            };

            enum accelerator_profile {
                accelerator_profile_none = 0,
                accelerator_profile_base = 1,
                accelerator_profile_full = 2
            };
        } // namespace hc::detail::enums

        template<std::size_t m, std::size_t n>
        inline
        void throwing_hsa_result_check(
            hsa_status_t s,
            const char (&file)[m],
            const char (&fn)[n],
            int line)
        {
            if (s == HSA_STATUS_SUCCESS || s == HSA_STATUS_INFO_BREAK) return;

            const char* p{};
            auto r = hsa_status_string(s, &p);

            throw std::system_error{
                (r == HSA_STATUS_SUCCESS) ? s : r,
                std::system_category(),
                "In " + (file +
                    (", in function " + (fn +
                    (((", on line " + std::to_string(line)) +
                    ", HSA RT failed: ") + p))))
            };
        }

        inline
        __attribute__((constructor))
        void construct_hc_runtime()
        {
            throwing_hsa_result_check(hsa_init(), __FILE__, __func__, __LINE__);
        }

        inline
        __attribute__((destructor))
        void destruct_hc_runtime()
        {
            throwing_hsa_result_check(
                hsa_shut_down(), __FILE__, __func__, __LINE__);
        }
    } // Namespace hc::detail.
} // Namespace hc.