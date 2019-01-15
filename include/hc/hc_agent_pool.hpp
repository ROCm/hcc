//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "hc_runtime.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace std
{
    template<>
    struct hash<hsa_agent_t> {
        std::size_t operator()(hsa_agent_t x) const noexcept
        {
            return std::hash<decltype(x.handle)>{}(x.handle);
        }
    };
}

inline
bool operator==(hsa_agent_t x, hsa_agent_t y) noexcept
{
    return x.handle == y.handle;
}

inline
bool operator==(hsa_region_t x, hsa_region_t y) noexcept
{
    return x.handle == y.handle;
}

namespace hc
{
    namespace detail
    {
        class Agent_pool {
            // IMPLEMENTATION - TYPES
            class HSA_agent;

            // IMPLEMENTATION - STATICS
            static
            const std::vector<hsa_agent_t>& agents_();
            static
            hsa_agent_t cpu_agent_();
            static
            hsa_agent_t default_agent_();
            static
            hsa_region_t system_cg_();
        public:
            // STATICS
            static
            std::unordered_map<hsa_agent_t, HSA_agent>& pool();
            static
            hsa_agent_t cpu_agent();
            static
            hsa_agent_t& default_agent();
        };

        class Agent_pool::HSA_agent {
            friend class Agent_pool;

            // IMPLEMENTATION - DATA
            hsa_agent_t agent_;

            // IMPLEMENTATION - STATICS
            static
            std::vector<hsa_region_t> global_regions_(hsa_agent_t x)
            {
                using C = std::vector<hsa_region_t>;

                C r;
                throwing_hsa_result_check(
                    hsa_agent_iterate_regions(x, [](hsa_region_t rg, void* pr) {
                        hsa_region_segment_t s{};
                        throwing_hsa_result_check(
                            hsa_region_get_info(
                                rg, HSA_REGION_INFO_SEGMENT, &s),
                            __FILE__, __func__, __LINE__);

                        if (s == HSA_REGION_SEGMENT_GLOBAL) {
                            static_cast<C*>(pr)->push_back(rg);
                        }

                        return HSA_STATUS_SUCCESS;
                    }, &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            std::uint32_t cu_count_(hsa_agent_t x)
            {
                std::uint32_t r{};
                throwing_hsa_result_check(
                    hsa_agent_get_info(
                        x,
                        static_cast<hsa_agent_info_t>(
                            HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT),
                        &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            hsa_region_t fine_grained_(hsa_agent_t x)
            {
                for (auto&& region : global_regions_(x)) {
                    std::uint32_t f{};
                    throwing_hsa_result_check(
                        hsa_region_get_info(
                            region, HSA_REGION_INFO_GLOBAL_FLAGS, &f),
                        __FILE__, __func__, __LINE__);

                    if (f & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) return region;
                }

                return {};
            }

            static
            hsa_region_t group_(hsa_agent_t x)
            {
                hsa_region_t g{};
                throwing_hsa_result_check(
                    hsa_agent_iterate_regions(x, [](hsa_region_t r, void* pg) {
                        hsa_region_segment_t s{};
                        throwing_hsa_result_check(
                            hsa_region_get_info(r, HSA_REGION_INFO_SEGMENT, &s),
                            __FILE__, __func__, __LINE__);

                        if (s == HSA_REGION_SEGMENT_GROUP) {
                            *static_cast<hsa_region_t*>(pg) = r;
                        }

                        return HSA_STATUS_SUCCESS;
                    }, &g),
                    __FILE__, __func__, __LINE__);

                return g;
            }

            static
            bool is_cpu_accessible_(hsa_region_t x)
            {
                if (x.handle == 0) return false;

                bool r{false};
                throwing_hsa_result_check(
                    hsa_region_get_info(
                        x,
                        static_cast<hsa_region_info_t>(
                            HSA_AMD_REGION_INFO_HOST_ACCESSIBLE),
                        &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            std::uint32_t max_queue_cnt_(hsa_agent_t x)
            {   // We assume that 8 queues per SE, out of which 3 / 4 are
                // dedicated to compute. TODO: assess if we need to subtract the
                // queues implicitly created by ROCr.
                static constexpr double compute_dedicated{0.75};
                static constexpr std::uint32_t queues_per_se{8u};

                std::uint32_t se_cnt{};
                throwing_hsa_result_check(
                    hsa_agent_get_info(
                        x,
                        static_cast<hsa_agent_info_t>(
                            HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES),
                        &se_cnt),
                    __FILE__, __func__, __LINE__);

                return se_cnt * queues_per_se * compute_dedicated;
            }

            static
            std::uint32_t max_queue_sz_(hsa_agent_t x)
            {
                std::uint32_t r{};
                throwing_hsa_result_check(
                    hsa_agent_get_info(x, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            std::uint32_t min_queue_sz_(hsa_agent_t x)
            {
                std::uint32_t r{};
                throwing_hsa_result_check(
                    hsa_agent_get_info(x, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            std::wstring name_(hsa_agent_t x)
            {
                static constexpr std::size_t max_name_length{64};

                char tmp[max_name_length]{};
                throwing_hsa_result_check(
                    hsa_agent_get_info(x, HSA_AGENT_INFO_NAME, tmp),
                    __FILE__, __func__, __LINE__);

                return std::wstring{tmp, tmp + max_name_length};
            }

            static
            enums::accelerator_profile profile_(hsa_agent_t x)
            {   // N.B.: AMD is not going to expose more than one ISA per agent
                //       at this point in time.
                bool p[2]{};
                throwing_hsa_result_check(
                    hsa_agent_iterate_isas(x, [](hsa_isa_t i, void* pp) {
                        throwing_hsa_result_check(
                            hsa_isa_get_info_alt(i, HSA_ISA_INFO_PROFILES, pp),
                            __FILE__, __func__, __LINE__);

                        return HSA_STATUS_SUCCESS;
                    }, p),
                    __FILE__, __func__, __LINE__);

                if (p[HSA_PROFILE_BASE]) return enums::accelerator_profile_base;
                if (p[HSA_PROFILE_FULL]) return enums::accelerator_profile_full;
                return enums::accelerator_profile_none;
            }

            static
            std::size_t size_(hsa_region_t x)
            {
                if (x.handle == 0) return 0u;

                std::size_t r{};
                throwing_hsa_result_check(
                    hsa_region_get_info(x, HSA_REGION_INFO_SIZE, &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            hsa_region_t agent_allocated_cg_(hsa_agent_t x)
            {
                std::vector<hsa_region_t> r{global_regions_(x)};

                for (auto&& agent : agents_()) {
                    if (agent == x) continue;

                    auto tmp = global_regions_(agent);
                    r.erase(
                        std::remove_if(r.begin(), r.end(), [&](hsa_region_t a) {
                            return std::find(
                                tmp.cbegin(), tmp.cend(), a) != tmp.cend();
                        }),
                        r.end());
                }

                if (r.empty()) return {};

                return r.front();
            }

            static
            hsa_device_type_t type_(hsa_agent_t x)
            {
                hsa_device_type_t r{};
                throwing_hsa_result_check(
                    hsa_agent_get_info(x, HSA_AGENT_INFO_DEVICE, &r),
                    __FILE__, __func__, __LINE__);

                return r;
            }

            static
            std::uint32_t version_(hsa_agent_t x)
            {
                std::uint16_t hi{};
                throwing_hsa_result_check(
                    hsa_agent_get_info(x, HSA_AGENT_INFO_VERSION_MAJOR, &hi),
                    __FILE__, __func__, __LINE__);

                std::uint16_t lo{};
                throwing_hsa_result_check(
                    hsa_agent_get_info(x, HSA_AGENT_INFO_VERSION_MINOR, &lo),
                    __FILE__, __func__, __LINE__);

                return (hi << 16u) | lo;
            }

            // IMPLEMENTATION - CREATORS
            explicit
            HSA_agent(hsa_agent_t x)
                :
                agent_{x},
                agent_allocated_coarse_grained_region{agent_allocated_cg_(x)},
                compute_unit_count{cu_count_(x)},
                dedicated_memory{size_(agent_allocated_coarse_grained_region)},
                default_cpu_access{(type_(x) == HSA_DEVICE_TYPE_CPU) ?
                    enums::access_type_read_write : enums::access_type_auto},
                fine_grained_region{fine_grained_(x)},
                has_cpu_accessible_agent_allocated_coarse_grained{
                    is_cpu_accessible_(agent_allocated_coarse_grained_region)},
                has_cpu_shared_memory{size_(fine_grained_region) > 0},
                is_cpu{type_(x) == HSA_DEVICE_TYPE_CPU},
                is_gpu{type_(x) == HSA_DEVICE_TYPE_GPU},
                max_queue_count{max_queue_cnt_(x)},
                max_queue_size{max_queue_sz_(x)},
                max_tile_static_size{size_(group_(x))},
                min_queue_size{min_queue_sz_(x)},
                name{name_(x)},
                profile{is_gpu ? profile_(x) : enums::accelerator_profile_none},
                system_coarse_grained_region{system_cg_()},
                version{version_(x)}
            {}
        public:
            // DATA
            hsa_region_t agent_allocated_coarse_grained_region{};
            std::uint32_t compute_unit_count{};
            std::size_t dedicated_memory{};
            enums::access_type default_cpu_access{};
            hsa_region_t fine_grained_region{};
            bool has_cpu_accessible_agent_allocated_coarse_grained{};
            bool has_cpu_shared_memory{};
            bool is_cpu{};
            bool is_gpu{};
            std::uint32_t max_queue_count{};
            std::uint32_t max_queue_size{};
            std::size_t max_tile_static_size{};
            std::uint32_t min_queue_size{};
            std::wstring name{};
            enums::accelerator_profile profile{};
            hsa_region_t system_coarse_grained_region{};
            std::uint32_t version{};

            // CREATORS
            HSA_agent() = default;
            HSA_agent(const HSA_agent&) = default;
            HSA_agent(HSA_agent&&) = default;
            ~HSA_agent() = default;

            // MANIPULATORS
            HSA_agent& operator=(const HSA_agent&) = default;
            HSA_agent& operator=(HSA_agent&&) = default;
        };

        inline
        const std::vector<hsa_agent_t>& Agent_pool::agents_()
        {
            static std::vector<hsa_agent_t> r;
            static std::once_flag f;

            std::call_once(f, []() {
                throwing_hsa_result_check(
                    hsa_iterate_agents([](hsa_agent_t agent, void*) {
                        r.push_back(agent);

                        return HSA_STATUS_SUCCESS;
                    }, nullptr),
                    __FILE__, __func__, __LINE__);
            });

            return r;
        }

        inline
        hsa_agent_t Agent_pool::cpu_agent_()
        {   // TODO: for e.g. multi-socket there can be multiple CPU agents.
            for (auto&& x : agents_()) {
                if (HSA_agent::type_(x) == HSA_DEVICE_TYPE_CPU) return x;
            }

            return {};
        }

        inline
        hsa_agent_t Agent_pool::default_agent_()
        {
            std::vector<HSA_agent> tmp;
            for (auto&& x : pool()) tmp.push_back(x.second);

            tmp.erase(
                std::remove_if(tmp.begin(), tmp.end(), [](const HSA_agent& x) {
                    return x.is_cpu;
                }),
                tmp.end());

            if (tmp.empty()) return cpu_agent_();

            return std::max_element(
                tmp.cbegin(),
                tmp.cend(),
                [](const HSA_agent& x, const HSA_agent& y) {
                return x.dedicated_memory < y.dedicated_memory;
            })->agent_;
        }

        inline
        hsa_region_t Agent_pool::system_cg_()
        {
            static hsa_region_t sys_cg{};
            static std::once_flag f;

            std::call_once(f, []() {
                for (auto&& region : HSA_agent::global_regions_(cpu_agent_())) {
                    std::uint32_t f{};
                    throwing_hsa_result_check(
                        hsa_region_get_info(
                            region, HSA_REGION_INFO_GLOBAL_FLAGS, &f),
                        __FILE__, __func__, __LINE__);

                    if (f & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
                        sys_cg = region;

                        return;
                    }
                }
            });

            return sys_cg;
        }

        inline
        std::unordered_map<hsa_agent_t, Agent_pool::HSA_agent>& Agent_pool::
            pool()
        {
            static std::unordered_map<hsa_agent_t, HSA_agent> r;
            static std::once_flag f;

            std::call_once(f, []() {
                for (auto&& x : agents_()) r.emplace(x, HSA_agent{x});
            });

            return r;
        }

        inline
        hsa_agent_t Agent_pool::cpu_agent()
        {
            static const hsa_agent_t r{cpu_agent_()};

            return r;
        }

        inline
        hsa_agent_t& Agent_pool::default_agent()
        {
            static hsa_agent_t r{default_agent_()};

            return r;
        }
    } // Namespace hc::detail.
} // Namespace hc.