//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../types/raii_handle.hpp"

#include <hc.hpp>

#include <hsa/hsa.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <array>

static
inline
bool operator==(hsa_agent_t x, hsa_agent_t y)
{
    return x.handle == y.handle;
}

static
inline
bool operator==(hsa_isa_t x, hsa_isa_t y)
{
    return x.handle == y.handle;
}

static
inline
bool operator==(hsa_executable_symbol_t x, hsa_executable_symbol_t y)
{
    return x.handle == y.handle;
}

namespace std
{
    template<>
    struct hash<hsa_agent_t> {
        using argument_type = hsa_agent_t;
        using result_type = std::size_t;

        result_type operator()(const argument_type& x) const
        {
            return std::hash<decltype(x.handle)>{}(x.handle);
        }
    };

    template<>
    struct hash<hsa_isa_t> {
        using argument_type = hsa_isa_t;
        using result_type = std::size_t;

        result_type operator()(const argument_type& x) const
        {
            return std::hash<decltype(x.handle)>{}(x.handle);
        }
    };

    template<>
    struct hash<hsa_executable_symbol_t> {
        using argument_type = hsa_executable_symbol_t;
        using result_type = std::size_t;

        result_type operator()(const argument_type& x) const
        {
            return std::hash<decltype(x.handle)>{}(x.handle);
        }
    };
}

namespace hc2
{
    inline
    std::string to_string(const hsa_isa_t& x)
    {
        std::size_t sz = 0u;
        hsa_isa_get_info(x, HSA_ISA_INFO_NAME_LENGTH, 0u, &sz);

        std::string r(sz, '\0');
        hsa_isa_get_info(x, HSA_ISA_INFO_NAME, 0u, &r.front());

        return r;
    }

    inline
    std::string to_string(hsa_status_t err)
    {
        const char* s = nullptr;
        if (hsa_status_string(err, &s) != HSA_STATUS_SUCCESS) {
            return "Unknown.";
        }

        return s;
    }

    inline
    void throwing_hsa_result_check(
        hsa_status_t res,
        const std::string& file,
        const std::string& fn,
        int line)
    {
        if (res != HSA_STATUS_SUCCESS) {
            throw std::runtime_error{
                "Failed in file " + file + ", in function \"" + fn +
                "\", on line " + std::to_string(line) + ", with error: " +
                to_string(res)};
        }
    }

    inline
    hsa_agent_t hsa_agent(const hc::accelerator_view& av)
    {
        assert(
            const_cast<hc::accelerator_view&>(av).get_hsa_agent() != nullptr);

        return *static_cast<hsa_agent_t*>(
            const_cast<hc::accelerator_view&>(av).get_hsa_agent());
    }

    inline
    hsa_agent_t hsa_agent(const hc::accelerator& acc)
    {
        return hsa_agent(acc.get_default_view());
    }

    inline
    hsa_isa_t hsa_agent_isa(const hc::accelerator_view& av)
    {
        hsa_isa_t r;
        throwing_hsa_result_check(
            hsa_agent_get_info(hsa_agent(av), HSA_AGENT_INFO_ISA, &r),
            __FILE__,
            __func__,
            __LINE__);

        return r;
    }

    inline
    hsa_isa_t hsa_agent_isa(const hc::accelerator& acc)
    {
        return hsa_agent_isa(acc.get_default_view());
    }

    inline
    std::string hsa_agent_name(hsa_agent_t x)
    {
        static constexpr std::size_t max_name_length = 63;
        static std::array<char, max_name_length> n = {{}};

        throwing_hsa_result_check(
            hsa_agent_get_info(x, HSA_AGENT_INFO_NAME, n.data()),
            __FILE__,
            __func__,
            __LINE__);

        return std::string{n.cbegin(), n.cend()};
    }

    inline
    std::string hsa_agent_name(const hc::accelerator_view& av)
    {
        return hsa_agent_name(hsa_agent(av));
    }

    inline
    std::string hsa_agent_name(const hc::accelerator& acc)
    {
        return hsa_agent_name(acc.get_default_view());
    }

    inline
    decltype(HSA_PROFILE_FULL) hsa_agent_profile(const hc::accelerator_view& av)
    {
        decltype(HSA_PROFILE_FULL) r;
        throwing_hsa_result_check(
            hsa_agent_get_info(hsa_agent(av), HSA_AGENT_INFO_PROFILE, &r),
            __FILE__,
            __func__,
            __LINE__);

        return r;
    }

    inline
    decltype(HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT) hsa_agent_float_rounding_mode(hsa_agent_t a)
    {
        decltype(HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT) r;
        throwing_hsa_result_check(
            hsa_agent_get_info(
            a, HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &r),
        __FILE__,
        __func__,
        __LINE__);

        return r;
    }

    inline
    decltype(HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT) hsa_agent_float_rounding_mode(const hc::accelerator_view& av)
    {
        return hsa_agent_float_rounding_mode(hsa_agent(av));
    }

    namespace
    {
        using Executable_symbol_info_t =
            decltype(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT);
    }

    struct Symbol_kind_tag {
        using value_type = hsa_symbol_kind_t;

        constexpr
        operator Executable_symbol_info_t() const
        {
            return HSA_EXECUTABLE_SYMBOL_INFO_TYPE;
        }
    };

    struct Symbol_name_size_tag {
        using value_type = std::uint32_t;

        constexpr
        operator Executable_symbol_info_t() const
        {
            return HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH;
        }
    };

    struct Private_size_tag {
        using value_type = std::uint32_t;

        constexpr
        operator Executable_symbol_info_t() const
        {
            return HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE;
        }
    };

    struct Lds_size_tag {
        using value_type = std::uint32_t;

        constexpr
        operator Executable_symbol_info_t() const
        {
            return HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE;
        }

    };

    struct Kernel_handle_tag {
        using value_type = std::uint64_t;

        constexpr
        operator Executable_symbol_info_t() const
        {
            return HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT;
        }
    };

    template<typename T>
    typename T::value_type hsa_kernel_info(hsa_executable_symbol_t x, T info)
    {
        typename T::value_type r = {};
        throwing_hsa_result_check(
            hsa_executable_symbol_get_info(x, info, &r),
            __FILE__,
            __func__,
            __LINE__);

        return r;
    }

    inline
    bool is_kernel(hsa_executable_symbol_t x)
    {
        return hsa_kernel_info(x, Symbol_kind_tag{}) ==
            hsa_symbol_kind_t::HSA_SYMBOL_KIND_KERNEL;
    }

    inline
    std::string hsa_symbol_name(hsa_executable_symbol_t x)
    {
        const auto sz = hsa_kernel_info(x, Symbol_name_size_tag{});

        std::string r(sz, '\0');
        hsa_executable_symbol_get_info(
            x, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &r.front());

        return r;
    }
}
