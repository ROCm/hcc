//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

namespace hc
{
    namespace attr_impl
    {
        template<typename, typename...> class Callable_with_AMDGPU_attributes;

        struct Flat_wg_tag {};
        struct Max_wg_dim_tag {};
        struct Waves_per_EU_tag {};
    } // Namespace attr_impl.

    namespace detail
    {
        template<typename, typename> struct Kernel_emitter;
    }

    template<unsigned int min_size = 0, unsigned int max_size = 0>
    class Flat_workgroup_size : public attr_impl::Flat_wg_tag {
        static_assert(
            min_size <= max_size,
            "Minimum workgroup size must not be greater than maximum size.");

        static constexpr Flat_workgroup_size* flat_workgroup_size_{};

        template<typename, typename...>
        friend class attr_impl::Callable_with_AMDGPU_attributes;
    public:
        static
        constexpr
        unsigned int minimum() noexcept [[cpu, hc]] { return min_size; }
        static
        constexpr
        unsigned int maximum() noexcept [[cpu, hc]] { return max_size; }
    };

    template<
        unsigned int max_dim_z = 1,
        unsigned int max_dim_y = 1,
        unsigned int max_dim_x = 1>
    class Max_workgroup_dim : public attr_impl::Max_wg_dim_tag {
        static_assert(
            max_dim_z * max_dim_y * max_dim_x > 0 &&
            max_dim_z * max_dim_y * max_dim_x <= 1024u,
            "Flattened required workgroup size must be in (0, 1024].");

        static constexpr Max_workgroup_dim* max_workgroup_dim_{};

        template<typename, typename...>
        friend class attr_impl::Callable_with_AMDGPU_attributes;
    public:
        static
        constexpr
        unsigned int maximum_x() noexcept [[cpu, hc]] { return max_dim_x; }
        static
        constexpr
        unsigned int maximum_y() noexcept [[cpu, hc]] { return max_dim_y; }
        static
        constexpr
        unsigned int maximum_z() noexcept [[cpu, hc]] { return max_dim_z; }
    };

    template<unsigned int min_wave_cnt = 0, unsigned int max_wave_cnt = 0>
    class Waves_per_eu : public attr_impl::Waves_per_EU_tag {
        static_assert(
            max_wave_cnt == 0 || min_wave_cnt <= max_wave_cnt,
            "Minimum number of waves per EU must not be greater than maximum, "
                "if the latter is specified.");

        static constexpr Waves_per_eu* waves_per_eu_{};

        template<typename, typename...>
        friend class attr_impl::Callable_with_AMDGPU_attributes;
    public:
        static
        constexpr
        unsigned int minimum() noexcept [[cpu, hc]] { return min_wave_cnt; }
        static
        constexpr
        unsigned int maximum() noexcept [[cpu, hc]] { return max_wave_cnt; }
    };

    namespace attr_impl
    {
        template<typename Callable, typename... Attrs>
        class Callable_with_AMDGPU_attributes : private Callable {
            struct Triple_ {
                std::size_t m0;
                std::size_t m1;
                std::size_t m2;
            };

            using AttrTuple_ = std::tuple<Attrs..., void>;

            template<std::size_t>
            static
            constexpr
            Triple_ attr_idx_(Triple_ tmp) noexcept [[cpu, hc]]
            {
                return tmp;
            }
            template<std::size_t n, typename T, typename... As>
            static
            constexpr
            Triple_ attr_idx_(Triple_ tmp) noexcept [[cpu, hc]]
            {
                return std::is_base_of<Flat_wg_tag, T>{} ?
                    attr_idx_<n + 1, As...>({n, tmp.m1, tmp.m2}) :
                    (std::is_base_of<Max_wg_dim_tag, T>{} ?
                        attr_idx_<n + 1, As...>({tmp.m0, n, tmp.m2}) :
                        (std::is_base_of<Waves_per_EU_tag, T>{} ?
                            attr_idx_<n + 1, As...>({tmp.m0, tmp.m1, n}) :
                            attr_idx_<n + 1, As...>(tmp)));
            }

            static constexpr Triple_ idxs_{attr_idx_<0u, Attrs...>({
                sizeof...(Attrs), sizeof...(Attrs), sizeof...(Attrs)})};

            using Flat_wg_size_ = typename std::conditional<
                idxs_.m0 != sizeof...(Attrs),
                typename std::tuple_element<idxs_.m0, AttrTuple_>::type,
                Flat_workgroup_size<>>::type;
            using Max_wg_dim_ = typename std::conditional<
                idxs_.m1 != sizeof...(Attrs),
                typename std::tuple_element<idxs_.m1, AttrTuple_>::type,
                Max_workgroup_dim<>>::type;
            using Waves_per_EU_ = typename std::conditional<
                idxs_.m2 != sizeof...(Attrs),
                typename std::tuple_element<idxs_.m2, AttrTuple_>::type,
                Waves_per_eu<>>::type;

            template<typename, typename>
            friend struct detail::Kernel_emitter;
        public:
            // CREATORS
            Callable_with_AMDGPU_attributes() [[cpu, hc]] = default;
            explicit
            Callable_with_AMDGPU_attributes(Callable callable)
                : Callable{std::move(callable)} {}
            Callable_with_AMDGPU_attributes(
                const Callable_with_AMDGPU_attributes&) [[cpu, hc]] = default;
            Callable_with_AMDGPU_attributes(
                Callable_with_AMDGPU_attributes&&) [[cpu, hc]] = default;
            ~Callable_with_AMDGPU_attributes() [[cpu, hc]] = default;

            // ACCESSORS
            using Callable::operator();
        };
    } // Namespace hc::attr_impl.

    template<typename... Attrs, typename Callable>
    inline
    attr_impl::Callable_with_AMDGPU_attributes<
        Callable, Attrs...> make_callable_with_AMDGPU_attributes(Callable f)
    {
        return attr_impl::Callable_with_AMDGPU_attributes<Callable, Attrs...>{
            std::move(f)};
    }
} // Namespace hc.