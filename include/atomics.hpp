//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace hc
{
    namespace atomics
    {
        /** @{ */
        /**
         * Atomically read the value stored in dest , replace it with the value
         * given in val and return the old value to the caller. This function
         * provides overloads for int, unsigned int, int64_t, uint64_t, float
         * and double parameters.
         *
         * @param[out] dest A pointer to the location which needs to be
         *                  atomically modified. The location may reside within
         *                  an array, an array_view, global or tile_static
         *                  memory.
         * @param[in] val The new value to be stored in the location pointed to
         *                be dest.
         * @return These functions return the old value which was previously
         *         stored at dest, and that was atomically replaced. These
         *         functions always succeed.
         */
        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>
        inline
        T atomic_exchange(T* dest, T val) [[cpu]][[hc]]
        {
            return __atomic_exchange_n(dest, val, __ATOMIC_RELAXED);
        }
        inline
        float atomic_exchange(float* dest, float val) //[[cpu]][[hc]]
        {
            static_assert(sizeof(float) == sizeof(unsigned int), "");

            unsigned int ui{};
            __builtin_memcpy(&ui, &val, sizeof(val));

            unsigned int tmp{
                atomic_exchange(reinterpret_cast<unsigned int*>(dest), ui)};

            float r{};
            __builtin_memcpy(&r, &tmp, sizeof(tmp));

            return r;
        }
        inline
        double atomic_exchange(double* dest, double val) //[[cpu]][[hc]]
        {
            static_assert(sizeof(double) == sizeof(std::uint64_t), "");

            std::uint64_t ui{};
            __builtin_memcpy(&ui, &val, sizeof(val));

            std::uint64_t tmp{
                atomic_exchange(reinterpret_cast<std::uint64_t*>(dest), ui)};

            double r{};
            __builtin_memcpy(&r, &tmp, sizeof(tmp));

            return r;
        }
        /** @} */

        /** @{ */
        /**
         * These functions attempt to perform these three steps atomically:
         * 1. Read the value stored in the location pointed to by dest
         * 2. Compare the value read in the previous step with the value
         *    contained in the location pointed by expected_val
         * 3. Carry the following operations depending on the result of the
         *    comparison of the previous step:
         *    a. If the values are identical, then the function tries to
         *       atomically change the value pointed by dest to the value in
         *       val. The function indicates by its return value whether this
         *       transformation has been successful or not.
         *    b. If the values are not identical, then the function stores the
         *       value read in step (1) into the location pointed to by
         *       expected_val, and returns false.
         *
         * @param[out] dest A pointer to the location which needs to be
         *                  atomically modified. The location may reside within
         *                  an array, an array_view, global or tile_static
         *                  memory.
         * @param[out] expected_val A pointer to a local variable or function
         *                          parameter. Upon calling the function, the
         *                          location pointed by expected_val contains
         *                          the value the caller expects dest to
         *                          contain. Upon return from the function,
         *                          expected_val will contain the most recent
         *                          value read from dest.
         * @param[in] val The new value to be stored in the location pointed to
         *                be dest.
         * @return The return value indicates whether the function has been
         *         successful in atomically reading, comparing and modifying the
         *         contents of the memory location.
         */
        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>
        inline
        bool atomic_compare_exchange(
            T* dest, T* expected_val, T val) [[cpu]][[hc]]
        {
            return __atomic_compare_exchange_n(
                dest,
                expected_val,
                val,
                false,
                __ATOMIC_RELAXED,
                __ATOMIC_RELAXED);
        }
        /** @} */

        /** @{ */
        /**
         * Atomically read the value stored in dest, apply the binary numerical
         * operation specific to the function with the read value and val
         * serving as input operands, and store the result back to the location
         * pointed by dest.
         *
         * In terms of sequential semantics, the operation performed by any of
         * the above function is described by the following piece of
         * pseudo-code:
         *
         * *dest = *dest @f$\otimes@f$ val;
         *
         * Where the operation denoted by @f$\otimes@f$ is one of: addition
         * (atomic_fetch_add), subtraction (atomic_fetch_sub), find maximum
         * (atomic_fetch_max), find minimum (atomic_fetch_min), bit-wise AND
         * (atomic_fetch_and), bit-wise OR (atomic_fetch_or), bit-wise XOR
         * (atomic_fetch_xor).
         *
         * @param[out] dest A pointer to the location which needs to be
         *                  atomically modified. The location may reside within
         *                  an array, an array_view, global or tile_static
         *                  memory.
         * @param[in] val The second operand which participates in the
         *                calculation of the binary operation whose result is
         *                stored into the location pointed to be dest.
         * @return These functions return the old value which was previously
         *         stored at dest, and that was atomically replaced. These
         *         functions always succeed.
         */

        /** @} */
        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>
        inline
        T atomic_fetch_add(T* dest, T val) [[cpu]][[hc]]
        {
            return __atomic_fetch_add(dest, val, __ATOMIC_RELAXED);
        }

        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>        inline
        T atomic_fetch_sub(T* dest, T val) [[cpu]][[hc]]
        {
            return __atomic_fetch_sub(dest, val, __ATOMIC_RELAXED);
        }

        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>        inline
        T atomic_fetch_max(T* dest, T val) [[cpu]][[hc]]
        {
            return __sync_fetch_and_max(dest, val);
        }

        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>        inline
        T atomic_fetch_min(T* dest, T val) [[cpu]][[hc]]
        {
            return __sync_fetch_and_min(dest, val);
        }

        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>        inline
        T atomic_fetch_and(T* dest, T val) [[cpu]][[hc]]
        {
            return __atomic_fetch_and(dest, val, __ATOMIC_RELAXED);
        }

        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>        inline
        T atomic_fetch_or(T* dest, T val) [[cpu]][[hc]]
        {
            return __atomic_fetch_or(dest, val, __ATOMIC_RELAXED);
        }

        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>        inline
        T atomic_fetch_xor(T* dest, T val) [[cpu]][[hc]]
        {
            return __atomic_fetch_xor(dest, val, __ATOMIC_RELAXED);
        }

        /** @{ */
        /**
         * Atomically increment or decrement the value stored at the location
         * point to by dest.
         *
         * @param[out] dest A pointer to the location which needs to be
         *                  atomically modified. The location may reside within
         *                  an array, an array_view, global or tile_static
         *                  memory.
         * @return These functions return the old value which was previously
         *         stored at dest, and that was atomically replaced. These
         *         functions always succeed.
         */

        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>
        inline
        T atomic_fetch_inc(T* dest) [[cpu]][[hc]]
        {
            return __atomic_fetch_add(dest, T{1}, __ATOMIC_RELAXED);
        }

        template<
            typename T,
            typename std::enable_if<
                std::is_integral<T>{} &&
                sizeof(T) >= sizeof(std::int32_t)>::type* = nullptr>
        inline
        T atomic_fetch_dec(T* dest) [[cpu]][[hc]]
        {
            return __atomic_fetch_sub(dest, T{1}, __ATOMIC_RELAXED);
        }
        /** @} */
    } // Namespace atomics.
} // Namespace hc.