//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <exception>
#include <string>

namespace hc
{
    namespace detail
    {
        #ifndef E_FAIL
            static constexpr auto E_FAIL = 0x80004005;
        #endif

        static constexpr const char __errorMsg_UnsupportedAccelerator[]{
            "hc::parallel_for_each is not supported on the selected accelerator"
            " \"CPU accelerator\"."};

        // TODO: this should use standard error_code / error_category.
        using HRESULT = typename std::remove_const<decltype(E_FAIL)>::type;
        class runtime_exception : public std::exception {
            std::string message_;
            HRESULT code_;
        public:
            // TODO: noexcept is somewhat debateable, given the string.
            runtime_exception(
                const char * message,
                HRESULT hresult) noexcept : message_{message}, code_{hresult}
            {}
            explicit
            runtime_exception(HRESULT hresult) noexcept : code_{hresult} {}
            runtime_exception(const runtime_exception& other) = default;
            runtime_exception(runtime_exception&&) = default;
            virtual
            ~runtime_exception() = default;

            runtime_exception& operator=(const runtime_exception&) = default;
            runtime_exception& operator=(runtime_exception&&) = default;

            virtual
            const char* what() const noexcept
            {
                return message_.c_str();
            }

            HRESULT get_error_code() const noexcept
            {
                return code_;
            }
        };

        struct invalid_compute_domain : public runtime_exception {
            explicit
            invalid_compute_domain(const char* message) noexcept
                : runtime_exception{message, E_FAIL}
            {}
            invalid_compute_domain() noexcept : runtime_exception{E_FAIL} {}
        };

        struct accelerator_view_removed : public runtime_exception {
            explicit
            accelerator_view_removed(
                const char* message, HRESULT view_removed_reason) noexcept
                : runtime_exception{message, view_removed_reason}
            {}
            accelerator_view_removed(HRESULT view_removed_reason) noexcept
                : runtime_exception{view_removed_reason}
            {}

            HRESULT get_view_removed_reason() const noexcept
            {
                return get_error_code();
            }
        };
    } // Namespace hc::detail.
} // Namespace hc.