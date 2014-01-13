// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
/**********************************************************************************
* amptest\device.h
*
**********************************************************************************/

// Attach the dpctest.lib
//#include <amptest\dpctest_lib.h>
#include <amp.h>

#define AMP_TEST_API

namespace Concurrency {
    namespace Test {

        enum class device_flags : int {
            INCLUDE_CPU  = 1 << 1,    // Whether when getting a list of devices to also include the CPU device

            // Compute device type flags
            EMULATED     = 1 << 2,
            NOT_EMULATED = 1 << 3,
            D3D11_REF    = 1 << 4,
            D3D11_WARP   = 1 << 5,
            D3D11_GPU    = 1 << 6,
            D3D11_ATI    = 1 << 7,
            D3D11_NVIDIA = 1 << 8,

            // Characteristic flags
            DOUBLE              = 1 << 11,
            LIMITED_DOUBLE      = 1 << 12,
            LIMITED_DOUBLE_ONLY = 1 << 13,
            NO_DOUBLE           = 1 << 14,

            // Special flags
            AMP_DEFAULT  = 1 << 10,    // Whether the device should be the default device used by the AMP runtime.
            UNKNOWN      = 1 << 31, // Used when parsing from a string

            // Default value
            NOT_SPECIFIED = 0
        };
        inline device_flags operator &(device_flags lhs, device_flags rhs) {
            return static_cast<device_flags>(static_cast<int>(lhs) & static_cast<int>(rhs));
        }
        inline device_flags operator |(device_flags lhs, device_flags rhs) {
            return static_cast<device_flags>(static_cast<int>(lhs) | static_cast<int>(rhs));
        }
        inline void operator |=(device_flags& lhs, device_flags rhs) {
            lhs = lhs | rhs;
        }

        /// A mask of all the valid device_flags values.
        static const device_flags all_valid_device_flags
            = device_flags::INCLUDE_CPU
            | device_flags::EMULATED | device_flags::NOT_EMULATED
            | device_flags::D3D11_REF | device_flags::D3D11_WARP
            | device_flags::D3D11_GPU | device_flags::D3D11_ATI | device_flags::D3D11_NVIDIA
            | device_flags::DOUBLE | device_flags::LIMITED_DOUBLE | device_flags::LIMITED_DOUBLE_ONLY | device_flags::NO_DOUBLE
            | device_flags::AMP_DEFAULT
            ;

        /// Parses the string as a single device_flags type.
        /// If the value isn't recognized, a flag of UNKNOWN will be returned.
        device_flags AMP_TEST_API parse_device_flag(const std::string& str);

        /// Parses the string as a one or more device_flags concatenated together
        /// with a single separator character (default is a pipe ('|') character).
        /// Note, whitespace is not allowed unless it's the separator character.
        /// If a value isn't recognized (or UNKNOWN) an amptest_cascade_failure exception will be thrown.
        device_flags AMP_TEST_API parse_device_flags(const std::string& src_desc, const std::string& src, char separator = '|', device_flags valid_flags = all_valid_device_flags);
        inline device_flags parse_device_flags(const std::string& src, char separator = '|', device_flags valid_flags = all_valid_device_flags) {
            return parse_device_flags("", src, separator, valid_flags);
        }

        /// Converts the device_flags to a string representation.
        std::string AMP_TEST_API device_flags_to_string(device_flags flags);

        /// Tells the AMPTest Device Management Runtime (DMR) which devices the current test is
        /// meant to support.
        /// This method may only be called before any DMR API that retrieves a device or set
        /// of devices, otherwise an exception will be thrown.
        void AMP_TEST_API set_amptest_supported_devices(device_flags required_flags = device_flags::NOT_SPECIFIED);


        /// Gets a prioritized list of the available devices for the current amptest.
        /// Optionally, you may specify some required device_flags to limit the types
        /// of devices that are returned.
        /// This method takes the following environment variables into account:
        ///    AMPTEST_ALLOWED_DEVICES - Set on cmd line before running tests to limit
        ///       which types of devices are allowed to be used for the run.
        ///    AMPTEST_SUPPORTED_DEVICES - Set by the test in an env.lst to declare when
        ///       a test is designed for only certain devices. This may also be specified
        ///       by using the set_amptest_supported_devices function before this.
        /// Exceptions:
        /// invalid_argument - If device_flags has the AMP_DEFAULT flag set.
        std::vector<accelerator> AMP_TEST_API get_available_devices(device_flags required_flags = device_flags::NOT_SPECIFIED);

        
        /// Retrieves a device from the DMF that has the required_flags.
        /// The DMF will return its first prioritized choice.
        /// If no device is available:
        ///   - if amptest_main.h is used an amptest_skip exception is thrown.
        ///   - if amptest_main.h IS NOT used, then exit() is called with an exit code of runall_skip.
        accelerator AMP_TEST_API require_device(device_flags required_flags = device_flags::NOT_SPECIFIED) {
            accelerator def;
            return def;
        }
        /// Retrieves a device from the DMF that has the required_flags and excludes the specified device.
        /// The DMF will return its first prioritized choice which is not excluded.
        /// This overload is used for when a test needs two distinct devices.
        /// If no device is available:
        ///   - if amptest_main.h is used an amptest_skip exception is thrown.
        ///   - if amptest_main.h IS NOT used, then exit() is called with an exit code of runall_skip.
        accelerator AMP_TEST_API require_device(const accelerator& excluded_device, device_flags required_flags = device_flags::NOT_SPECIFIED) {
            accelerator def;
            return def;
        }

        /// Requires a device and when T is 'double' will also verify the device has double support.
        /// The type of double support requested is determined by the parameter full_double_support:
        ///    true  => device_flags::DOUBLE
        ///    false => device_flags::LIMITED_DOUBLE
        /// The default value of full_double_support if false.
        template <typename T>
        inline accelerator require_device_for(device_flags required_flags = device_flags::NOT_SPECIFIED, bool full_double_support = false) {
            static_assert(0 == (sizeof(T) % sizeof(int)), "only value types whose size is a multiple of the size of an integer are allowed on accelerator");

            //UNREFERENCED_PARAMETER(full_double_support);    // only used for double support
            return require_device(required_flags);
        }

        /// Requires a device and when T is 'double' will also verify the device has double support.
        /// The type of double support requested is determined by the parameter full_double_support:
        ///    true  => device_flags::DOUBLE
        ///    false => device_flags::LIMITED_DOUBLE
        /// The default value of full_double_support if false.
        template <>
        inline accelerator require_device_for<double>(device_flags required_flags, bool full_double_support) {
            if(full_double_support) {
                required_flags |= device_flags::DOUBLE;
            } else {
                required_flags |= device_flags::LIMITED_DOUBLE;
            }

            return require_device(required_flags);
        }

        /// Requires a device and when T is 'double' will also verify the device has double support.
        /// The type of double support requested is determined by the parameter full_double_support:
        ///    true  => device_flags::DOUBLE
        ///    false => device_flags::LIMITED_DOUBLE
        /// The default value of full_double_support if false.
        template <typename T>
        inline accelerator require_device_for(const accelerator& excluded_device, device_flags required_flags = device_flags::NOT_SPECIFIED, bool full_double_support = false) {
            static_assert(0 == (sizeof(T) % sizeof(int)), "only value types whose size is a multiple of the size of an integer are allowed on accelerator");

            //UNREFERENCED_PARAMETER(full_double_support);    // only used for double support
            return require_device(excluded_device, required_flags);
        }

        /// Requires a device and when T is 'double' will also verify the device has double support.
        /// The type of double support requested is determined by the parameter full_double_support:
        ///    true  => device_flags::DOUBLE
        ///    false => device_flags::LIMITED_DOUBLE
        /// The default value of full_double_support if false.
        template <>
        inline accelerator require_device_for<double>(const accelerator& excluded_device, device_flags required_flags, bool full_double_support) {
            if(full_double_support) {
                required_flags |= device_flags::DOUBLE;
            } else {
                required_flags |= device_flags::LIMITED_DOUBLE;
            }

            return require_device(excluded_device, required_flags);
        }
    }
}

