// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
// DMF.cpp
  //
  // This file contains the implementation of the AMPTest Device Management Framework (DMF)
  //
#include <amptest/context.h>
#include <amptest/device.h>
#include <amptest/logging.h>
#include <amptest/runall.h>
#include <amptest/string_utils.h>
#include <sstream>
#include <math.h>
#include <vector>

namespace {

	template<typename T>
		inline bool has_bits_set(T val, T bits) {
		static_assert(sizeof(T) == sizeof(int), "has_bits_set requires the parameter type T to be the same size as an int.");
		return (static_cast<int>(val) & static_cast<int>(bits)) == static_cast<int>(bits);
	}

	template<typename T>
		inline bool has_any_bits_set(T val, T bits) {
		static_assert(sizeof(T) == sizeof(int), "has_bits_set requires the parameter type T to be the same size as an int.");
		return (static_cast<int>(val) & static_cast<int>(bits)) != 0;
	}

}


namespace Concurrency {
    namespace Test {
		using namespace std;

#pragma region String helper functions
		/// These functions may be moved out to a utility header file when needed.

		inline bool is_not_whitespace(char c) {
			return !isspace (c);
		}

		/// Splits a string and optionally skips empty instances.
		std::vector<std::string> split_string(const std::string& instr, char sep_token, bool skip_empty = false) {
			vector<string> results;

			string::size_type len = instr.length();
			string::size_type startIdx = 0;
			string::size_type nextIdx = instr.find(sep_token, startIdx);
			while(nextIdx != string::npos) {
				if(!skip_empty || (nextIdx-startIdx) > 0) {
					results.push_back(instr.substr(startIdx, nextIdx-startIdx));
				}

				// skip the seperator token and find the next one
				startIdx = nextIdx+1;
				nextIdx = instr.find(sep_token, startIdx);
			}

			// And add the last
			if(startIdx < len || (startIdx == len && !skip_empty)) {
				results.push_back(instr.substr(startIdx));
			}

			return results;
		}

#pragma endregion

#pragma region Device Management Framework (DMF)

		namespace details {

#pragma region enum types

			enum class device_bit_flags : int {
					IS_CPU     = device_flags::INCLUDE_CPU,
					IS_REF     = device_flags::D3D11_REF,
					IS_WARP    = device_flags::D3D11_WARP,
					IS_GPU     = device_flags::D3D11_GPU,
					IHV_ATI    = device_flags::D3D11_ATI,
					IHV_NVIDIA = device_flags::D3D11_NVIDIA,

				//IS_AMP_DEFAULT  = device_flags::AMP_DEFAULT,	// We don't want to instantiate this so we'll check this seperately
					IS_EMULATED     = device_flags::EMULATED,
					SUPPORTS_DOUBLE = device_flags::DOUBLE,
					SUPPORTS_LIMITED_DOUBLE = device_flags::LIMITED_DOUBLE,

					EMPTY = 0
					};
			device_bit_flags operator |(device_bit_flags lhs, device_bit_flags rhs) {
				return static_cast<device_bit_flags>(static_cast<int>(lhs) | static_cast<int>(rhs));
			}
			void operator |=(device_bit_flags& lhs, device_bit_flags rhs) {
				lhs = lhs | rhs;
			}

			static const char* IHV_FAILURE_IHV_TOKENS = "ati,nvidia,warp";
			static const char* IHV_FAILURE_CONTEXT_TOKENS = "win7,win8, x86,amd64,arm, chk,ret, aliasing_forced,aliasing_not_forced";
			enum class known_IHV_failure : int {
				NONE       = 0,

				IHV_ATI    = 1 << 1,
				IHV_NVIDIA = 1 << 2,
				IHV_WARP   = 1 << 3,
				IHV_MASK   = IHV_ATI | IHV_NVIDIA | IHV_WARP,

				OS_WIN7    = 1 << 4,
				OS_WIN8    = 1 << 5,
				OS_MASK    = OS_WIN7 | OS_WIN8,

				ARCH_X86   = 1 << 6,
				ARCH_AMD64 = 1 << 7,
				ARCH_ARM   = 1 << 8,
				ARCH_MASK  = ARCH_X86 | ARCH_AMD64 | ARCH_ARM,

				FLAV_CHK   = 1 << 9,
				FLAV_RET   = 1 << 10,
				FLAV_MASK  = FLAV_CHK | FLAV_RET,

				ALIASING_FORCED     = 1 << 11,
				ALIASING_NOT_FORCED = 1 << 12,
				ALIASING_MASK       = ALIASING_FORCED | ALIASING_NOT_FORCED
			};
			known_IHV_failure operator |(known_IHV_failure lhs, known_IHV_failure rhs) {
				return static_cast<known_IHV_failure>(static_cast<int>(lhs) | static_cast<int>(rhs));
			}
			void operator |=(known_IHV_failure& lhs, known_IHV_failure rhs) {
				lhs = lhs | rhs;
			}

			enum class known_IHV_failure_behavior {
				EXCLUDE_DEVICE = 1,
					IGNORE_FAILURE,
					FAIL_TEST,
					SKIP_TEST,
					};

#pragma endregion

			// Gets the flags set on val that are not set in valid_flags.
			inline device_flags get_invalid_flags(device_flags val, device_flags valid_flags) {
				int invalid_bits = static_cast<int>(val) & ~static_cast<int>(valid_flags);
				return static_cast<device_flags>(invalid_bits);
			}


			bool is_dmf_initialized = false;
			bool dmf_require_device_exits_on_no_device = true;
			device_flags dmf_allowed_device_flags = device_flags::NOT_SPECIFIED;
			vector<wstring> dmf_allowed_gpu_device_paths;
			device_flags dmf_supported_device_flags = device_flags::NOT_SPECIFIED;
			bool dmf_supported_device_flags_set_in_code = false;

			bool dmf_TARGET_DEVICE_throws_failure = false;
			device_flags dmf_TARGET_DEVICE_flags = device_flags::NOT_SPECIFIED;

			vector<known_IHV_failure> dmf_known_IHV_failures;
			known_IHV_failure_behavior dmf_known_IHV_failure_behavior;

			static const char* env_var_name_allowed_devices = "AMPTEST_ALLOWED_DEVICES";
			static const char* env_var_name_allowed_gpu_device_paths = "AMPTEST_ALLOWED_GPU_DEVICE_PATHS";
			static const char* env_var_name_supported_devices = "AMPTEST_SUPPORTED_DEVICES";
			static const char* env_var_name_TARGET_DEVICE = "TARGET_DEVICE";
			static const char* env_var_name_known_IHV_failures = "IHV_FAILURES";
			static const char* env_var_name_known_IHV_failure_behavior = "AMPTEST_IHV_FAILURE_BEHAVIOR";

			// Mask of all the flags which allow for filtering on the type of a compute device.
			// NOTE: This doesn't include CPU since it's not a valid compute device
			static const device_flags compute_device_type_flags_mask
			= device_flags::EMULATED | device_flags::NOT_EMULATED
																																																															 | device_flags::D3D11_REF
																																																															 | device_flags::D3D11_WARP
																																																															 | device_flags::D3D11_GPU | device_flags::D3D11_ATI | device_flags::D3D11_NVIDIA
																																																															 ;
			static const device_flags device_capability_flags_mask
			= device_flags::DOUBLE
																																																															 | device_flags::LIMITED_DOUBLE
																																																															 | device_flags::LIMITED_DOUBLE_ONLY
																																																															 | device_flags::NO_DOUBLE
																																																															 ;

			// We allow all device types, except for IHV specific flags, as tests should not target specific vendors.
			// This also helps to prevent the simple renaming of TARGET_DEVICE to AMPTEST_SUPPORTED_DEVICES.
			static const device_flags valid_supported_device_flags
			= device_flags::EMULATED | device_flags::NOT_EMULATED
																																																															 | device_flags::D3D11_REF
																																																															 | device_flags::D3D11_WARP
																																																															 | device_flags::D3D11_GPU // No IHV-specific flags accepted
																																																															 | device_capability_flags_mask
																																																															 ;

			std::string get_known_IHV_failure_behavior_name() {
				switch(dmf_known_IHV_failure_behavior) {
				case known_IHV_failure_behavior::EXCLUDE_DEVICE: return "EXCLUDE_DEVICE";
				case known_IHV_failure_behavior::IGNORE_FAILURE: return "IGNORE_FAILURE";
				case known_IHV_failure_behavior::FAIL_TEST:      return "FAIL_TEST";
				case known_IHV_failure_behavior::SKIP_TEST:      return "SKIP_TEST";
				default: return "Unknown";
				}
			}

#pragma region DMF Initialization

			void parse_allowed_devices() {
				static const device_flags valid_flags = compute_device_type_flags_mask;

				dmf_allowed_device_flags = device_flags::NOT_SPECIFIED;

				string env_var_val = trim(amptest_context.get_environment_variable(env_var_name_allowed_devices));

				// Split by ' ' in case of RUNALL_CROSSLIST usages
				auto parts = split_string(env_var_val, ' ', true);
				for (auto&& part : parts) {
						// 'RESET' is only valid when it's surrounded by spaces
						if(part == "RESET") {
							// Reset to the default value
							dmf_allowed_device_flags = device_flags::NOT_SPECIFIED;
							return;
						} else if(part.find("RESET") != string::npos) {
							stringstream ss;
							ss << "'RESET' is only allowed in environment variable " << env_var_name_allowed_devices << " when it is all alone. i.e. it must be separated by spaces (from a RUNALL_CROSSLIST) and not by commas.";
							throw amptest_cascade_failure(ss.str());
						} else if(part.find("NOT_SPECIFIED") != string::npos) {
							stringstream ss;
							ss << "The device type 'NOT_SPECIFIED' is not allowed in environment variable " << env_var_name_allowed_devices << ".";
							throw amptest_cascade_failure(ss.str());
						}

						device_flags flags = parse_device_flags(string("environment variable ") + env_var_name_allowed_devices, part, ',', valid_flags);

						dmf_allowed_device_flags |= flags;
					}
			}

			void parse_allowed_gpu_device_paths() {
				dmf_allowed_gpu_device_paths.clear();

				string env_var_val = trim(amptest_context.get_environment_variable(env_var_name_allowed_gpu_device_paths));

				// Split by ' ' in case of RUNALL_CROSSLIST usages
				auto parts1 = split_string(env_var_val, ' ', true);
				for (auto&& part1 : parts1) {
						auto parts2 = split_string(part1, ',', true);
						for_each(parts2.begin(), parts2.end(), [](const string& part2) {
								// Detect whether 'NOT_SPECIFIED' is used, because this would cause the prev flags to be cleared
								if(part2 == "NOT_SPECIFIED") {
									dmf_allowed_gpu_device_paths.clear();
								} else {
									throw amptest_cascade_failure("Not Implemented. parse_allowed_gpu_device_paths needs to convert string to wstring for device paths.");
									//dmf_allowed_gpu_device_paths.push_back(part);
								}
							});
					}
			}

			void parse_supported_devices() {
				dmf_supported_device_flags = device_flags::NOT_SPECIFIED;

				string env_var_val = trim(amptest_context.get_environment_variable(env_var_name_supported_devices));

				// Split by ' ' in case of RUNALL_CROSSLIST usages
				auto parts = split_string(env_var_val, ' ', true);
				for (auto&& part : parts) {
						// Detect whether 'NOT_SPECIFIED' is used, because it doesn't have any meaning.
						if(part.find("NOT_SPECIFIED") != string::npos) {
							stringstream ss;
							ss << "The device type 'NOT_SPECIFIED' is not allowed in environment variable " << env_var_name_supported_devices << ".";
							throw amptest_cascade_failure(ss.str());
						}

						device_flags flags = parse_device_flags(string("environment variable ") + env_var_name_supported_devices, part, ',', valid_supported_device_flags);

						dmf_supported_device_flags |= flags;
					}
			}

			void parse_TARGET_DEVICE() {
				static const device_flags valid_flags = device_flags::D3D11_REF | device_flags::D3D11_WARP
					| device_flags::D3D11_GPU | device_flags::D3D11_ATI | device_flags::D3D11_NVIDIA;

				dmf_TARGET_DEVICE_flags = device_flags::NOT_SPECIFIED;

				string env_var_val = trim(amptest_context.get_environment_variable(env_var_name_TARGET_DEVICE));
				if(env_var_val.length() == 0) return;

				// Handle when they use ALL_DEVICES, as it's the same as NOT_SPECIFIED.
				if(env_var_val == "ALL_DEVICES") {
					Log(LogType::Warning, true) << env_var_name_TARGET_DEVICE << " has unsupported value of 'ALL_DEVICES'. This is the same as saying NOT_SPECIFIED." << std::endl;
					return;
				}

				device_flags flg = parse_device_flag(env_var_val);
				if(flg == device_flags::UNKNOWN) {
					stringstream ss;
					ss << "Environment variable " << env_var_name_TARGET_DEVICE << " has unknown value: '" << env_var_val << "'";
					throw amptest_cascade_failure(ss.str());
				}

				if(flg != device_flags::NOT_SPECIFIED
				   && !has_any_bits_set(flg, valid_flags)
				   ) {
					stringstream ss;
					ss << "Environment variable " << env_var_name_TARGET_DEVICE << " has an unsupported value: '" << env_var_val << "'";
					throw amptest_cascade_failure(ss.str());
				} else {
					dmf_TARGET_DEVICE_flags = flg;
				}
			}

			known_IHV_failure parse_IHV_failure(const std::string& failure_str) {
				string::size_type pos;
				known_IHV_failure failure = known_IHV_failure::NONE;

				// Determine the IHV
				pos = failure_str.find(':');
				if(failure_str.compare(0, pos, "ati") == 0) {
					failure |= known_IHV_failure::IHV_ATI;
				} else if(failure_str.compare(0, pos, "nvidia") == 0) {
					failure |= known_IHV_failure::IHV_NVIDIA;
				} else if(failure_str.compare(0, pos, "warp") == 0) {
					failure |= known_IHV_failure::IHV_WARP;
				} else {
					stringstream ss;
					ss << "Environment variable " << env_var_name_known_IHV_failures << " is missing a valid IHV specification: '" << failure_str << "'. Valid IHV tokens: " << IHV_FAILURE_IHV_TOKENS;
					throw amptest_cascade_failure(ss.str());
				}

				// Parse the context parameters
				if(pos != string::npos) {
					auto parts = split_string(failure_str.substr(pos+1), ',', true);
					for_each(parts.begin(), parts.end(), [&failure](const string& part) {
							if(part == "win7") {
								failure |= known_IHV_failure::OS_WIN7;
							} else if(part == "win8") {
								failure |= known_IHV_failure::OS_WIN8;

								// Architecture
							} else if(part == "x86") {
								failure |= known_IHV_failure::ARCH_X86;
							} else if(part == "amd64") {
								failure |= known_IHV_failure::ARCH_AMD64;
							} else if(part == "arm") {
								failure |= known_IHV_failure::ARCH_ARM;

								// Build flavor
							} else if(part == "chk") {
								failure |= known_IHV_failure::FLAV_CHK;
							} else if(part == "ret") {
								failure |= known_IHV_failure::FLAV_RET;

								// Buffer aliasing
							} else if(part == "aliasing_forced") {
								failure |= known_IHV_failure::ALIASING_FORCED;
							} else if(part == "aliasing_not_forced") {
								failure |= known_IHV_failure::ALIASING_NOT_FORCED;

							} else {
								stringstream ss;
								ss << "Environment variable " << env_var_name_known_IHV_failures << " has an unknown context parameter value: '" << part << "'. Valid context tokens: " << IHV_FAILURE_CONTEXT_TOKENS;
								throw amptest_cascade_failure(ss.str());
							}
						});
				}

				return failure;
			}

			void parse_IHV_failures() {
				dmf_known_IHV_failures.clear();

				string env_var_val = trim(amptest_context.get_environment_variable(env_var_name_known_IHV_failures));

				// Split by ' ' in case of RUNALL_CROSSLIST usages
				auto parts = split_string(env_var_val, ' ', true);
				for_each(parts.begin(), parts.end(), [](const string& part) {
						known_IHV_failure known_failure = parse_IHV_failure(part);
						dmf_known_IHV_failures.push_back(known_failure);
					});
			}

			void parse_IHV_failure_behavior() {
				dmf_known_IHV_failure_behavior = known_IHV_failure_behavior::EXCLUDE_DEVICE;

				string env_var_val = trim(amptest_context.get_environment_variable(env_var_name_known_IHV_failure_behavior));

				// Split by ' ' in case of RUNALL_CROSSLIST usages, we take the last one specified
				auto parts = split_string(env_var_val, ' ', true);
				for_each(parts.begin(), parts.end(), [](const string& part) {
						if(part == "RESET") {
							// Explicitly set to the default value
							dmf_known_IHV_failure_behavior = known_IHV_failure_behavior::EXCLUDE_DEVICE;
						} else if(part == "EXCLUDE_DEVICE") {
							dmf_known_IHV_failure_behavior = known_IHV_failure_behavior::EXCLUDE_DEVICE;
						} else if(part == "IGNORE") {
							dmf_known_IHV_failure_behavior = known_IHV_failure_behavior::IGNORE_FAILURE;
						} else if(part == "FAIL_TEST") {
							dmf_known_IHV_failure_behavior = known_IHV_failure_behavior::FAIL_TEST;
						} else if(part == "SKIP_TEST") {
							dmf_known_IHV_failure_behavior = known_IHV_failure_behavior::SKIP_TEST;
						} else {
							stringstream ss;
							ss << "Environment variable " << env_var_name_known_IHV_failure_behavior << " has an unsupported value: '" << part << "'";
							throw amptest_cascade_failure(ss.str());
						}
					});
			}

			void initialize_dmf() {
				if(is_dmf_initialized) {
					throw amptest_cascade_failure("The DMF is already initialized.");
				}

				Log(LogType::Info, true) << "DMF: Initializing the AMPTest Device Management Framework..." << std::endl;

				// Look for common typos in the environment variable name
				if(amptest_context.get_environment_variable("AMPTEST_AVAILABLE_DEVICES").length() > 0) {
					stringstream ss;
					ss << "Typo: Environment variable AMPTEST_AVAILABLE_DEVICES is not supported. Use " << env_var_name_allowed_devices << " instead.";
					throw amptest_cascade_failure(ss.str());
				}
				if(amptest_context.get_environment_variable("IHV_FAILURE").length() > 0) {
					stringstream ss;
					ss << "Typo: Environment variable IHV_FAILURE is not supported. Use " << env_var_name_known_IHV_failures << " instead.";
					throw amptest_cascade_failure(ss.str());
				}

				// Allowed Devices
				parse_allowed_devices();
				Log(LogType::Info, true) << "DMF:    Allowed device types: " << device_flags_to_string(dmf_allowed_device_flags) << std::endl;

				parse_allowed_gpu_device_paths();
				if(dmf_allowed_gpu_device_paths.size() > 0) {
					Log(LogType::Info, true) << "DMF:    Allowed GPU device paths: " << std::endl;
					std::for_each(dmf_allowed_gpu_device_paths.begin(), dmf_allowed_gpu_device_paths.end(), [](const wstring& dpath) {
							WLog(LogType::Info, true) << "DMF:       " << dpath << std::endl;
						});
				}

				// Supported Devices
				if(!dmf_supported_device_flags_set_in_code) {
					parse_supported_devices();
				}
				Log(LogType::Info, true) << "DMF:    Supported devices for this test: " << device_flags_to_string(dmf_supported_device_flags) << std::endl;

				// BACKWARD COMPATABILITY
				parse_TARGET_DEVICE();
				if(dmf_TARGET_DEVICE_flags != device_flags::NOT_SPECIFIED) {
					Log(LogType::Info, true) << "DMF:    TARGET_DEVICE (deprecated!!!): " << device_flags_to_string(dmf_TARGET_DEVICE_flags) << std::endl;
				}

				// Detect IHV failures
				parse_IHV_failures();
				parse_IHV_failure_behavior();
				if(dmf_known_IHV_failures.size() > 0) {
					// Only bother reporting the IHV failure behavior setting if the test has some specified.
					Log(LogType::Info, true) << "DMF:    Known IHV failures have been marked for this test. count = " << dmf_known_IHV_failures.size() << std::endl;
					Log(LogType::Info, true) << "DMF:    IHV failure behavior: " << get_known_IHV_failure_behavior_name() << std::endl;
				}

				Log(LogType::Info, true) << "DMF: Finished Initializing." << std::endl << std::endl;
				is_dmf_initialized = true;
			}

			inline void ensure_dmf_initialized() {
				if(!is_dmf_initialized) initialize_dmf();
			}

#pragma endregion

			typedef std::pair<device_bit_flags, accelerator> device_info;

			inline device_bit_flags compute_device_bit_flags(const accelerator& device) {
				device_bit_flags bit_flags = device_bit_flags::EMPTY;
				const wstring& dpath = device.get_device_path();
				const wstring& ddesc = device.get_description();

				// First determine the type of the device
				if(!device.get_is_emulated()) {
					// It's a gpu device
					bit_flags |= device_bit_flags::IS_GPU;

					// Determine it's vendor by looking at the description
					if(ddesc.find(L"ATI") != wstring::npos || ddesc.find(L"AMD") != wstring::npos) {
						bit_flags |= device_bit_flags::IHV_ATI;
					} else if(ddesc.find(L"NVIDIA") != wstring::npos) {
						bit_flags |= device_bit_flags::IHV_NVIDIA;
					} else {
						WLog(LogType::Warning, true) << "Could not determine IHV for accelerator '" << ddesc << "' (" << dpath << ")" << std::endl;
					}
				} else {
					// Emulated devices
					bit_flags |= device_bit_flags::IS_EMULATED;

					if(dpath == accelerator::cpu_accelerator) {
						bit_flags |= device_bit_flags::IS_CPU;
#ifdef AMP_TEST_PLATFORM_MSVC
					} else if(dpath == accelerator::direct3d_warp) {
						bit_flags |= device_bit_flags::IS_WARP;
					} else if(dpath == accelerator::direct3d_ref) {
						bit_flags |= device_bit_flags::IS_REF;
#endif
					} else {
						WLog(LogType::Error, true) << "Unknown accelerator '" << ddesc << "' (" << dpath << ")" << std::endl;
					}
				}

				// Now set the capabilities bits
				if(device.get_supports_double_precision()) {
					bit_flags |= device_bit_flags::SUPPORTS_DOUBLE;
				}
				if(device.get_supports_limited_double_precision()) {
					bit_flags |= device_bit_flags::SUPPORTS_LIMITED_DOUBLE;
				}

				return bit_flags;
			}

			// Gets the flags for all the device types supported on the machine.
			device_flags get_all_existing_device_types() {
				device_flags all_flags = device_flags::NOT_SPECIFIED;

				// Go thru each device and determine which device_flags they'd match
				vector<accelerator> all_devices = accelerator::get_all();
				std::for_each(all_devices.begin(), all_devices.end(), [&](accelerator& accl) {
						device_bit_flags dev_bits = compute_device_bit_flags(accl);
						// Exclude the CPU device
						if(!has_bits_set(dev_bits, device_bit_flags::IS_CPU)) {
							if(has_bits_set(dev_bits, device_bit_flags::IS_EMULATED)) {
								all_flags |= device_flags::EMULATED;
							} else {
								all_flags |= device_flags::NOT_EMULATED;
							}
							if(has_bits_set(dev_bits, device_bit_flags::IS_REF)) {
								all_flags |= device_flags::D3D11_REF;
							}
							if(has_bits_set(dev_bits, device_bit_flags::IS_WARP)) {
								all_flags |= device_flags::D3D11_WARP;
							}
							if(has_bits_set(dev_bits, device_bit_flags::IS_GPU)) {
								all_flags |= device_flags::D3D11_GPU;

								// We only set the D3D11 IHV flags if they are a GPU AND from the IHV
								if(has_bits_set(dev_bits, device_bit_flags::IHV_ATI)) {
									all_flags |= device_flags::D3D11_ATI;
								}
								if(has_bits_set(dev_bits, device_bit_flags::IHV_NVIDIA)) {
									all_flags |= device_flags::D3D11_NVIDIA;
								}
							}
						}
					});

				return all_flags;
			}

#pragma region device predicates

			inline bool is_compute_device_type_match(device_bit_flags device_bits, device_flags required_flags) {
				// If none of these bits are set, then we don't filter on compute device type
				if(!has_any_bits_set(required_flags, compute_device_type_flags_mask)) {
					return true;
				}

				return (has_bits_set(required_flags, device_flags::EMULATED)     && has_bits_set(device_bits, device_bit_flags::IS_EMULATED))
					|| (has_bits_set(required_flags, device_flags::NOT_EMULATED) && !has_bits_set(device_bits, device_bit_flags::IS_EMULATED))
					|| (has_bits_set(required_flags, device_flags::D3D11_REF)    && has_bits_set(device_bits, device_bit_flags::IS_REF))
					|| (has_bits_set(required_flags, device_flags::D3D11_WARP)   && has_bits_set(device_bits, device_bit_flags::IS_WARP))
					|| (has_bits_set(required_flags, device_flags::D3D11_GPU)    && has_bits_set(device_bits, device_bit_flags::IS_GPU))
					|| (has_bits_set(required_flags, device_flags::D3D11_NVIDIA) && has_bits_set(device_bits, device_bit_flags::IS_GPU | device_bit_flags::IHV_NVIDIA))
					|| (has_bits_set(required_flags, device_flags::D3D11_ATI)    && has_bits_set(device_bits, device_bit_flags::IS_GPU | device_bit_flags::IHV_ATI))
					;
			}

			inline bool is_device_capabilities_match(device_bit_flags device_bits, device_flags required_flags) {
				if(!has_any_bits_set(required_flags, device_capability_flags_mask)) {
					return true;
				}

				return false
					// double support flags
					|| (has_bits_set(required_flags, device_flags::DOUBLE) && has_bits_set(device_bits, device_bit_flags::SUPPORTS_DOUBLE))
					|| (has_bits_set(required_flags, device_flags::LIMITED_DOUBLE) && has_bits_set(device_bits, device_bit_flags::SUPPORTS_LIMITED_DOUBLE))
					|| (has_bits_set(required_flags, device_flags::LIMITED_DOUBLE_ONLY)
						&& has_bits_set(device_bits, device_bit_flags::SUPPORTS_LIMITED_DOUBLE)
						&& !has_bits_set(device_bits, device_bit_flags::SUPPORTS_DOUBLE)
						)
					|| (has_bits_set(required_flags, device_flags::NO_DOUBLE)
						&& !has_bits_set(device_bits, device_bit_flags::SUPPORTS_LIMITED_DOUBLE)
						&& !has_bits_set(device_bits, device_bit_flags::SUPPORTS_DOUBLE)
						)
					;
			}

			inline bool is_amptest_device_allowed(const device_info& dinfo) {
				// The AMPTEST_ALLOWED_DEVICES only allow specifying device type flags.

				if(has_bits_set(dinfo.first, device_bit_flags::IS_CPU)) {
					// The CPU device is always ALLOWED
					return true;
				}

				if(!is_compute_device_type_match(dinfo.first, dmf_allowed_device_flags)) {
					return false;
				}

				// Filter GPUs by the allowed device types AND by device path
				if(has_bits_set(dinfo.first, device_bit_flags::IS_GPU) && dmf_allowed_gpu_device_paths.size() > 0) {
					const wstring& dpath = dinfo.second.get_device_path();
					return std::any_of(dmf_allowed_gpu_device_paths.begin(), dmf_allowed_gpu_device_paths.end()
							, [&](const wstring& allowed_dpath) { return allowed_dpath == dpath; }
							)
						;
				} else {
					return true;
				}
			}

			inline bool is_amptest_device_supported(const device_info& dinfo) {
				bool is_supported = true;

				// Check the device type flags
				if(!has_bits_set(dinfo.first, device_bit_flags::IS_CPU)) {
					is_supported &= is_compute_device_type_match(dinfo.first, dmf_supported_device_flags);
				}   // The CPU is always a SUPPORTED device type

				// Check the device capabilities flags
				is_supported &= is_device_capabilities_match(dinfo.first, dmf_supported_device_flags);

				return is_supported;
			}

			inline bool is_amptest_device_the_TARGET_DEVICE(const device_info& dinfo) {
				// TARGET_DEVICE only supports device type flags:
				return is_compute_device_type_match(dinfo.first, dmf_TARGET_DEVICE_flags);
			}

			inline bool is_amptest_device_required(const device_info& dinfo, device_flags required_flags) {
				bool is_supported = true;

				// Check the device type flags
				if(has_bits_set(dinfo.first, device_bit_flags::IS_CPU)) {
					// The CPU is valid only if it's been included
					is_supported &= has_bits_set(required_flags, device_flags::INCLUDE_CPU);
				} else {
					is_supported &= is_compute_device_type_match(dinfo.first, required_flags);
				}

				// Check the device capabilities flags
				is_supported &= is_device_capabilities_match(dinfo.first, required_flags);

				return is_supported;
			}

			inline bool is_device_known_IHV_failure(device_bit_flags device_bits) {
				return std::any_of(dmf_known_IHV_failures.begin(), dmf_known_IHV_failures.end(), [device_bits](known_IHV_failure failure) {
					// Check the IHV first
					if(!(  (has_bits_set(failure, known_IHV_failure::IHV_ATI)    && has_bits_set(device_bits, device_bit_flags::IHV_ATI))
						|| (has_bits_set(failure, known_IHV_failure::IHV_NVIDIA) && has_bits_set(device_bits, device_bit_flags::IHV_NVIDIA))
						|| (has_bits_set(failure, known_IHV_failure::IHV_WARP)   && has_bits_set(device_bits, device_bit_flags::IS_WARP))
						)) {
						// then the device isn't the same IHV as this failure
						return false;
					}

					// Check against the test context
					bool matches_context = true;

					// context - Build Flavor
					if(has_any_bits_set(failure, known_IHV_failure::FLAV_MASK)) {
						Log(LogType::Warning, true) << "DMF: amptest_context.is_chk_build and is_ret_build functions are not implemented yet. Ignoring IHV_FAILURE context specifying build flavor." << std::endl;
					}

					// context - AMP buffer aliasing
					if(has_any_bits_set(failure, known_IHV_failure::ALIASING_MASK)) {
						if(has_bits_set(failure, known_IHV_failure::ALIASING_FORCED)) {
							matches_context &= amptest_context.is_buffer_aliasing_forced();
						} else if(has_bits_set(failure, known_IHV_failure::ALIASING_NOT_FORCED)) {
							matches_context &= !amptest_context.is_buffer_aliasing_forced();
						} else {
							throw amptest_exception("is_device_known_IHV_failure_impl: Unhandled aliasing flag.");
						}
					}

					return matches_context;
				});
			}

			#pragma endregion

			/// Gets a string that represents the device via a list of flags.
			std::string retrieved_device_type_to_string(const accelerator& device) {
				device_bit_flags dev_bit_flags = compute_device_bit_flags(device);

				// Handle the CPU seperately because it's not a value in device_flags.
				if(has_bits_set(dev_bit_flags, device_bit_flags::IS_CPU)) {
					return "CPU";
				}

				// We don't want to print out all the flags, just the important ones.
				device_flags dev_flgs = device_flags::UNKNOWN;
				if(has_bits_set(dev_bit_flags, device_bit_flags::IS_REF)) {
					dev_flgs = device_flags::D3D11_REF;
				} else if(has_bits_set(dev_bit_flags, device_bit_flags::IS_WARP)) {
					dev_flgs = device_flags::D3D11_WARP;
				} else if(has_bits_set(dev_bit_flags, device_bit_flags::IS_GPU)) {
					if(has_bits_set(dev_bit_flags, device_bit_flags::IHV_ATI)) {
						dev_flgs = device_flags::D3D11_ATI;
					} else if(has_bits_set(dev_bit_flags, device_bit_flags::IHV_NVIDIA)) {
						dev_flgs = device_flags::D3D11_NVIDIA;
					} else {
						// Some unknown IHV
						dev_flgs = device_flags::D3D11_GPU;
					}
				}

				return device_flags_to_string(dev_flgs);
			}

			/// Gets a string that represents the device via a list of flags.
			std::string retrieved_device_caps_to_string(const accelerator& device) {
				device_flags caps_flgs = device_flags::UNKNOWN;

				// Determine the max double support
				if(device.get_supports_double_precision()) {
					caps_flgs = device_flags::DOUBLE;
				} else if(device.get_supports_limited_double_precision()) {
					caps_flgs = device_flags::LIMITED_DOUBLE_ONLY;
				} else {
					caps_flgs = device_flags::NO_DOUBLE;
				}

				return device_flags_to_string(caps_flgs);
			}

			/// Predicate indicating whether the left device is of higher priority than the right device.
			bool in_priority_order(const device_info& dev1, const device_info& dev2) {
				const wstring& dpath1 = dev1.second.get_device_path();
				const wstring& dpath2 = dev2.second.get_device_path();

				if(dev1.second.get_is_emulated() != dev2.second.get_is_emulated()) {
					return !dev1.second.get_is_emulated();
				} else if(dpath1 == dpath2) {
					return true;
				} else if(dev1.second.get_is_emulated()) {
					// Compare emulated devices against each other
					return
#ifdef AMP_TEST_PLATFORM_MSVC
						dpath1 == accelerator::direct3d_warp ||
#endif
						dpath2 == accelerator::cpu_accelerator
						;
				} else {
					// Compare GPUs: double precision support, is used for display

					if(dev1.second.get_supports_double_precision() != dev2.second.get_supports_double_precision()) {
						return dev1.second.get_supports_double_precision();
					} else if(dev1.second.get_supports_limited_double_precision() != dev2.second.get_supports_limited_double_precision()) {
						return dev1.second.get_supports_limited_double_precision();
					} else if(dev1.second.get_has_display() != dev2.second.get_has_display()) {
						return !dev1.second.get_has_display();
					} else {
						return dev1.second.get_dedicated_memory() > dev2.second.get_dedicated_memory();
					}
				}
			}


			std::vector<device_info> get_available_device_infos(device_flags required_flags) {
				// Handle the TARGET_DEVICE support here since it's the core of all the APIs
				if(dmf_TARGET_DEVICE_flags != device_flags::NOT_SPECIFIED && dmf_TARGET_DEVICE_throws_failure) {
					throw amptest_cascade_failure("TARGET_DEVICE is used and no longer supported.");
				}

				vector<accelerator> all_devices = accelerator::get_all();

				// Compute the bit flags for each device
				vector<device_info> deviceInfos;
				std::for_each(all_devices.begin(), all_devices.end(), [&](accelerator& accl) {
					device_info dinfo(compute_device_bit_flags(accl), accl);
					bool keep = true;

					keep &= is_amptest_device_allowed(dinfo);
					keep &= is_amptest_device_supported(dinfo);
					keep &= is_amptest_device_required(dinfo, required_flags);

					// Backward compatability for TARGET_DEVICE
					keep &= is_amptest_device_the_TARGET_DEVICE(dinfo);

					if(keep) {
						deviceInfos.push_back(dinfo);
					}
				});

				// Sort the remaining devices by priority
				std::sort(deviceInfos.begin(), deviceInfos.end(), in_priority_order);

				return deviceInfos;
			}

			accelerator require_device_core(device_flags required_flags, vector<wstring> excluded_device_paths = vector<wstring>(0)) {
				// Report what we're doing
				Log(LogType::Info, true) << "DMF: Getting required device: " << device_flags_to_string(required_flags) << std::endl;
				if(excluded_device_paths.size() > 0) {
					Log(LogType::Info, true) << "DMF:    excluding:" << std::endl;
					std::for_each(excluded_device_paths.begin(), excluded_device_paths.end(), [](const wstring& dpath) {
						WLog(LogType::Info, true) << "DMF:        " << dpath << std::endl;
					});
				}

				vector<device_info> avail_infos = get_available_device_infos(required_flags);

				auto newLast = avail_infos.end();

				// Handle the AMP_DEFAULT flag
				if(has_bits_set(required_flags, device_flags::AMP_DEFAULT)) {
					throw amptest_cascade_failure("device_flags::AMP_DEFAULT is not implemented yet.");
				}

				// Exclude devices by path
				if(excluded_device_paths.size() > 0) {
					newLast = remove_if(avail_infos.begin(), newLast, [&](device_info dinfo) {
						const wstring& dpath = dinfo.second.get_device_path();
						return std::any_of(excluded_device_paths.begin(), excluded_device_paths.end()
							, [&](const wstring& excluded_dpath) { return excluded_dpath == dpath; }
						);
					});
				}

				// When the IHV failure behavior is EXCLUDE_DEVICE, remove the ones that are knownfail
				if(dmf_known_IHV_failure_behavior == known_IHV_failure_behavior::EXCLUDE_DEVICE
					&& dmf_known_IHV_failures.size() != 0
					) {
					newLast = remove_if(avail_infos.begin(), newLast, [](device_info dinfo) {
						if(is_device_known_IHV_failure(dinfo.first)) {
							Log_writeline(LogType::Warning, "DMF:    excluding knownfail IHV device: %ws (%s)", dinfo.second.get_description().c_str(), retrieved_device_type_to_string(dinfo.second).c_str());
							return true;
						} else {
							return false;
						}
					});
				}

				// Make sure we have one to return
				if(newLast == avail_infos.begin()) {
					throw amptest_skip("The required device could not be retrieved.");
				}

				accelerator device = avail_infos[0].second;

				// Log what we are returning
				auto device_type = retrieved_device_type_to_string(device);
				auto device_caps = retrieved_device_caps_to_string(device);
				Log(LogType::Info, true) << "DMF:    Returning " << device_type << " (" << device_caps << ")"
					<< " accelerator: " << device.get_description() << " (" << device.get_device_path() << ")" << std::endl;

				// Handle IHV failures
				if(dmf_known_IHV_failure_behavior != known_IHV_failure_behavior::EXCLUDE_DEVICE
					&& is_device_known_IHV_failure(avail_infos[0].first)
					) {
					static const char* err_msg = "The retrieved device is marked as a known IHV device failure.";
					switch(dmf_known_IHV_failure_behavior) {
					case known_IHV_failure_behavior::IGNORE_FAILURE:
						Log(LogType::Warning, true) << "DMF: " << err_msg << std::endl;
						break;
					case known_IHV_failure_behavior::SKIP_TEST:
						throw amptest_skip(err_msg);
					case known_IHV_failure_behavior::FAIL_TEST:
						throw amptest_failure(err_msg);
					default:
						throw amptest_exception("The IHV failure behavior is not handled.");
					}
				}

				Log(LogType::Info, true) << std::endl;
				return device;
			}

			template <typename TFunc>
			auto exit_on_skip_if(bool cond, TFunc func) -> decltype(func()) {
				if(cond) {
					try {
						return func();
					} catch(amptest_skip& ex) {
						Log(LogType::Warning, true) << ex.what() << std::endl;
						exit(runall_skip);
					}
				} else {
					return func();
				}
			}

		}

		/// Parses the string as a single device_flags type.
		device_flags AMP_TEST_API parse_device_flag(const std::string& str) {
			if(str == "NOT_SPECIFIED") return device_flags::NOT_SPECIFIED;

			if(str == "INCLUDE_CPU") return device_flags::INCLUDE_CPU;
			if(str == "D3D11_REF") return device_flags::D3D11_REF;
			if(str == "D3D11_WARP") return device_flags::D3D11_WARP;
			if(str == "D3D11_GPU") return device_flags::D3D11_GPU;
			if(str == "D3D11_ATI") return device_flags::D3D11_ATI;
			if(str == "D3D11_NVIDIA") return device_flags::D3D11_NVIDIA;

			if(str == "EMULATED") return device_flags::EMULATED;
			if(str == "NOT_EMULATED") return device_flags::NOT_EMULATED;
			if(str == "DOUBLE") return device_flags::DOUBLE;
			if(str == "LIMITED_DOUBLE") return device_flags::LIMITED_DOUBLE;
			if(str == "LIMITED_DOUBLE_ONLY") return device_flags::LIMITED_DOUBLE_ONLY;
			if(str == "NO_DOUBLE") return device_flags::NO_DOUBLE;

			if(str == "AMP_DEFAULT") return device_flags::AMP_DEFAULT;

			return device_flags::UNKNOWN;
		}

		device_flags AMP_TEST_API parse_device_flags(const std::string& src_desc, const std::string& str, char separator, device_flags valid_flags) {
			device_flags flags = device_flags::NOT_SPECIFIED;

			// Parse the list
			auto parts = split_string(trim(str), separator, true);
			for (auto&& prt : parts) {
				string part = trim(prt);
				if(part.length() > 0) {
					device_flags flg = parse_device_flag(part);
					if(flg == device_flags::UNKNOWN) {
						stringstream ss;
						ss << "Unknown device_flag '" << part << "'";
						if(src_desc.length() > 0) ss << " found in " << src_desc;
						ss << ".";
						throw amptest_cascade_failure(ss.str());
					}

					flags |= flg;
				}
			}

			// Detect invalid flags
			if(valid_flags != device_flags::NOT_SPECIFIED) {
				device_flags invalid_flags = details::get_invalid_flags(flags, valid_flags);
				if(invalid_flags != device_flags::NOT_SPECIFIED) {
					stringstream ss;
					ss << "The device_flags " << device_flags_to_string(invalid_flags) << " is not valid";
					if(src_desc.length() > 0) ss << " for " << src_desc;
					ss << ".";
					throw amptest_cascade_failure(ss.str());
				}
			}

			return flags;
		}

		/// Converts the device_flags to a string representation.
		std::string AMP_TEST_API device_flags_to_string(device_flags flags) {
			vector<string> parts;

			if(has_bits_set(flags, device_flags::INCLUDE_CPU)) parts.push_back("INCLUDE_CPU");

			if(has_bits_set(flags, device_flags::D3D11_REF)) parts.push_back("D3D11_REF");
			if(has_bits_set(flags, device_flags::D3D11_WARP)) parts.push_back("D3D11_WARP");
			if(has_bits_set(flags, device_flags::D3D11_GPU)) parts.push_back("D3D11_GPU");
			if(has_bits_set(flags, device_flags::D3D11_ATI)) parts.push_back("D3D11_ATI");
			if(has_bits_set(flags, device_flags::D3D11_NVIDIA)) parts.push_back("D3D11_NVIDIA");

			if(has_bits_set(flags, device_flags::EMULATED)) parts.push_back("EMULATED");
			if(has_bits_set(flags, device_flags::NOT_EMULATED)) parts.push_back("NOT_EMULATED");
			if(has_bits_set(flags, device_flags::DOUBLE)) parts.push_back("DOUBLE");
			if(has_bits_set(flags, device_flags::LIMITED_DOUBLE)) parts.push_back("LIMITED_DOUBLE");
			if(has_bits_set(flags, device_flags::LIMITED_DOUBLE_ONLY)) parts.push_back("LIMITED_DOUBLE_ONLY");
			if(has_bits_set(flags, device_flags::NO_DOUBLE)) parts.push_back("NO_DOUBLE");

			if(has_bits_set(flags, device_flags::AMP_DEFAULT)) parts.push_back("AMP_DEFAULT");
			if(has_bits_set(flags, device_flags::UNKNOWN)) parts.push_back("UNKNOWN");

			// Determine if a bit is set that we aren't aware of:
			device_flags unknown_flags = details::get_invalid_flags(flags, all_valid_device_flags | device_flags::UNKNOWN);
			if(unknown_flags != device_flags::NOT_SPECIFIED) {
				throw amptest_cascade_failure("device_flags_to_string detected device_flag bits that aren't detected.");
			}

			// Detect when nothing was specified
			if(parts.size() == 0) {
				return "NOT_SPECIFIED";
			}

			// Loop thru the parts and add a seperator
			stringstream ss;
			ss << parts[0];
			std::for_each(parts.begin()+1, parts.end(), [&](const string& part) { ss << " | " << part; });

			return ss.str();
		}

		void AMP_TEST_API set_amptest_supported_devices(device_flags required_flags) {
			if(required_flags != device_flags::NOT_SPECIFIED) {
				device_flags invalid_flags = details::get_invalid_flags(required_flags, details::valid_supported_device_flags);
				if(invalid_flags != device_flags::NOT_SPECIFIED) {
					stringstream ss;
					ss << "required_flags has invalid flags set: " << device_flags_to_string(invalid_flags);
					throw std::invalid_argument(ss.str());
				}
			}
			if(details::is_dmf_initialized) {
				throw amptest_cascade_failure("set_amptest_supported_devices cannot be called after the AMPTest DMF has been initialized.");
			}

			details::dmf_supported_device_flags = required_flags;
			details::dmf_supported_device_flags_set_in_code = true;
		}

		// This is used internally to control whether require_device should exit with a SKIP or throw an exception
		// when no device is available.
		void AMP_TEST_API set_require_device_behavior(bool exit_on_no_device) {
			if(details::is_dmf_initialized) {
				throw amptest_cascade_failure("set_require_device_behavior cannot be called after the AMPTest DMF has been initialized.");
			}

			details::dmf_require_device_exits_on_no_device = exit_on_no_device;
		}


		std::vector<accelerator> AMP_TEST_API get_available_devices(device_flags required_flags) {
			// Detect invalid flags:
			if(has_bits_set(required_flags, device_flags::AMP_DEFAULT)) {
				throw invalid_argument("device_flags::AMP_DEFAULT is not supported by get_available_devices.");
			}

			details::ensure_dmf_initialized();

			Log(LogType::Info, true) << "DMF: Getting available devices: " << device_flags_to_string(required_flags) << std::endl;
			vector<details::device_info> avail_infos = details::get_available_device_infos(required_flags);
			auto newLast = avail_infos.end();

			// Handle IHV failures
			if(details::dmf_known_IHV_failures.size() != 0) {
				int knownfail_count = 0;
				newLast = remove_if(avail_infos.begin(), newLast, [&knownfail_count](details::device_info dinfo) {
					if(!details::is_device_known_IHV_failure(dinfo.first)) {
						return false;
					}

					knownfail_count++;

					// No matter what, we log the known failure
					const char* msg_format = nullptr;
					if(details::dmf_known_IHV_failure_behavior == details::known_IHV_failure_behavior::IGNORE_FAILURE) {
						msg_format = "DMF:    Device marked as a known IHV failure, but still included: %ws (%s)";
					} else if(details::dmf_known_IHV_failure_behavior == details::known_IHV_failure_behavior::EXCLUDE_DEVICE) {
						msg_format = "DMF:    Excluding device marked as a known IHV failure: %ws (%s)";
					} else { // FAIL_TEST and SKIP_TEST log the same thing
						msg_format = "DMF:    Device marked as a known IHV failure: %ws (%s)";
					}
					Log_writeline(LogType::Warning, msg_format, dinfo.second.get_description().c_str(), details::retrieved_device_type_to_string(dinfo.second).c_str());

					 // We only remove if we would be excluding the device
					return details::dmf_known_IHV_failure_behavior == details::known_IHV_failure_behavior::EXCLUDE_DEVICE;
				});

				if(knownfail_count > 0) {
					static const char* err_msg = "At least one available device is marked as a known IHV failure.";
					if(details::dmf_known_IHV_failure_behavior == details::known_IHV_failure_behavior::SKIP_TEST) {
						throw amptest_skip(err_msg);
					} else if(details::dmf_known_IHV_failure_behavior == details::known_IHV_failure_behavior::FAIL_TEST) {
						throw amptest_failure(err_msg);
					}
				}
			}

			// And convert to just a vector of accelerators
			vector<accelerator> devices;
			std::for_each(avail_infos.begin(), newLast, [&](details::device_info& dinfo) {
				devices.push_back(dinfo.second);
			});

            Log(LogType::Info, true) << "DMF:    Found " << devices.size() << " available devices." << std::endl;
			return devices;
		}

        accelerator AMP_TEST_API require_device(device_flags required_flags) {
			details::ensure_dmf_initialized();

			return details::exit_on_skip_if(details::dmf_require_device_exits_on_no_device, [&](){
				return details::require_device_core(required_flags);
			});
		}

        accelerator AMP_TEST_API require_device(const accelerator& excluded_device, device_flags required_flags) {
			details::ensure_dmf_initialized();

			vector<wstring> excluded_device_paths(1);
			excluded_device_paths[0] = excluded_device.get_device_path();

			return details::exit_on_skip_if(details::dmf_require_device_exits_on_no_device, [&](){
				return details::require_device_core(required_flags, excluded_device_paths);
			});
		}

		#pragma region Obsolete APIs

        bool AMP_TEST_API get_device(accelerator &device, device_flags required_flags) {
			// Note, since we'll be using the new device_flags features we could possibly get more coverage than the previous implementation

			details::ensure_dmf_initialized();

			try {
				device = details::require_device_core(required_flags);
				return true;
			} catch(amptest_skip&) {
				// When someone uses this API, they will handle the 'device not found' scenario manually. therefore we'll suppress the error here
				return false;
			}
		}

		#pragma endregion


		// This namespace is provided to contain functions that are only exposed to test internal
		// parts of the DMF.
		// Note, these functions aren't marked with AMP_TEST_API as the tests in which they are used don't need it.
		namespace dmf_testing {

			/// Ensures that the DMF is initialized without calling any of the public APIs.
			/// This will cause all environment variables related to the DMF to be parsed.
			void ensure_dmf_initialized() {
				details::ensure_dmf_initialized();
			}

			// Tells the DMF whether to exit the application immediately when require_device has no device available.
			// Otherwise, it will throw an amptest_skip exception.
			// The default of the runtime is to call exit(runall_skip). amptest_main calls this to have it NOT exit.
			void set_require_device_behavior(bool exit_on_no_device) {
				details::dmf_require_device_exits_on_no_device = exit_on_no_device;
			}

			// Tells the DMF whether to throw an exception when the TARGET_DEVICE env var is used.
			void set_TARGET_DEVICE_behavior(bool throw_failure) {
				details::dmf_TARGET_DEVICE_throws_failure = throw_failure;
				Log(LogType::Info, true) << "DMF Testing: The TARGET_DEVICE behavior has been changed to: " << (throw_failure ? "throws amptest_failure" : "is supported") << std::endl;
			}

			/// Gets the parsed setting for the AMPTEST_ALLOWED_DEVICES environment variable.
			device_flags get_allowed_device_flags() {
				details::ensure_dmf_initialized();
				return details::dmf_allowed_device_flags;
			}

			/// Gets the parsed setting for the AMPTEST_ALLOWED_DEVICES environment variable.
			std::vector<std::wstring> get_allowed_gpu_device_paths() {
				details::ensure_dmf_initialized();

				// Copy the paths to a new vector
				return vector<wstring>(details::dmf_allowed_gpu_device_paths.begin(), details::dmf_allowed_gpu_device_paths.end());
			}

			/// Gets the parsed setting for the AMPTEST_SUPPORTED_DEVICES environment variable.
			device_flags get_supported_device_flags() {
				details::ensure_dmf_initialized();
				return details::dmf_supported_device_flags;
			}

			/// Gets the parsed setting for the AMPTEST_IHV_FAILURE_BEHAVIOR environment variable.
			std::string get_known_IHV_failure_behavior_name() {
				return details::get_known_IHV_failure_behavior_name();
			}

			bool does_each_device_type_exist(device_flags required_devices) {
				if(required_devices == device_flags::NOT_SPECIFIED) {
					return true;
				}

				device_flags all_flags = details::get_all_existing_device_types();
				if(all_flags == device_flags::NOT_SPECIFIED) {
					return false;
				}

				return has_bits_set(all_flags, required_devices & details::compute_device_type_flags_mask);
			}

			/// Determines if the device matches all the flags specified by required_flags.
			/// The implementation of this method doesn't depend on the DMF implementation
			/// so we aren't using itself to test itself.
			bool does_device_match(const accelerator& device, device_flags required_flags) {
				using details::device_bit_flags;

				// Verify device type flags first
				bool dev_type_match = true;
				const wstring& dpath = device.get_device_path();
				if(dpath == accelerator::cpu_accelerator) {
					// Only include the CPU accelerator if the INCLUDE_CPU flag has been set
					dev_type_match &= has_bits_set(required_flags, device_flags::INCLUDE_CPU);
				} else if(has_any_bits_set(required_flags, details::compute_device_type_flags_mask)) {
					const wstring& ddesc = device.get_description();

					dev_type_match = false
						|| (has_bits_set(required_flags, device_flags::EMULATED) && device.get_is_emulated())
						|| (has_bits_set(required_flags, device_flags::NOT_EMULATED) && !device.get_is_emulated())
#ifdef AMP_TEST_PLATFORM_MSVC
						|| (has_bits_set(required_flags, device_flags::D3D11_REF) && dpath == accelerator::direct3d_ref)
						|| (has_bits_set(required_flags, device_flags::D3D11_WARP) && dpath == accelerator::direct3d_warp)
#endif
						|| (!device.get_is_emulated() && (
							has_bits_set(required_flags, device_flags::D3D11_GPU)
							|| (has_bits_set(required_flags, device_flags::D3D11_ATI) && (ddesc.find(L"ATI") != wstring::npos || ddesc.find(L"AMD") != wstring::npos))
							|| (has_bits_set(required_flags, device_flags::D3D11_NVIDIA) && ddesc.find(L"NVIDIA") != wstring::npos)
							))
						;
				}

				// Verify device capabilities
				bool dev_caps_match = true;
				if(has_any_bits_set(required_flags, details::device_capability_flags_mask)) {
					dev_caps_match = false
						|| (has_bits_set(required_flags, device_flags::DOUBLE) && device.get_supports_double_precision())
						|| (has_bits_set(required_flags, device_flags::LIMITED_DOUBLE) && device.get_supports_limited_double_precision())
						|| (has_bits_set(required_flags, device_flags::LIMITED_DOUBLE_ONLY)
							&& device.get_supports_limited_double_precision()
							&& !device.get_supports_double_precision()
							)
						|| (has_bits_set(required_flags, device_flags::NO_DOUBLE)
							&& !device.get_supports_limited_double_precision()
							&& !device.get_supports_double_precision()
							)
						;
				}

				return dev_type_match && dev_caps_match;
			}

			/// Uses the AMP runtime to count the number of devices that match the required_devices flags.
			size_t count_devices_that_match(device_flags required_devices) {
				auto amp_devices = accelerator::get_all();
				return std::count_if(amp_devices.begin(), amp_devices.end(), [=](accelerator& accl) {
					return does_device_match(accl, required_devices);
				});
			}

			/// Determines if the AMP runtime has at least one device with the required flags.
			bool does_device_exist(device_flags required_devices) {
				auto amp_devices = accelerator::get_all();
				return std::any_of(amp_devices.begin(), amp_devices.end(), [=](accelerator& accl) {
					return does_device_match(accl, required_devices);
				});
			}


			/// Determines if the device and the current amptest_context is known to fail.
			bool AMP_TEST_API is_device_known_IHV_failure(const accelerator& device) {
				details::ensure_dmf_initialized();
				details::device_bit_flags device_bits = details::compute_device_bit_flags(device);
				return details::is_device_known_IHV_failure(device_bits);
			}

		}

		#pragma endregion

	}
}

