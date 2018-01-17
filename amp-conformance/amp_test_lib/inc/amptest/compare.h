// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
#include <amptest/math.h>
#include <amptest/logging.h>
#include <amptest/operators.h>

namespace Concurrency
{
	namespace Test
	{

		/// Composes a generic string that conveys the index of an element that is incorrect.
		/// I prints the index, expected and actual values along with their hex representation.
		/// The hex representation helps when analyzing failures to identify when the value
		/// is off by one or two random bits. (e.g. like when NVIDIA cards get bits stuck)
		template <typename Tidx, typename Texpected, typename Tactual>
		std::string compose_incorrect_element_message(Tidx idx, Texpected expected, Tactual actual) {
			std::stringstream ss;
			// Use the same default formatting we use in dpctest.cpp\amptest_initialize_logging
			ss.precision(12);	// To help display floats more accurately
			ss << std::boolalpha;	// Make bool values print out as true/false rather than 1/0

			ss << "Element at index " << idx << " is incorrect."
				<< " Expected: " << format_as_code(expected) << " (" << format_as_hex(expected) << "),"
				<< " Actual: " << format_as_code(actual) << " (" << format_as_hex(actual) << ")"
				;
			return ss.str();
		}

		#pragma region type_comparer classes

		template <typename T>
		struct type_comparer {

			type_comparer() restrict(cpu,amp) {}

			//T get_default_value() const restrict(cpu,amp) { return T(); }	// requires default ctor
			//__declspec(property(get = get_default_value)) T default_value;

			bool are_equal(const T& actual, const T& expected) const restrict(cpu,amp) {
				return expected == actual;
			}

			// The following are function signatures that should be defined depending on the type T.
			// In general, no implementation indicates the type T doesn't support it.
			bool isnan(const T&) const restrict(cpu,amp) {
				return false;
			}

		};

		template <>
		struct type_comparer<float> {

			float max_abs_diff;
			float max_rel_diff;

			type_comparer(
				  float max_abs_dif = DEFAULT_MAX_ABS_DIFF_FLT
				, float max_rel_dif = DEFAULT_MAX_REL_DIFF_FLT
				) restrict(cpu,amp)
				: max_abs_diff(max_abs_dif), max_rel_diff(max_rel_dif)
			{}

			//float get_default_value() const restrict(cpu,amp) { return 0.0f; }
			//__declspec(property(get = get_default_value)) float default_value;

			bool are_equal(const float& actual, const float& expected) const restrict(cpu,amp) {
				return amptest_math::are_almost_equal(actual, expected, max_abs_diff, max_rel_diff);
			}

			bool isnan(float val) const restrict(cpu,amp) {
				return amptest_math::isnan(val);
			}

		};

		template <>
		struct type_comparer<double> {

			double max_abs_diff;
			double max_rel_diff;

			type_comparer(
				  double max_abs_dif = DEFAULT_MAX_ABS_DIFF_DBL
				, double max_rel_dif = DEFAULT_MAX_REL_DIFF_DBL
				) restrict(cpu,amp)
				: max_abs_diff(max_abs_dif), max_rel_diff(max_rel_dif)
			{}

			//double get_default_value() const restrict(cpu,amp) { return 0.0; }
			//__declspec(property(get = get_default_value)) double default_value;

			bool are_equal(const double& actual, const double& expected) const restrict(cpu,amp) {
				return amptest_math::are_almost_equal(actual, expected, max_abs_diff, max_rel_diff);
			}

			bool isnan(double val) const restrict(cpu,amp) {
				return amptest_math::isnan(val);
			}

		};

		// Macro for defining the generic type_comparer for a type that isn't supported in AMP-restricted code
		// This just makes the implementation to use restrict(cpu) instead of restrict(cpu,amp).
		#define DEFINE_NON_AMP_TYPE_COMPARER(_T) \
				template <> \
				struct type_comparer<_T> { \
					bool are_equal(const _T& actual, const _T& expected) const { \
						return expected == actual; \
					} \
				}

		DEFINE_NON_AMP_TYPE_COMPARER(char);
		DEFINE_NON_AMP_TYPE_COMPARER(unsigned char);
		DEFINE_NON_AMP_TYPE_COMPARER(short);
		DEFINE_NON_AMP_TYPE_COMPARER(unsigned short);

		#undef DEFINE_NON_AMP_TYPE_COMPARER
		
		#pragma endregion

		#pragma region are_equal

		template <typename T>
		struct are_equal {
		private:
			type_comparer<T> _comparer;
		public:
			are_equal(const type_comparer<T>& comparer = type_comparer<T>()) restrict(cpu,amp)
				: _comparer(comparer)
			{}

			template <typename Texpected>
			bool operator()(const T& actual, const Texpected& expected) const restrict(cpu,amp) {
				return _comparer.are_equal(actual, expected);
			}

		};

		template <typename Tos, typename T>
        inline Tos& operator<<(Tos& os, const are_equal<T>&) {
			os << "are_equal<" << get_type_name<T>() << ">";
			return os;
		}

		#pragma endregion

		#pragma region equal_to

		/// Generic predicate for determining equality of a value
		template <typename T, typename Texpected>
		struct equal_to_func {
		private:
			type_comparer<T> _comparer;
			Texpected _expected;
		public:
			equal_to_func(Texpected expected, const type_comparer<T>& comparer) restrict(cpu,amp)
				: _comparer(comparer), _expected(expected)
			{}

			const Texpected& get_expected() const { return _expected; }

			bool operator()(const T& actual) const restrict(cpu,amp) {
				return _comparer.are_equal(actual, _expected);
			}

		};

		/// This function wrapps the creation of the equal_to functor so the compiler can deduce the
		/// 2nd type parameter.
		template <typename T, typename Texpected>
		inline equal_to_func<T, Texpected> equal_to(Texpected expected, const type_comparer<T>& comparer = type_comparer<T>()) restrict(cpu,amp) {
			return equal_to_func<T, Texpected>(expected, comparer);
		}

		template <typename Tos, typename T, typename Texpected>
        inline Tos& operator<<(Tos& os, const equal_to_func<T, Texpected>& func) {
			os << "equal_to<" << get_type_name<T>() << ">(" << format_as_code(func.get_expected()) << ")";
			return os;
		}

		#pragma endregion

	}
}

