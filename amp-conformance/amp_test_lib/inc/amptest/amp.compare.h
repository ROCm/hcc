// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
/**********************************************************************************
* amp.compare.h
*
*
**********************************************************************************/

#include <amp.h>
#include <amp_math.h>
#include <vector>
#include <sstream>
#include <amptest/logging.h>
#include <amptest/math.h>
#include <amptest/compare.h>
#include <amptest/operators.h>


namespace Concurrency
{
    namespace Test
    {
		template<typename _type> class Difference;

		// The maximum number of incorrect elements to log before just returning the summary
		static const size_t max_failed_elements_to_log = 20;

        // Details namespace serves as private namespace
        namespace details
        {
			// TODO: These AreEqual (and the AreAlmostEqual) functions are now obsolete. Use a type_comparer<T> instead as
			// it provides the 'are almost equal' semantics by default.
			// The only place where this function's used (but shouldn't be) is in Functional\Language\VectorSubset\GeneralUtilitiesLibrary\Pointer\*
			// because each test includes Pointer\inc\common.h which uses these functions. :-(
			// Once those are cleaned up, then these AreEqual and AreAlmostEqual functions can be deleted.

            template<typename T>
            bool AreEqual(const T &v1, const T &v2) restrict(cpu,amp)
            {
                // This function is constructed in a way that requires T
                // only to define operator< to check for equality

                if (v1 < v2)
                {
                    return false;
                }
                if (v2 < v1)
                {
                    return false;
                }
                return true;
            }

			inline bool AreEqual(const char &v1, const char &v2) { return v1 == v2; }
			inline bool AreEqual(const short &v1, const short &v2) { return v1 == v2; }
			inline bool AreEqual(const unsigned char &v1, const unsigned char &v2) { return v1 == v2; }
			inline bool AreEqual(const unsigned short &v1, const unsigned short &v2) { return v1 == v2; }

            template<typename T, typename Tpredicate>
            bool Verify_impl(const T* ary_actual, const T* ary_expected, size_t ary_length, const Tpredicate& pred)
			{
				size_t num_failed = 0;
				for(size_t i = 0; i < ary_length; ++i)
				{
					if (!pred(ary_expected[i], ary_actual[i]))
					{
						num_failed++;

						if(num_failed == 1) {
							Log(LogType::Error, true) << "Verify found elements with incorrect values: " << std::endl;
						}
						if(num_failed <= max_failed_elements_to_log) {
							Log(LogType::Error, true) << "   " << compose_incorrect_element_message(i, ary_expected[i], ary_actual[i]) << std::endl;
						} else if (num_failed == max_failed_elements_to_log+1) {
							Log(LogType::Error, true) << "      and more..." << std::endl;
						}
					}
				}

				if(num_failed != 0) {
					Log(LogType::Error, true) << "      " << num_failed << " out of " << ary_length << " elements failed." << std::endl;
				}

				return num_failed == 0;
			}

		}

        // Compare two floats and return true if they are close to each other.
        inline bool AreAlmostEqual(float v1, float v2,
            const float maxAbsoluteDiff = DEFAULT_MAX_ABS_DIFF_FLT,
            const float maxRelativeDiff = DEFAULT_MAX_REL_DIFF_FLT
			) restrict(cpu,amp)
        {
            return amptest_math::are_almost_equal(v1, v2, maxAbsoluteDiff, maxRelativeDiff);
        }

        // Compare two doubles and return true if they are close to each other.
        inline bool AreAlmostEqual(double v1, double v2,
            const double maxAbsoluteDiff = DEFAULT_MAX_ABS_DIFF_DBL,
            const double maxRelativeDiff = DEFAULT_MAX_REL_DIFF_DBL
			) restrict(cpu,amp)
        {
            return amptest_math::are_almost_equal(v1, v2, maxAbsoluteDiff, maxRelativeDiff);
        }

        // Compare arrays for arbitrary type, T is required to define operator<
        // Returns 'true' if arrays contain same results and 'false' otherwise.
        // Start of Verify functions for c-style arrays
        template<typename T>
        bool Verify(const T *c, const T *refc, size_t size, type_comparer<T> comparer = type_comparer<T>())
        {
			return details::Verify_impl(c, refc, size, [=](T exp, T act) {
					// Need to wrap with lambda so the char/short overloads can be picked by the compiler
					return comparer.are_equal(exp, act);
				});
        }

        // End of Verify functions for c-style arrays

        // Start of Verify functions for std::vector
        template<typename T>
        bool Verify(const std::vector<T> &c, const std::vector<T> &refc, type_comparer<T> comparer = type_comparer<T>())
        {
            if (c.size() != refc.size()) { return false; }
            return Verify(c.data(), refc.data(), refc.size(), comparer);
        }

		// This overload just simplifies when wanting to control the options when comparing floating-point types.
		template<typename T, typename Tmax_args>
		bool Verify(
			const std::vector<T> &c,
			const std::vector<T> &refc,
			const Tmax_args maxAbsoluteDiff,
			const Tmax_args maxRelativeDiff)
        {
			static_assert(std::is_floating_point<Tmax_args>::value, "The args passed in for the limits should be a floating point type (i.e. both float or both double)");

			// This enforces that T must be float/double since (currently) they're the only type_comparers
			// that provide this ctor.
			type_comparer<T> comparer(static_cast<T>(maxAbsoluteDiff), static_cast<T>(maxRelativeDiff));
            return Verify(c, refc, comparer);
        }

        // End of Verify functions for c-style arrays

		#pragma region VerifyDataOnCpu()

		// Verifies that data contained in the two C++ AMP containers differs by value 'diff'. The computation
		// happens on CPU. If any of supplied array is on GPU, it will get copied to CPU.
		template<typename _type, int _rank, template<typename, int> class _amp_container_type_1, template<typename, int> class _amp_container_type_2>
		bool VerifyDataOnCpu(const _amp_container_type_1<_type, _rank>& actual, const _amp_container_type_2<_type, _rank>& expected, _type diff = 0)
		{
			if(actual.get_extent() != expected.get_extent())
			{
				Log(LogType::Error, true) << "Extent values for actual and  expected does not match.";
				Log(LogType::Error, true) << "Actual: " << actual.get_extent() << " Expected: " << expected.get_extent();
				return false;
			}

			std::vector<_type> vect_actual(actual.get_extent().size());
			copy(actual, vect_actual.begin());

			std::vector<_type> vect_expected(expected.get_extent().size());
			copy(expected, vect_expected.begin());

			return Equal(vect_actual.begin(), vect_actual.end(), vect_expected.begin(), Difference<_type>(diff));
		}

		// Verifies that data contained in the supplied C++ AMP container and standard container differs by value 'diff'.
		// The computation happens on CPU. If the supplied array have data on GPU, it will get copied on CPU.
		template<typename _type, int _rank, template<typename, int> class _amp_container_type, template<typename T, typename=std::allocator<T>> class _stl_cont>
		bool VerifyDataOnCpu(const _amp_container_type<_type, _rank>& actual, const _stl_cont<_type>& expected, _type diff = 0)
		{
			if(actual.get_extent().size() != expected.size())
			{
				Log(LogType::Error, true) << "Size of actual and expected does not match.\n";
				Log(LogType::Error, true) << "Size of actual : " << actual.get_extent().size();
				Log(LogType::Error, true) << "Size of expected : " << expected.size();
				return false;
			}

			std::vector<_type> temp_cont(actual.get_extent().size());
			copy(actual, temp_cont.begin());

			return Equal(temp_cont.begin(), temp_cont.end(), expected.begin(), Difference<_type>(diff));
		}

		// Verifies that data contained in the supplied array_view<const T, N> and standard container differs by value 'diff'.
		// The computation happens on CPU. If the supplied array have data on GPU, it will get copied on CPU.
		template<typename _type, int _rank,	template<typename T, typename=std::allocator<T>> class _stl_cont>
		bool VerifyDataOnCpu(const array_view<const _type, _rank>& actual, const _stl_cont<_type>& expected, _type diff = 0)
		{
			if(actual.get_extent().size() != expected.size())
			{
				Log(LogType::Error, true) << "Size of actual and expected does not match.\n";
				Log(LogType::Error, true) << "Size of actual : " << actual.get_extent().size();
				Log(LogType::Error, true) << "Size of expected : " << expected.size();
				return false;
			}

			std::vector<_type> temp_cont(actual.get_extent().size());
			copy(actual, temp_cont.begin());

			return Equal(temp_cont.begin(), temp_cont.end(), expected.begin(), Difference<_type>(diff));
		}

		// Verifies that data containes in the supplied C++ AMP container and standard container differs by value 'diff'.
		// The computation happens on CPU. If the supplied array have data on GPU, it will get copied on CPU.
		template<typename _type, int _rank,	template<typename, int> class _amp_container_type, template<typename T, typename=std::allocator<T>> class _stl_cont>
		bool VerifyDataOnCpu(const _stl_cont<_type>& actual, const _amp_container_type<_type, _rank>& expected, _type diff = 0)
		{
			if(expected.get_extent().size() != actual.size())
			{
				Log(LogType::Error, true) << "Size of expected and actual does not match.\n";
				Log(LogType::Error, true) << "Size of actual: " << actual.size();
				Log(LogType::Error, true) << "Size of expected: " << expected.get_extent().size();
				return false;
			}

			std::vector<_type> temp_cont(expected.get_extent().size());
			copy(expected, temp_cont.begin());

			return Equal(actual.begin(), actual.end(), temp_cont.begin(), Difference<_type>(diff));
		}

		// Verifies that data contained in the two array_view<const T, N> and C++ AMP container differs by
		// value 'diff'. The computation happens on CPU. If any of supplied array is on GPU, it will get copied to CPU.
		template<typename _type, int _rank, template<typename, int> class _amp_container_type>
		bool VerifyDataOnCpu(const array_view<const _type, _rank>& actual, const _amp_container_type<_type, _rank>& expected, _type diff = 0)
		{
			std::vector<_type> vect_actual(actual.get_extent().size());
			copy(actual, vect_actual.begin());

			return VerifyDataOnCpu(vect_actual, expected, diff);
		}


		#pragma endregion

		#pragma region VerifyAllSameValue()

		// Verifies that all the values contained in input container are equal to 'value'
        template<typename _type>
        int VerifyAllSameValue(const std::vector<_type>& inputVector, const _type& value)
        {
			type_comparer<_type> comparer;
            typename std::vector<_type>::const_iterator iter = std::find_if_not(inputVector.begin(), inputVector.end(), [&](_type el) -> bool {
				return comparer.are_equal(el, value);
			});

            if(iter == inputVector.end())
            {
                return -1;
            }
            else
            {
                int res = (int)(iter - inputVector.begin());

				// Now report all the failed elements
				Log(LogType::Error, true) << "VerifyAllSameValue found elements with incorrect values. Expected value: " << format_as_code(value) << std::endl;
				size_t num_failed = 0;
				for(; iter < inputVector.end(); ++iter)
				{
					const auto& elm = *iter;
					if (!comparer.are_equal(elm, value))
					{
						num_failed++;

						if(num_failed <= max_failed_elements_to_log) {
							Log(LogType::Error, true) << "   " << compose_incorrect_element_message((int)(iter - inputVector.begin()), value, elm) << std::endl;
						} else if (num_failed == max_failed_elements_to_log+1) {
							Log(LogType::Error, true) << "     and more..." << std::endl;
						}
					}
				}

				// Always report the summary since this block has already found one
				Log(LogType::Error, true) << "   " << num_failed << " out of " << inputVector.size() << " elements failed." << std::endl;

                return res; // old implementation returns the index of the first element not equal to value.
            }
        }

		// Verifies that all the values contained in input array are equal to 'value'
		template<typename _type, int _rank>
		int VerifyAllSameValue(const array<_type, _rank>& inputArray, const _type& value)
		{
			std::vector<_type> vect1 = inputArray;
			return VerifyAllSameValue<_type>(vect1, value);
		}

		// Verifies that all the values contained in input array view are equal to 'value'
		template<typename _type, int _rank>
		int VerifyAllSameValue(const array_view<_type, _rank>& inputArrayView, const _type& value)
		{
			std::vector<_type> vect1(inputArrayView.get_extent().size());
			copy(inputArrayView, vect1.begin());

			return VerifyAllSameValue<_type>(vect1, value);
		}

		#pragma endregion

		#pragma region VerifyDataOnAcc()

        // Verifies that data containes in the supplied arrays differs by value 'diff'.
        // The computation happens on GPU. The supplied input arrays are required to be on GPU.
        // The result array 'stagingArrResult' should be staging array with source device GPU and dest device CPU.
        template<typename _type, int _rank>
        bool VerifyDataOnAcc(array<_type, _rank>& actual, array<_type, _rank>& expected, array<_type, _rank>& stagingArrResult, _type diff = 0)
        {
            if(actual.get_extent() != expected.get_extent())
            {
                Log(LogType::Error, true) << "Grid values for actual and expected array does not match.";
                Log(LogType::Error, true) << "Actual: " << actual.get_extent() << " Expected: " << expected.get_extent();
                return false;
            }

            if(stagingArrResult.get_extent() != expected.get_extent())
            {
                Log(LogType::Error, true) << "Grid value for result staging array and input containers does not match.";
                Log(LogType::Error, true) << "Input containers: " << actual.get_extent() << " staging array result: " << expected.get_extent();
                return false;
            }

            parallel_for_each(actual.get_extent(), [&actual, &expected, &stagingArrResult](index<_rank> idx) restrict(amp)
            {
                stagingArrResult[idx] = expected[idx] - actual[idx];
            });

            int res = VerifyAllSameValue<_type, _rank>(stagingArrResult, diff);
            if(res == -1)
            {
                return true;
            }
            else
            {
                //TO DO: If needed in case of mismatch log the actual and expected data at point of mismatch.
                return false;
            }
        }

        // Verifies that data containes in the supplied array views differs by value 'diff'.
        // The computation happens on GPU. The supplied input array views should have data on GPU otherwise it
        // will involve implicit caching of data. The result array 'stagingArrResult' should be staging array with source
        // device as GPU and dest as device CPU.
        template<typename _type, int _rank>
        bool VerifyDataOnAcc(array_view<_type, _rank>& actual, array_view<_type, _rank>& expected, array<_type, _rank>& stagingArrResult, _type diff = 0)
        {
            if(actual.get_extent() != expected.get_extent())
            {
                Log(LogType::Error, true) << "Grid values for actual and  expected array view does not match.";
                Log(LogType::Error, true) << "Actual: " << actual.get_extent() << " Expected: " << expected.get_extent();
                return false;
            }

            if(stagingArrResult.get_extent() != expected.get_extent())
            {
                Log(LogType::Error, true) << "Grid value for result staging array and input containers does not match.";
                Log(LogType::Error, true) << "Input containers: " << actual.get_extent() << " staging array result: " << expected.get_extent();
                return false;
            }

            parallel_for_each(actual.get_extent(), [actual, expected, &stagingArrResult](index<_rank> idx) restrict(amp)
            {
                stagingArrResult[idx] = expected[idx] - actual[idx];
            });

            int res = VerifyAllSameValue<_type, _rank>(stagingArrResult, diff);
            if(res == -1)
            {
                return true;
            }
            else
            {
                //TO DO: If needed in case of mismatch log the actual and expected data at point of mismatch.
                return false;
            }
        }

        // Verifies that data containes in the supplied array and array view differs by value 'diff'.
        // The computation happens on GPU. The supplied input array and array view are required to have data on GPU.
        // The result array 'stagingArrResult' should be staging array with source device GPU and dest device CPU.
        template<typename _type, int _rank>
        bool VerifyDataOnAcc(array<_type, _rank>& actual, array_view<_type, _rank>& expected, array<_type, _rank>& stagingArrResult, _type diff = 0)
        {
            if(actual.get_extent() != expected.get_extent())
            {
                Log(LogType::Error, true) << "Grid values for actual array and  expected array view does not match.";
                Log(LogType::Error, true) << "Actual: " << actual.get_extent() << " Expected: " << expected.get_extent();
                return false;
            }

            if(stagingArrResult.get_extent() != expected.get_extent())
            {
                Log(LogType::Error, true) << "Grid value for result staging array and input containers does not match.";
                Log(LogType::Error, true) << "Input containers: " << actual.get_extent() << " staging array result: " << expected.get_extent();
                return false;
            }

            parallel_for_each(actual.get_extent(), [&actual, expected, &stagingArrResult](index<_rank> idx) restrict(amp)
            {
                stagingArrResult[idx] = expected[idx] - actual[idx];
            });

            int res = VerifyAllSameValue<_type, _rank>(stagingArrResult, diff);
            if(res == -1)
            {
                return true;
            }
            else
            {
                //TO DO: If needed in case of mismatch log the actual and expected data at point of mismatch.
                return false;
            }
        }

		#pragma endregion

		// Compares two iterators for equivalence as specified by binary predicate.
		// It uses the standard template library method mismatch(...) for this and reports the first mismatch.
		template<typename InputIterator1, typename InputIterator2, typename BinaryPredicate>
		bool Equal(InputIterator1 beginActual, InputIterator1 endActual, InputIterator2 beginExpected, BinaryPredicate comp)
		{
			std::pair<InputIterator1, InputIterator2> result;
			result = std::mismatch<InputIterator1, InputIterator2>(beginActual, endActual, beginExpected, comp);

			if ( result.first == endActual )
			{
				return true;
			}
			else
			{
				Log(LogType::Error, true) << "Result mismatch between two iterators";
				Log(LogType::Error, true) << "First mismatch at element number: " << result.first - beginActual;
				Log(LogType::Error, true) << "Value at actual iterator = " << *result.first << "\nValue at expected iterator = " << *result.second;
				return false;
			}
		}

		// A functor class which compares the difference between two values with a
		// given value.
		template<typename _type>
		class Difference
		{
		private:
			typename std::remove_const<_type>::type diff;
		public:
			Difference(_type _Diff) : diff(_Diff)
			{
			}

			bool operator()(_type actualValue, _type expectedValue) const
			{
				return (expectedValue - actualValue == diff);
			}
		};

	}
}


