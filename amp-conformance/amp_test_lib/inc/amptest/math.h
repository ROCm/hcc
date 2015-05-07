// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
#pragma once
/**********************************************************************************
* amptest\amp.math.h
*
* type-safe math functions.
* On the cpu, these functions prefer those in the std namespace.
* On the gpu, these functions default to the fast_math functions
* and only use the precise_math functions when the inputs are of type double.
**********************************************************************************/

#include <amptest/platform.h>
#include <amp_math.h>
#include <limits>

/*
max_abs_diff:
The absolute maximum difference between two numbers for them to be considered equal.
This is used by verifying abs(a-b) < max_abs_diff


max_rel_diff:
The relative maximum difference between two numbers for them to be considered equal.

First, the distance (DIST) between the two numbers is determined. The distance removes the
issue of magnitude of two relatively close numbers.
Next, the value with the greatest magnitude is identified (GVAL).
i.e. with the highest IEEE Floatingpoint exponent, ignoring negativity.

The final determination is whether the distance between the two values is smaller
than max_rel_diff times the value with greatest magnitude.
i.e. DIST < GVAL * max_rel_diff.

So really, max_rel_diff is a multiplier of the greatest value.
*/

#define DEFAULT_MAX_ABS_DIFF_FLT 0.000005f
#define DEFAULT_MAX_REL_DIFF_FLT 0.0001f

#define DEFAULT_MAX_ABS_DIFF_DBL 0.00000000005
#define DEFAULT_MAX_REL_DIFF_DBL 0.0001

#define DEFAULT_MAX_ULP_DIFF 1

/* This is a work in progress. Currently the expressions don't work.
// These constants should help to get these values w/o needing to contrive an expression or
// using std::numeric_limits, which are not restrict(amp).
// see http://en.wikipedia.org/wiki/Not_a_Number#Creation
#define FLT_POS_INF float(0x7F800000)
#define FLT_NEG_INF float(0xFF800000)
#define FLT_QNAN    float(0xFFC00001)
#define FLT_SNAN    float(0xFF800001)
// The double constants don't work in amp restricted code: C3595: constant value is out of supported range in amp restricted code
//#define DBL_POS_INF	double(0x7FF0000000000000)
//#define DBL_NEG_INF	double(0xFFF0000000000000)
//#define DBL_QNAN    double(0xFFF8000000000001)
//#define DBL_SNAN    double(0xFFF0000000000001)
#define DBL_POS_INF	double(FLT_POS_INF)
#define DBL_NEG_INF	double(FLT_NEG_INF)
#define DBL_QNAN    double(FLT_QNAN)
#define DBL_SNAN    double(FLT_SNAN)

Next thing to try:
INF = (1.0f/0.0f)
NaN = (0.0f/0.0f)
*/


namespace Concurrency
{
    namespace Test
    {
        // The amptest_math namespace shouldn't ever be included by a using declaration.
        // Instead, it should be used explicitly in code.
        // e.g. amptest_math::fabs(..)
        namespace amptest_math
        {
            // templated type for storing
            // different size decimal types as integer
            template<typename T>
            struct ulp_value {};

            // used for 32bit float type
            template <>
            struct ulp_value<float> {
                typedef int32_t bits_t;
            };

            // used for 64bit float type
            template <>
            struct ulp_value<double> {
                typedef int64_t bits_t;
            };


            template<typename T>
            inline bool isnan(T v) restrict(cpu) {
                return std::isnan(v);
            }
            inline bool isnan(float v) restrict(amp) {
                return fast_math::isnan(v) != 0;
            }
            inline bool isnan(double v) restrict(amp) {
                return precise_math::isnan(v) != 0;
            }


            // Provides a simple implementation of the abs functions that will work for data types without needing double support (as does precise_math::fabs).
            template<typename T>
            inline T fabs(T v) restrict(cpu) {
                return std::fabs(v);
            }
            inline float fabs(float v) restrict(amp) {
                return fast_math::fabs(v);
            }
            inline double fabs(double v) restrict(amp) {
                return precise_math::fabs(v);
            }


            inline bool fequal(float v1, float v2) restrict(cpu,amp);
            inline bool fequal(double v1, double v2) restrict(cpu,amp);


            namespace details
            {

                template <typename T>
                inline static bool fequal_impl(const T& v1, const T& v2) restrict(cpu,amp) {
                    // Implementation notes:
                    // when compiled with /fp:fast the restrict(cpu) expression of
                    // 1.#QNAN == 0.0f equates to true. Which is NOT the IEEE standard.
                    // Therefore I must handle any NAN values first and then all other values
                    // can use the == operator.

                    bool is_v1_nan = amptest_math::isnan(v1);
                    bool is_v2_nan = amptest_math::isnan(v2);

                    if(is_v1_nan || is_v2_nan) {
                        return is_v1_nan && is_v2_nan;
                    } else {
                        return v1 == v2;
                    }
                }


                template <typename T>
                inline static bool are_almost_equal_impl(const T& v1, const T& v2, const T& max_abs_diff, const T& max_rel_diff) restrict(cpu,amp) {
                    // Implementation notes:
                    // Refer to notes in the fequal_impl function.

                    if(fequal(v1,v2)) {
                        return true;
                    } else if(amptest_math::isnan(v1) || amptest_math::isnan(v2)) {
                        // If not equal, but one is a NAN, then 'almost' part doesn't matter
                        return false;
                    }

                    // Look for absolute comparison
                    if (amptest_math::fabs(v1 - v2) < max_abs_diff) { // absolute comparison
                        return true;
                    }

                    T diff = 0;
                    T diff2 = max_rel_diff;
                    if (amptest_math::fabs(v1) > amptest_math::fabs(v2))
                    {
                        diff = amptest_math::fabs(v1 - v2);
                        diff2 *= amptest_math::fabs(v1); // Because WDDM1.1 doesn't support double divison, use multiplication here.
                    }
                    else
                    {
                        diff = amptest_math::fabs(v2 - v1);
                        diff2 *= amptest_math::fabs(v2);
                    }

                    return (diff < diff2); // relative comparison
                }


                template <typename T>
                inline typename ulp_value<T>::bits_t get_ulp_diff_impl(const T& v1, const T& v2) restrict(cpu) {

                    typedef typename ulp_value<T>::bits_t bits_t;

                    const static bits_t sign_bit_mask = static_cast<bits_t>(1) << (8 * sizeof(T) - 1);

                    // Make a_int lexicographically ordered as a twos-complement int
                    bits_t a_int = *(bits_t*) &v1;
                    if ( a_int < 0 )
                        a_int = sign_bit_mask - a_int;

                    // Make b_int lexicographically ordered as a twos-complement int
                    bits_t b_int = *(bits_t*) &v2;
                    if ( b_int < 0 )
                        b_int = sign_bit_mask - b_int;

                    // Avoid using abs for future amp implementation
                    // as it adds precision tolerance
                    return (a_int > b_int? a_int - b_int: b_int - a_int);
                }
            }


            // Tests whether two values are explicitly equal and handles denormalized numbers (which == does not)
            inline bool fequal(float v1, float v2) restrict(cpu,amp) {
                return details::fequal_impl<float>(v1, v2);
            }

            // Tests whether two values are explicitly equal and handles denormalized numbers (which == does not)
            inline bool fequal(double v1, double v2) restrict(cpu,amp) {
                return details::fequal_impl<double>(v1, v2);
            }


            // Gets the ULP difference between two decimal numbers.
            // See http://en.wikipedia.org/wiki/Unit_in_the_last_place
            // These two functions is to enforce using
            // get_ulp_diff_impl only with float & double types
            inline ulp_value<float>::bits_t get_ulp_diff(float v1, float v2) {
                return details::get_ulp_diff_impl<float>(v1, v2);
            }
            inline ulp_value<double>::bits_t get_ulp_diff(double v1, double v2) {
                return details::get_ulp_diff_impl<double>(v1, v2);
            }


            template<typename T>
            inline bool are_ulp_equal(const T& v1, const T& v2, typename ulp_value<T>::bits_t max_ulp_diff_allowed)
            {
                // Implementation notes:
                // Refer to notes in the fequal_impl function.
                if(fequal(v1,v2)) {
                    return true;
                } else if(amptest_math::isnan(v1) || amptest_math::isnan(v2)) {
                    // If not equal, but one is a NAN, then 'almost' part doesn't matter
                    return amptest_math::isnan(v1) && amptest_math::isnan(v2);
                }

                return get_ulp_diff( v1, v2 ) <= max_ulp_diff_allowed;
            }


            inline bool are_almost_equal(float v1, float v2
                , const float max_abs_diff = DEFAULT_MAX_ABS_DIFF_FLT
                , const float max_rel_diff = DEFAULT_MAX_REL_DIFF_FLT
                ) restrict(cpu,amp) {
                    return details::are_almost_equal_impl<float>(v1, v2, max_abs_diff, max_rel_diff);
            }

            inline bool are_almost_equal(double v1, double v2
                , const double max_abs_diff = DEFAULT_MAX_ABS_DIFF_DBL
                , const double max_rel_diff = DEFAULT_MAX_REL_DIFF_DBL
                ) restrict(cpu,amp) {
                    return details::are_almost_equal_impl<double>(v1, v2, max_abs_diff, max_rel_diff);
            }
        }
    }
}
