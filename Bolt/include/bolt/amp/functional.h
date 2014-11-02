/***************************************************************************
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/

/*! \file bolt/amp/functional.h
    \brief List all the unary and binary functions.
*/


#pragma once
#if !defined( BOLT_AMP_FUNCTIONAL_H )
#define BOLT_AMP_FUNCTIONAL_H

#include "bolt/amp/bolt.h"


namespace bolt {
namespace amp {
	template<typename Argument1,
		typename Result>
	struct unary_function
		: public std::unary_function<Argument1, Result>
	{
	};

	template<typename Argument1,
		typename Argument2,
		typename Result>
	struct binary_function
		: public std::binary_function<Argument1, Argument2, Result>
	{
	};

/******************************************************************************
 * Unary Operators
 *****************************************************************************/
	template <typename T>
	struct square : public unary_function<T,T>
	{
		T operator() (const T& x)  const restrict(cpu,amp) {
			return x * x;
		}
	};

    template< typename T >
    struct cube : public unary_function<T,T>
    {
        T operator() (const T& x)  const restrict(cpu,amp) { return x * x * x; }
    };


	template<typename T>
	struct negate : public unary_function<T,T>
	{
		T operator()(const T &__x) const restrict(cpu,amp) {return -__x;}
	};

/******************************************************************************
 * Binary Operators
 *****************************************************************************/

	template<typename T>
	struct plus : public binary_function<T,T,T>
	{
		T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp)  {return lhs + rhs;}
	};

    template<typename T>
    struct minus : public binary_function<T,T,T>
    {
        T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs - rhs;}
    };

    template<typename T>
    struct multiplies : public binary_function<T,T,T>
    {
        T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs * rhs;}
    };

    template<typename T>
    struct divides : public binary_function<T,T,T>
    {
        T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs / rhs;}
    };


	  template<typename T>
	  struct maximum : public binary_function<T,T,T>
	  {
	  	T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return rhs > lhs ? rhs:lhs;}
	  };

	  template<typename T>
	  struct minimum : public binary_function<T,T,T>
	  {
	  	T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return rhs < lhs ? rhs:lhs;}
	  };

    template<typename T>
    struct modulus : public binary_function<T,T,T>
    {
        T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs % rhs;}
    };

    template<typename T>
    struct bit_and : public binary_function<T,T,T>
    {
        T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs & rhs;}
    };

    template<typename T>
    struct bit_or : public binary_function<T,T,T>
    {
        T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs | rhs;}
    };

    template<typename T>
    struct bit_xor : public binary_function<T,T,T>
    {
        T operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs ^ rhs;}
    };


    /******************************************************************************
     * Unary Predicates
     *****************************************************************************/

    template<typename T>
    struct logical_not : public unary_function<T,T>
    {
        bool operator()(const T &x) const restrict(cpu,amp) {return !x;}
    };


    /******************************************************************************
     * Binary Predicates
     *****************************************************************************/

    template<typename T>
    struct equal_to : public binary_function<T,T,T>
    {
        bool operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs == rhs;}
    };

    template<typename T>
    struct not_equal_to : public binary_function<T,T,T>
    {
        bool operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs != rhs;}
    };

    template<typename T>
    struct greater : public binary_function<T,T,T>
    {
        bool operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs > rhs;}
    };

	template<typename T>
	struct identity : public unary_function<T,T>
	{
		T operator()(const T &x) const restrict(cpu,amp) {return x;}
	};

    template<typename T>
    struct less : public binary_function<T,T,T>
    {
        bool operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs < rhs;}
    };

    template<typename T>
    struct greater_equal : public binary_function<T,T,T>
    {
        bool operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs >= rhs;}
    };

    template<typename T>
    struct less_equal : public binary_function<T,T,T>
    {
        bool operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs <= rhs;}
    };

    template<typename T>
    struct logical_and : public binary_function<T,T,T>
    {
        bool operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs && rhs;}
    };

    template<typename T>
    struct logical_or : public binary_function<T,T,T>
    {
        bool operator()(const T &lhs, const T &rhs) const restrict(cpu,amp) {return lhs || rhs;}
    };


};
};

#endif