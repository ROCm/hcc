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

#pragma once
#if !defined( BOLT_CL_FUNCTIONAL_H )
#define BOLT_CL_FUNCTIONAL_H

#include "bolt/cl/bolt.h"

/*! \file bolt/cl/functional.h
    \brief List all the unary and binary functions.
*/

namespace bolt {
namespace cl {

/******************************************************************************
 * Unary Operators
 *****************************************************************************/
static const std::string squareFunctor = BOLT_HOST_DEVICE_DEFINITION(
template< typename T >
struct square
{
    T operator() (const T& x)  const { return x * x; }
};
);

static const std::string cubeFunctor = BOLT_HOST_DEVICE_DEFINITION(
template< typename T >
struct cube
{
    T operator() (const T& x)  const { return x * x * x; }
};
);

static const std::string negateFunctor = BOLT_HOST_DEVICE_DEFINITION(
template< typename T >
struct negate 
{
    T operator()(const T& x) const {return -x;}
}; 
);

/******************************************************************************
 * Binary Operators
 *****************************************************************************/

static const std::string plusFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct plus
{
    T operator()(const T &lhs, const T &rhs) const {return lhs + rhs;}
}; 
);

static const std::string minusFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct minus
{
    T operator()(const T &lhs, const T &rhs) const {return lhs - rhs;}
}; 
);

static const std::string multipliesFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct multiplies
{
    T operator()(const T &lhs, const T &rhs) const {return lhs * rhs;}
}; 
);

static const std::string dividesFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct divides
{
    T operator()(const T &lhs, const T &rhs) const {return lhs / rhs;}
}; 
);

static const std::string modulusFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct modulus
{
    T operator()(const T &lhs, const T &rhs) const {return lhs % rhs;}
}; 
);

static const std::string maximumFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct maximum 
{
    T operator()(const T &lhs, const T &rhs) const  {return (lhs > rhs) ? lhs:rhs;}
}; 
);

static const std::string minimumFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct minimum
{
    T operator()(const T &lhs, const T &rhs) const  {return (lhs < rhs) ? lhs:rhs;}
}; 
);

static const std::string bit_andFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct bit_and
{
    T operator()(const T &lhs, const T &rhs) const  {return lhs & rhs;}
}; 
);

static const std::string bit_orFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct bit_or
{
    T operator()(const T &lhs, const T &rhs) const  {return lhs | rhs;}
}; 
);

static const std::string bit_xorFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct bit_xor
{
    T operator()(const T &lhs, const T &rhs) const  {return lhs ^ rhs;}
}; 
);


/******************************************************************************
 * Unary Predicates
 *****************************************************************************/

static const std::string logical_notFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct logical_not
{
    bool operator()(const T &x) const  {return !x;}
}; 
);

static const std::string identityFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct identity
{
    T operator()(const T &x) const  {return x;}
};
);


/******************************************************************************
 * Binary Predicates
 *****************************************************************************/

static const std::string equal_toFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct equal_to
{
    bool operator()(const T &lhs, const T &rhs) const  {return lhs == rhs;}
}; 
);

static const std::string not_equal_toFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct not_equal_to
{
    bool operator()(const T &lhs, const T &rhs) const  {return lhs != rhs;}
}; 
);

static const std::string greaterFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct greater
{
    bool operator()(const T &lhs, const T &rhs) const  {return lhs > rhs;}
}; 
);

static const std::string lessFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct less
{
    bool operator()(const T &lhs, const T &rhs) const  {return lhs < rhs;}
}; 
);

static const std::string greater_equalFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct greater_equal
{
    bool operator()(const T &lhs, const T &rhs) const  {return lhs >= rhs;}
}; 
);

static const std::string less_equalFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct less_equal
{
    bool operator()(const T &lhs, const T &rhs) const  {return lhs <= rhs;}
}; 
);

static const std::string logical_andFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct logical_and
{
    bool operator()(const T &lhs, const T &rhs) const  {return lhs && rhs;}
}; 
);

static const std::string logical_orFunctor = BOLT_HOST_DEVICE_DEFINITION(
template<typename T>
struct logical_or
{
    bool operator()(const T &lhs, const T &rhs) const  {return lhs || rhs;}
}; 
);

}; // namespace cl
}; // namespace bolt

BOLT_CREATE_TYPENAME( bolt::cl::square< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::square< cl_int >, bolt::cl::squareFunctor );


BOLT_CREATE_TYPENAME( bolt::cl::cube< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::cube< cl_int >, bolt::cl::cubeFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::negate< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::negate< cl_int >, bolt::cl::negateFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::plus< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::plus< cl_int >, bolt::cl::plusFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::minus< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::minus< cl_int >, bolt::cl::minusFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::multiplies< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::multiplies< cl_int >, bolt::cl::multipliesFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::divides< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::divides< cl_int >, bolt::cl::dividesFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::modulus< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::modulus< cl_int >, bolt::cl::modulusFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::maximum< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::maximum< cl_int >, bolt::cl::maximumFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::minimum< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::minimum< cl_int >, bolt::cl::minimumFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::bit_and< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::bit_and< cl_int >, bolt::cl::bit_andFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::bit_or< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::bit_or< cl_int >, bolt::cl::bit_orFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::bit_xor< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::bit_xor< cl_int >, bolt::cl::bit_xorFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::logical_not< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::logical_not< cl_int >, bolt::cl::logical_notFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::identity< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::identity< cl_int >, bolt::cl::identityFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::equal_to< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::equal_to< cl_int >, bolt::cl::equal_toFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::not_equal_to< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::not_equal_to< cl_int >, bolt::cl::not_equal_toFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::greater< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::greater< cl_int >, bolt::cl::greaterFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::less< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::less< cl_int >, bolt::cl::lessFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::greater_equal< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::greater_equal< cl_int >, bolt::cl::greater_equalFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::less_equal< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::less_equal< cl_int >, bolt::cl::less_equalFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::logical_and< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::logical_and< cl_int >, bolt::cl::logical_andFunctor );

BOLT_CREATE_TYPENAME( bolt::cl::logical_or< cl_int > );
BOLT_CREATE_CLCODE( bolt::cl::logical_or< cl_int >, bolt::cl::logical_orFunctor );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::square, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::square, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::square, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::square, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::square, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::square, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::square, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::cube, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::cube, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::cube, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::cube, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::cube, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::cube, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::cube, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::negate, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::negate, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::negate, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::negate, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::negate, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::negate, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::negate, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::plus, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::plus, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::plus, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::plus, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::plus, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::plus, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::plus, int, cl_double );


BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minus, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minus, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minus, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minus, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minus, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minus, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minus, int, cl_double );


BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::multiplies, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::multiplies, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::multiplies, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::multiplies, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::multiplies, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::multiplies, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::multiplies, int, cl_double );


BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::divides, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::divides, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::divides, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::divides, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::divides, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::divides, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::divides, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::modulus, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::modulus, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::modulus, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::modulus, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::modulus, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::modulus, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::modulus, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::maximum, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::maximum, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::maximum, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::maximum, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::maximum, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::maximum, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::maximum, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minimum, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minimum, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minimum, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minimum, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minimum, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minimum, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::minimum, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_and, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_and, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_and, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_and, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_and, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_and, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_and, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_or, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_or, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_or, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_or, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_or, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_or, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_or, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_xor, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_xor, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_xor, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_xor, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_xor, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_xor, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::bit_xor, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_not, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_not, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_not, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_not, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_not, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_not, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_not, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::identity, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::identity, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::identity, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::identity, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::identity, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::identity, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::identity, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::equal_to, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::equal_to, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::equal_to, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::equal_to, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::equal_to, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::equal_to, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::equal_to, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::not_equal_to, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::not_equal_to, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::not_equal_to, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::not_equal_to, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::not_equal_to, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::not_equal_to, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::not_equal_to, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater_equal, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater_equal, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater_equal, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater_equal, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater_equal, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater_equal, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::greater_equal, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less_equal, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less_equal, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less_equal, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less_equal, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less_equal, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less_equal, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::less_equal, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_and, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_and, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_and, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_and, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_and, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_and, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_and, int, cl_double );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_or, int, cl_short );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_or, int, cl_ushort );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_or, int, cl_uint );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_or, int, cl_long );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_or, int, cl_ulong );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_or, int, cl_float );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::logical_or, int, cl_double );

#endif
