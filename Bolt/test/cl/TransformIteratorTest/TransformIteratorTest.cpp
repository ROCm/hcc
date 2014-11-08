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
#pragma warning(disable:4244)

#include "common/stdafx.h"
#include <vector>
#include <array>
//#include "bolt/cl/iterator/counting_iterator.h"
#include "bolt/cl/iterator/transform_iterator.h"
#include "bolt/cl/copy.h"
#include "bolt/cl/transform_scan.h"
#include "bolt/cl/transform_reduce.h"
#include "bolt/cl/transform.h"
#include "bolt/cl/count.h"
#include "bolt/cl/reduce.h"
#include "bolt/cl/reduce_by_key.h"
#include "bolt/cl/generate.h"
#include "bolt/cl/inner_product.h"
#include "bolt/cl/scatter.h"
#include "bolt/cl/gather.h"
#include "bolt/cl/functional.h"
#include "bolt/cl/distance.h"
#include "bolt/miniDump.h"
#include "bolt/unicode.h"
#include "bolt/cl/scan.h"
#include "bolt/cl/scan_by_key.h"

#include <gtest/gtest.h>
#include "common/test_common.h"
#include <boost/program_options.hpp>
#define BCKND cl

namespace po = boost::program_options;


BOLT_FUNCTOR(square,
    struct square
    {
        int operator() (const int x)  const { return x + 2; }
        typedef int result_type;
    };
);

BOLT_FUNCTOR(add_4,
    struct add_4
    {
        int operator() (const int x)  const { return x + 4; }
        typedef int result_type;
    };
);

BOLT_FUNCTOR(add_3,
    struct add_3
    {
        int operator() (const int x)  const { return x + 3; }
        typedef int result_type;
    };
);
;

BOLT_FUNCTOR(add_0,
    struct add_0
    {
        int operator() (const int x)  const { return x; }
        typedef int result_type;
    };
);


int global_id = 0;

int get_global_id(int i)
{
    return global_id++;
}

BOLT_FUNCTOR(gen_input,
    struct gen_input
    {
        int operator() ()  const { return get_global_id(0); }
        typedef int result_type;
    };
);

BOLT_FUNCTOR(UDD, 
struct UDD
{
    int i;
    float f;
  
    bool operator == (const UDD& other) const {
        return ((i == other.i) && (f == other.f));
    }

	UDD operator ++ () 
    {
      UDD _result;
      _result.i = i + 1;
      _result.f = f + 1.0f;
      return _result;
    }

	UDD operator = (const int rhs) 
    {
      UDD _result;
      _result.i = i + rhs;
      _result.f = f + (float)rhs;
      return _result;
    }

	UDD operator + (const UDD &rhs) const
    {
      UDD _result;
      _result.i = this->i + rhs.i;
      _result.f = this->f + rhs.f;
      return _result;
    }

	UDD operator * (const UDD &rhs) const
    {
      UDD _result;
      _result.i = this->i * rhs.i;
      _result.f = this->f * rhs.f;
      return _result;
    }

	UDD operator + (const int rhs)
    {
      UDD _result;
      _result.i = i = i + rhs;
      _result.f = f = f + (float)rhs;
      return _result;
    }

	UDD operator-() const
    {
        UDD r;
        r.i = -i;
        r.f = -f;
        return r;
    }

    UDD()
        : i(0), f(0) { }
    UDD(int _in)
        : i(_in), f((float)(_in+2) ){ }
};
);

BOLT_FUNCTOR(UDDadd_3,
    struct UDDadd_3
    {
        UDD operator() (const UDD &x) const
		{ 
			UDD temp;
			temp.i = x.i + 3;
			temp.f = x.f + 3.0f;
			return temp; 
		}
        typedef UDD result_type;
    };
);

BOLT_FUNCTOR(squareUDD_result_float,
    struct squareUDD_result_float
    {
        float operator() (const UDD& x)  const 
        { 
            return ((float)x.i + x.f);
        }
        typedef float result_type;
    };
);

BOLT_FUNCTOR(squareUDD_result_int,
    struct squareUDD_result_int
    {
        int operator() (const UDD& x)  const 
        { 
            return (x.i + (int) x.f);
        }
        typedef int result_type;
    };
);

BOLT_FUNCTOR(squareUDD_resultUDD,
    struct squareUDD_resultUDD
    {
        UDD operator() (const UDD& x)  const 
        { 
            UDD tmp;
            tmp.i = x.i * x.i;
            tmp.f = x.f * x.f;
            return tmp;
        }
        typedef UDD result_type;
    };
);

BOLT_FUNCTOR(add3UDD_resultUDD,
    struct add3UDD_resultUDD
    {
        UDD operator() (const UDD& x)  const 
        { 
            UDD tmp;
            tmp.i = x.i + 3;
            tmp.f = x.f + 3.f;
            return tmp;
        }
        typedef UDD result_type;
    };
);

BOLT_FUNCTOR(add4UDD_resultUDD,
    struct add4UDD_resultUDD
    {
        UDD operator() (const UDD& x)  const 
        { 
            UDD tmp;
            tmp.i = x.i + 4;
            tmp.f = x.f + 4.f;
            return tmp;
        }
        typedef UDD result_type;
    };
);

BOLT_FUNCTOR(cubeUDD,
    struct cubeUDD
    {
        float operator() (const UDD& x)  const 
        { 
            return ((float)x.i + x.f + 3.0f);
        }
        typedef float result_type;
    };
);

BOLT_FUNCTOR(cubeUDD_result_int,
    struct cubeUDD_result_int
    {
        float operator() (const UDD& x)  const 
        { 
            return (x.i + (int)x.f + 3);
        }
        typedef int result_type;
    };
);

BOLT_FUNCTOR(cubeUDD_resultUDD,
    struct cubeUDD_resultUDD
    {
        UDD operator() (const UDD& x)  const 
        { 
            UDD tmp;
            tmp.i = x.i * x.i * x.i;
            tmp.f = x.f * x.f * x.f;
            return tmp;
        }
        typedef UDD result_type;
    };
);


BOLT_FUNCTOR( UDDplus,
struct UDDplus
{
   UDD operator() (const UDD &lhs, const UDD &rhs) const
   {
     UDD _result;
     _result.i = lhs.i + rhs.i;
     _result.f = lhs.f + rhs.f;
     return _result;
   }

};
);

BOLT_FUNCTOR( UDDnegate,
struct UDDnegate
{
   UDD operator() (const UDD &lhs) const
   {
     UDD _result;
     _result.i = -lhs.i;
     _result.f = -lhs.f;
     return _result;
   }

};
);

BOLT_FUNCTOR( UDDmul,
struct UDDmul
{
   UDD operator() (const UDD &lhs, const UDD &rhs) const
   {

     return lhs*rhs;
   }

};
);

BOLT_FUNCTOR( UDDminus,
struct UDDminus
{
   UDD operator() (const UDD &lhs, const UDD &rhs) const
   {
     UDD _result;
     _result.i = lhs.i - rhs.i;
     _result.f = lhs.f - rhs.f;
     return _result;
   }

};
);

/*Create Device Vector Iterators*/
BOLT_TEMPLATE_REGISTER_NEW_ITERATOR( bolt::cl::device_vector, int, UDD);

/*Create Transform iterators*/
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, square, UDD);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, add_3, int);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, add_4, int);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, add_0, int);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, UDDadd_3, UDD);

BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, squareUDD_result_int, UDD);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, cubeUDD_result_int, UDD);

BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, squareUDD_result_float, UDD);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, squareUDD_resultUDD, UDD);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, cubeUDD_resultUDD, UDD);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, cubeUDD, UDD);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, add3UDD_resultUDD, UDD);
BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, add4UDD_resultUDD, UDD);


BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::negate, float, UDD );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::constant_iterator, int, UDD );
BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::counting_iterator, int, UDD );
BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::plus, float, UDD );
BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::multiplies, float, UDD );
BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::equal_to, float, UDD );

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::detail::CountIfEqual, int, UDD );


BOLT_FUNCTOR(gen_input_udd,
    struct gen_input_udd
    {
        UDD operator() ()  const 
       { 
            int i=get_global_id(0);
            UDD temp;
            temp.i = i;
            temp.f = (float)i;
            return temp; 
        }
        typedef UDD result_type;
    };
);

BOLT_FUNCTOR(gen_input_udd2,
    struct gen_input_udd2
    {
        UDD operator() ()  const 
       { 
            int i=get_global_id(0);
            UDD temp;
            temp.i = i*2;
            temp.f = (float)i*2;
            return temp; 
        }
        typedef UDD result_type;
    };
);


BOLT_FUNCTOR( is_even,				  
struct is_even{
    bool operator () (int x)
    {
        return ( (x % 2)==0);
    }
};
);

template<
    typename oType,
    typename BinaryFunction,
    typename T>
oType*
Serial_scan(
    oType *values,
    oType *result,
    unsigned int  num,
    const BinaryFunction binary_op,
    const bool Incl,
    const T &init)
{
    oType  sum, temp;
    if(Incl){
      *result = *values; // assign value
      sum = *values;
    }
    else {
        temp = *values;
       *result = (oType)init;
       sum = binary_op( *result, temp);
    }
    for ( unsigned int i= 1; i<num; i++)
    {
        oType currentValue = *(values + i); // convertible
        if (Incl)
        {
            oType r = binary_op( sum, currentValue);
            *(result + i) = r;
            sum = r;
        }
        else // new segment
        {
            *(result + i) = sum;
            sum = binary_op( sum, currentValue);

        }
    }
    return result;
}

template<
    typename kType,
    typename vType,
    typename koType,
    typename voType,
    typename BinaryFunction>
unsigned int
Serial_reduce_by_key(
	kType* keys,
	vType* values,
	koType* keys_output,
	voType* values_output,
	BinaryFunction binary_op,
	unsigned int  numElements
	)
{
 
    static_assert( std::is_convertible< vType, voType >::value,
                   "InputIterator2 and OutputIterator's value types are not convertible." );

    // do zeroeth element
    values_output[0] = values[0];
    keys_output[0] = keys[0];
    unsigned int count = 1;
    // rbk oneth element and beyond

    unsigned int vi=1, vo=0, ko=0;
    for ( unsigned int i= 1; i<numElements; i++)
    {
        // load keys
        kType currentKey  = keys[i];
        kType previousKey = keys[i-1];

        // load value
        voType currentValue = values[vi];
        voType previousValue = values_output[vo];

        previousValue = values_output[vo];
        // within segment
        if (currentKey == previousKey)
        {
            voType r = binary_op( previousValue, currentValue);
            values_output[vo] = r;
            keys_output[ko] = currentKey;

        }
        else // new segment
        {
            vo++;
            ko++;
            values_output[vo] = currentValue;
            keys_output[ko] = currentKey;
            count++; //To count the number of elements in the output array
        }
        vi++;
    }

    return count;

}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>

void Serial_scatter (InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 map,
                     OutputIterator result)
{
       size_t numElements = static_cast<  size_t >( std::distance( first1, last1 ) );

	   for (int iter = 0; iter<(int)numElements; iter++)
                *(result+*(map + iter)) = *(first1 + iter);
}

template<typename InputIterator1,
         typename InputIterator2,
		 typename InputIterator3,
         typename OutputIterator,
         typename Predicate>

void Serial_scatter_if (InputIterator1 first1,
                     InputIterator1 last1,
                     InputIterator2 map,
					 InputIterator3 stencil,
                     OutputIterator result,
					 Predicate pred)
{
       size_t numElements = static_cast< size_t >( std::distance( first1, last1 ) );
	   for (int iter = 0; iter< (int)numElements; iter++)
       {
             if(pred(stencil[iter]) != 0)
                  result[*(map+(iter))] = first1[iter];
       }
}

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator>

void Serial_gather (InputIterator1 map_first,
                     InputIterator1 map_last,
                     InputIterator2 input,
                     OutputIterator result)
{
       int numElements = static_cast< int >( std::distance( map_first, map_last ) );
       typedef typename  std::iterator_traits<InputIterator1>::value_type iType1;
       iType1 temp;
       for(int iter = 0; iter < numElements; iter++)
       {
              temp = *(map_first + (int)iter);
              *(result + (int)iter) = *(input + (int)temp);
       }
}

template<typename InputIterator1,
         typename InputIterator2,
		 typename InputIterator3,
         typename OutputIterator,
         typename Predicate>

void Serial_gather_if (InputIterator1 map_first,
                     InputIterator1 map_last,
                     InputIterator2 stencil,
					 InputIterator3 input,
                     OutputIterator result,
					 Predicate pred)
{
    unsigned int numElements = static_cast< unsigned int >( std::distance( map_first, map_last ) );
    for(unsigned int iter = 0; iter < numElements; iter++)
    {
        if(pred(*(stencil + (int)iter)))
             result[(int)iter] = input[map_first[(int)iter]];
    }
}


TEST( TransformIterator, FirstTest)
{
    {
        const int length = 1<<10;
        std::vector< int > svInVec( length );
        std::vector< int > svOutVec( length );
        //bolt::BCKND::device_vector< int > dvInVec( length );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        square sq;
        gen_input gen;
        typedef std::vector< int >::const_iterator                                                         sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                                dv_itr;
        typedef bolt::BCKND::transform_iterator< square, std::vector< int >::const_iterator>               sv_trf_itr;
        typedef bolt::BCKND::transform_iterator< square, bolt::BCKND::device_vector< int >::iterator>      dv_trf_itr;
    
        /*Create Iterators*/
        sv_trf_itr sv_trf_begin (svInVec.begin(), sq), sv_trf_end (svInVec.end(), sq);
        //dv_trf_itr dv_trf_begin (dvInVec.begin(), sq), dv_trf_end (dvInVec.end(), sq);
    
        /*Generate inputs*/
        std::generate(svInVec.begin(), svInVec.end(), gen);    
        //bolt::BCKND::generate(dvInVec.begin(), dvInVec.end(), gen);

		bolt::BCKND::device_vector< int > dvInVec(svInVec.begin(), svInVec.end());
		dv_trf_itr dv_trf_begin (dvInVec.begin(), sq), dv_trf_end (dvInVec.end(), sq);

        sv_trf_itr::difference_type dist1 = bolt::cl::distance(sv_trf_begin, sv_trf_end);
        sv_trf_itr::difference_type dist2 = bolt::cl::distance(dv_trf_begin, dv_trf_end );

        EXPECT_EQ( dist1, dist2 );
        //std::cout << "distance = " << dist1 << "\n" ;

        for(int i =0; i< length; i++)
        {
            int temp1, temp2;
            temp1 = *sv_trf_begin++;
            temp2 = *dv_trf_begin++;
            EXPECT_EQ( temp1, temp2 );
            //std::cout << temp1 << "   " << temp2 << "\n";
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, UDDTest)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svInVec( length );
        std::vector< UDD > svOutVec( length );
        //bolt::BCKND::device_vector< UDD > dvInVec( length );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );

        squareUDD_result_float sqUDD;
        gen_input_udd genUDD;
        /*Type defintions*/
        typedef std::vector< UDD >::const_iterator                                                          sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                                 dv_itr;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_float, std::vector< UDD >::const_iterator>             sv_trf_itr;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_float, bolt::BCKND::device_vector< UDD >::iterator>    dv_trf_itr;
    
        /*Create Iterators*/
        sv_trf_itr sv_trf_begin (svInVec.begin(), sqUDD), sv_trf_end (svInVec.begin(), sqUDD);

        //dv_trf_itr dv_trf_begin (dvInVec.begin(), sqUDD), dv_trf_end (dvInVec.begin(), sqUDD);
 
        /*Generate inputs*/
        std::generate(svInVec.begin(), svInVec.end(), genUDD);
        //bolt::BCKND::generate(dvInVec.begin(), dvInVec.end(), genUDD);
		bolt::BCKND::device_vector< UDD > dvInVec(svInVec.begin(), svInVec.end());
		dv_trf_itr dv_trf_begin (dvInVec.begin(), sqUDD), dv_trf_end (dvInVec.begin(), sqUDD);

        int dist1 = static_cast< int >(std::distance(sv_trf_begin, sv_trf_end));
        int dist2 = static_cast< int >(std::distance(dv_trf_begin, dv_trf_end));

        EXPECT_EQ( dist1, dist2 );
        //std::cout << "distance = " << dist1 << "\n" ;

        for(int i =0; i< length; i++)
        {
            float temp1, temp2; //Return type of the unary function is a float
            temp1 = (float)*sv_trf_begin++;
            temp2 = (float)*dv_trf_begin++;
            EXPECT_FLOAT_EQ( temp1, temp2 );
            //std::cout << temp1 << "   " << temp2 << "\n";
        }
        global_id = 0; // Reset the global id counter
    }
    
}

TEST( TransformIterator, UnaryTransformRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );
        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;

        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);

        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;


        {/*Test case when inputs are trf Iterators*/
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, svOutVec.begin(), add3);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, dvOutVec.begin(), add3);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, stlOut.begin(), add3);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the both are randomAccessIterator */
            bolt::cl::transform(svIn1Vec.begin(), svIn1Vec.end(), svOutVec.begin(), add3);
            bolt::cl::transform(dvIn1Vec.begin(), dvIn1Vec.end(), dvOutVec.begin(), add3);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), add3);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::transform(const_itr_begin, const_itr_end, svOutVec.begin(), add3);
            bolt::cl::transform(const_itr_begin, const_itr_end, dvOutVec.begin(), add3);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::transform(const_vector.begin(), const_vector.end(), stlOut.begin(), add3);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::transform(count_itr_begin, count_itr_end, svOutVec.begin(), add3);
            bolt::cl::transform(count_itr_begin, count_itr_end, dvOutVec.begin(), add3);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            std::transform(count_vector.begin(), count_vector.end(), stlOut.begin(), add3);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, UnaryTransformUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );
        std::vector< UDD > svOutVec( length );
        std::vector< UDD > stlOut( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );

		std::vector< float > stlOut_float( length );
		std::vector< float > svOutVec_float( length );
		bolt::BCKND::device_vector< float > dvOutVec_float( length );

        squareUDD_resultUDD sqUDD;
		squareUDD_result_float sqUDD_float;
		squareUDD_result_int sq_int;

        typedef std::vector< UDD>::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD>                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< squareUDD_resultUDD, std::vector< UDD >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>   dv_trf_itr_add3;
     
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), sqUDD), sv_trf_end1 (svIn1Vec.end(), sqUDD);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), sqUDD), dv_trf_end1 (dvIn1Vec.end(), sqUDD);

		UDD temp;
		temp.i=1, temp.f=2.5f;

		UDD init;
		init.i=0, init.f=0.0f;

        counting_itr count_itr_begin(init);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;

		 {/*Test case when input is trf Iterator and UDD is returning int*/
		    typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>            tsv_trf_itr_add3;
            typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator>   tdv_trf_itr_add3;
            std::vector< int >                  tsvOutVec( length );
            std::vector< int >                  tstlOut( length );
            bolt::BCKND::device_vector< int >   tdvOutVec( length );

		    tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
            tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);

            bolt::cl::transform(tsv_trf_begin1, tsv_trf_end1, tsvOutVec.begin(), bolt::cl::negate<int>());
            bolt::cl::transform(tdv_trf_begin1, tdv_trf_end1, tdvOutVec.begin(), bolt::cl::negate<int>());
            /*Compute expected results*/
            std::transform(tsv_trf_begin1, tsv_trf_end1, tstlOut.begin(), bolt::cl::negate<int>());
            /*Check the results*/
            cmpArrays(tsvOutVec, tstlOut, length);
            cmpArrays(tdvOutVec, tstlOut, length);
        }

        {/*Test case when input is trf Iterator*/
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, svOutVec.begin(), sqUDD);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, dvOutVec.begin(), sqUDD);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, stlOut.begin(), sqUDD);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		 {/*Test case when input is trf Iterator and Output is float vector*/
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, svOutVec_float.begin(), sqUDD_float);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, dvOutVec_float.begin(), sqUDD_float);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, stlOut_float.begin(), sqUDD_float);
            /*Check the results*/
            cmpArrays(svOutVec_float, stlOut_float, length);
            cmpArrays(dvOutVec_float, stlOut_float, length);
        }


        {/*Test case when the input is randomAccessIterator */
            bolt::cl::transform(svIn1Vec.begin(), svIn1Vec.end(), svOutVec.begin(), sqUDD);
            bolt::cl::transform(dvIn1Vec.begin(), dvIn1Vec.end(), dvOutVec.begin(), sqUDD);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), sqUDD);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the input is constant iterator  */
            bolt::cl::transform(const_itr_begin, const_itr_end, svOutVec.begin(), sqUDD);
            bolt::cl::transform(const_itr_begin, const_itr_end, dvOutVec.begin(), sqUDD);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,temp);
            std::transform(const_vector.begin(), const_vector.end(), stlOut.begin(), sqUDD);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the input is a counting iterator */
            bolt::cl::transform(count_itr_begin, count_itr_end, svOutVec.begin(), sqUDD);
            bolt::cl::transform(count_itr_begin, count_itr_end, dvOutVec.begin(), sqUDD);
            /*Compute expected results*/
            std::transform(count_itr_begin, count_itr_end, stlOut.begin(), sqUDD);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, BinaryTransformRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svIn2Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;
        add_4 add4;
        bolt::cl::plus<int> plus;

        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add4);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add4);
        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;


        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec.begin(), plus);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svOutVec.begin(), plus);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the second input is trf_itr and the first is a randomAccessIterator */
            bolt::cl::transform(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svOutVec.begin(), plus);
            bolt::cl::transform(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the both are randomAccessIterator */
            bolt::cl::transform(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svOutVec.begin(), plus);
            bolt::cl::transform(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is trf_itr and the second is a constant iterator */
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, const_itr_begin, svOutVec.begin(), plus);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, const_itr_begin, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::transform(sv_trf_begin1, sv_trf_end1, const_vector.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, count_itr_begin, svOutVec.begin(), plus);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, count_itr_begin, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            std::transform(sv_trf_begin1, sv_trf_end1, count_vector.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::transform(const_itr_begin, const_itr_end, count_itr_begin, svOutVec.begin(), plus);
            bolt::cl::transform(const_itr_begin, const_itr_end, count_itr_begin, dvOutVec.begin(), plus);
            /*Compute expected results*/
           std::vector<int> const_vector(length,1);
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            std::transform(const_vector.begin(), const_vector.end(), count_vector.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, BinaryTransformUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );
        std::vector< UDD > svIn2Vec( length );
        std::vector< UDD > svOutVec( length );
        std::vector< int > tsvOutVec( length );
        std::vector< UDD > stlOut( length );
        std::vector< int > tstlOut( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), genUDD);
        global_id = 0;
        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< UDD > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );
        bolt::BCKND::device_vector< int > tdvOutVec( length );

        bolt::cl::plus<UDD> plus;

		cubeUDD_resultUDD cbUDD;
		squareUDD_resultUDD sqUDD;

		squareUDD_result_int sq_int;
		cubeUDD_result_int cb_int;

		typedef std::vector< UDD>::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< squareUDD_resultUDD, std::vector< UDD >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< cubeUDD_resultUDD, std::vector< UDD >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< cubeUDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>   dv_trf_itr_add4;    

		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>            tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator>   tdv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< cubeUDD_result_int, std::vector< UDD >::const_iterator>            tsv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< cubeUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator>   tdv_trf_itr_add4;    


        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), sqUDD), sv_trf_end1 (svIn1Vec.end(), sqUDD);
		sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), cbUDD);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), sqUDD), dv_trf_end1 (dvIn1Vec.end(), sqUDD);
		dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), cbUDD);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
		tsv_trf_itr_add4 tsv_trf_begin2 (svIn2Vec.begin(), cb_int);
        tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);
		tdv_trf_itr_add4 tdv_trf_begin2 (dvIn2Vec.begin(), cb_int);

		UDD temp;
		temp.i=1, temp.f=2.5f;

		UDD init;
		init.i=0, init.f=0.0f;

        counting_itr count_itr_begin(init);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;

       {/*Test case when both inputs are trf Iterators and Return type of UDD is int*/
            bolt::cl::plus<int> plus_int;
            bolt::cl::transform(tsv_trf_begin1, tsv_trf_end1, tsv_trf_begin2, tsvOutVec.begin(), plus_int);
            bolt::cl::transform(tdv_trf_begin1, tdv_trf_end1, tdv_trf_begin2, tdvOutVec.begin(), plus_int);
            /*Compute expected results*/
            std::transform(tsv_trf_begin1, tsv_trf_end1, tsv_trf_begin2, tstlOut.begin(), plus_int);
            /*Check the results*/
            cmpArrays(tsvOutVec, tstlOut, length);
            cmpArrays(tdvOutVec, tstlOut, length);
        }


        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec.begin(), plus);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svOutVec.begin(), plus);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the second input is trf_itr and the first is a randomAccessIterator */
            bolt::cl::transform(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svOutVec.begin(), plus);
            bolt::cl::transform(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the both are randomAccessIterator */
            bolt::cl::transform(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svOutVec.begin(), plus);
            bolt::cl::transform(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is trf_itr and the second is a constant iterator */
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, const_itr_begin, svOutVec.begin(), plus);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, const_itr_begin, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,temp);
            std::transform(sv_trf_begin1, sv_trf_end1, const_vector.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is counting iterator and the second is a trf_itr */
            bolt::cl::transform( count_itr_begin, count_itr_end, sv_trf_begin1, svOutVec.begin(), plus);
            bolt::cl::transform( count_itr_begin, count_itr_end, dv_trf_begin1, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::transform(count_itr_begin, count_itr_end, sv_trf_begin1, stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
		 {/*Test case when the first input is counting iterator and the second is a constant iterator */
            bolt::cl::transform(count_itr_begin, count_itr_end, const_itr_begin, svOutVec.begin(), plus);
            bolt::cl::transform(count_itr_begin, count_itr_end, const_itr_begin, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,temp);        
            std::transform(count_itr_begin, count_itr_end, const_vector.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::transform(sv_trf_begin1, sv_trf_end1, count_itr_begin, svOutVec.begin(), plus);
            bolt::cl::transform(dv_trf_begin1, dv_trf_end1, count_itr_begin, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::vector<UDD> count_vector(count_itr_begin, count_itr_end);    
            std::transform(sv_trf_begin1, sv_trf_end1, count_vector.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
		
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::transform(const_itr_begin, const_itr_end, count_itr_begin, svOutVec.begin(), plus);
            bolt::cl::transform(const_itr_begin, const_itr_end, count_itr_begin, dvOutVec.begin(), plus);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,temp);
            std::vector<UDD> count_vector(count_itr_begin, count_itr_end); 
            std::transform(const_vector.begin(), const_vector.end(), count_vector.begin(), stlOut.begin(), plus);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}


TEST( TransformIterator, InclusiveTransformScanRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;

        bolt::cl::negate<int> nI2;
        bolt::cl::plus<int> addI2;

        typedef std::vector< int >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator> dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>          sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator> dv_trf_itr_add4;    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;

        {/*Test case when inputs are trf Iterators*/
            bolt::cl::transform_inclusive_scan(sv_trf_begin1, sv_trf_end1, svOutVec.begin(), nI2, addI2);
            bolt::cl::transform_inclusive_scan(dv_trf_begin1, dv_trf_end1, dvOutVec.begin(), nI2, addI2);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, stlOut.begin(), nI2);
            std::partial_sum(stlOut.begin(), stlOut.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the both are randomAccessIterator */
            bolt::cl::transform_inclusive_scan(svIn1Vec.begin(), svIn1Vec.end(), svOutVec.begin(), nI2, addI2);
            bolt::cl::transform_inclusive_scan(dvIn1Vec.begin(), dvIn1Vec.end(), dvOutVec.begin(), nI2, addI2);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), nI2);
            std::partial_sum(stlOut.begin(), stlOut.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::transform_inclusive_scan(const_itr_begin, const_itr_end, svOutVec.begin(), nI2, addI2);
            bolt::cl::transform_inclusive_scan(const_itr_begin, const_itr_end, dvOutVec.begin(), nI2, addI2);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::transform(const_vector.begin(), const_vector.end(), stlOut.begin(), nI2);
            std::partial_sum(stlOut.begin(), stlOut.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::transform_inclusive_scan(count_itr_begin, count_itr_end, svOutVec.begin(), nI2, addI2);
            bolt::cl::transform_inclusive_scan(count_itr_begin, count_itr_end, dvOutVec.begin(), nI2, addI2);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            std::transform(count_vector.begin(), count_vector.end(), stlOut.begin(), nI2);
            std::partial_sum(stlOut.begin(), stlOut.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, InclusiveTransformScanUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );
        std::vector< UDD > svOutVec( length );
        std::vector< UDD > stlOut( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );

        bolt::cl::negate<UDD> nI2;
        bolt::cl::plus<UDD> addI2;

		add3UDD_resultUDD sqUDD;
		squareUDD_result_int sq_int;

        typedef std::vector< UDD >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator> dv_trf_itr_add3;

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), sqUDD), sv_trf_end1 (svIn1Vec.end(), sqUDD);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), sqUDD), dv_trf_end1 (dvIn1Vec.end(), sqUDD);

        UDD temp;
		temp.i=1, temp.f=2.5f;
		UDD t;
		t.i=0, t.f=0.0f;

        counting_itr count_itr_begin(t);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;

        {/*Test case when input is trf Iterator and return type of UDD is int*/
		    typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>          tsv_trf_itr_add3;
            typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator> tdv_trf_itr_add3;
		    tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
            tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);
            bolt::BCKND::device_vector< int > tdvOutVec( length );
            std::vector< int > tstlOut( length );
            std::vector< int > tsvOutVec( length );

		    bolt::cl::negate<int> nI2_int;
            bolt::cl::plus<int> addI2_int;

            bolt::cl::transform_inclusive_scan(tsv_trf_begin1, tsv_trf_end1, tsvOutVec.begin(), nI2_int, addI2_int);
            bolt::cl::transform_inclusive_scan(tdv_trf_begin1, tdv_trf_end1, tdvOutVec.begin(), nI2_int, addI2_int);
            /*Compute expected results*/
            std::transform(tsv_trf_begin1, tsv_trf_end1, tstlOut.begin(), nI2_int);
            std::partial_sum(tstlOut.begin(), tstlOut.end(), tstlOut.begin(), addI2_int);
            /*Check the results*/
            cmpArrays(tsvOutVec, tstlOut, length);
            cmpArrays(tdvOutVec, tstlOut, length);
        }

        {/*Test case when input is trf Iterator*/
            bolt::cl::transform_inclusive_scan(sv_trf_begin1, sv_trf_end1, svOutVec.begin(), nI2, addI2);
            bolt::cl::transform_inclusive_scan(dv_trf_begin1, dv_trf_end1, dvOutVec.begin(), nI2, addI2);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, stlOut.begin(), nI2);
            std::partial_sum(stlOut.begin(), stlOut.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the input is randomAccessIterator */
            bolt::cl::transform_inclusive_scan(svIn1Vec.begin(), svIn1Vec.end(), svOutVec.begin(), nI2, addI2);
            bolt::cl::transform_inclusive_scan(dvIn1Vec.begin(), dvIn1Vec.end(), dvOutVec.begin(), nI2, addI2);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), nI2);
            std::partial_sum(stlOut.begin(), stlOut.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the input is constant iterator */
            bolt::cl::transform_inclusive_scan(const_itr_begin, const_itr_end, svOutVec.begin(), nI2, addI2);
            bolt::cl::transform_inclusive_scan(const_itr_begin, const_itr_end, dvOutVec.begin(), nI2, addI2);
            /*Compute expected results*/
            std::vector<UDD> const_vector(const_itr_begin, const_itr_end);
            std::transform(const_vector.begin(), const_vector.end(), stlOut.begin(), nI2);
            std::partial_sum(stlOut.begin(), stlOut.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the input is a counting iterator */
            bolt::cl::transform_inclusive_scan(count_itr_begin, count_itr_end, svOutVec.begin(), nI2, addI2);
            bolt::cl::transform_inclusive_scan(count_itr_begin, count_itr_end, dvOutVec.begin(), nI2, addI2);
            /*Compute expected results*/  
            std::transform(count_itr_begin, count_itr_end, stlOut.begin(), nI2);
            std::partial_sum(stlOut.begin(), stlOut.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, ExclusiveTransformScanRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;


        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;

        bolt::cl::negate<int> nI2;
        bolt::cl::plus<int> addI2;
        int n = (int) 1 + rand()%10;


        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);

        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;


        {/*Test case when inputs are trf Iterators*/
            bolt::cl::transform_exclusive_scan(sv_trf_begin1, sv_trf_end1, svOutVec.begin(), nI2, n, addI2);
            bolt::cl::transform_exclusive_scan(dv_trf_begin1, dv_trf_end1, dvOutVec.begin(), nI2, n, addI2);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, stlOut.begin(), nI2);
            Serial_scan<int,  bolt::cl::plus< int >, int>(&stlOut[0], &stlOut[0], length, addI2, false, n);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the both are randomAccessIterator */
            bolt::cl::transform_exclusive_scan(svIn1Vec.begin(), svIn1Vec.end(), svOutVec.begin(), nI2, n, addI2);
            bolt::cl::transform_exclusive_scan(dvIn1Vec.begin(), dvIn1Vec.end(), dvOutVec.begin(), nI2, n, addI2);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), nI2);
            Serial_scan<int,  bolt::cl::plus< int >, int>(&stlOut[0], &stlOut[0], length, addI2, false, n);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::transform_exclusive_scan(const_itr_begin, const_itr_end, svOutVec.begin(), nI2, n, addI2);
            bolt::cl::transform_exclusive_scan(const_itr_begin, const_itr_end, dvOutVec.begin(), nI2, n, addI2);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::transform(const_vector.begin(), const_vector.end(), stlOut.begin(), nI2);
            Serial_scan<int,  bolt::cl::plus< int >, int>(&stlOut[0], &stlOut[0], length, addI2, false, n);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::transform_exclusive_scan(count_itr_begin, count_itr_end, svOutVec.begin(), nI2, n, addI2);
            bolt::cl::transform_exclusive_scan(count_itr_begin, count_itr_end, dvOutVec.begin(), nI2, n, addI2);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            std::transform(count_vector.begin(), count_vector.end(), stlOut.begin(), nI2);
            Serial_scan<int,  bolt::cl::plus< int >, int>(&stlOut[0], &stlOut[0], length, addI2, false, n);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, ExclusiveTransformScanUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );
        std::vector< UDD > svOutVec( length );
        std::vector< UDD > stlOut( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;


        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );

		bolt::cl::negate<UDD> nI2;
        bolt::cl::plus<UDD> addI2;

		add3UDD_resultUDD sqUDD;


		squareUDD_result_int sq_int;
		UDD n;
	    n.i = (int) 1 + rand()%10;
		n.f = (float) rand();

        typedef std::vector< UDD >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator> dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>          tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator> tdv_trf_itr_add3;

    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), sqUDD), sv_trf_end1 (svIn1Vec.end(), sqUDD);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), sqUDD), dv_trf_end1 (dvIn1Vec.end(), sqUDD);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
        tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);

        UDD temp;
		temp.i=1, temp.f=2.5f;

		UDD t;
		t.i=0, t.f=0.0f;

        counting_itr count_itr_begin(t);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;


		{/*Test case when input is trf Iterator and return type of UDD is int*/
		    int nint = rand()%10;            
		    typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>          tsv_trf_itr_add3;
            typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator> tdv_trf_itr_add3;
		    tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
            tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);
            bolt::BCKND::device_vector< int > tdvOutVec( length );
            std::vector< int > tstlOut( length );
            std::vector< int > tsvOutVec( length );

		    bolt::cl::negate<int> nI2_int;
            bolt::cl::plus<int> addI2_int;

            bolt::cl::transform_exclusive_scan(tsv_trf_begin1, tsv_trf_end1, tsvOutVec.begin(), nI2_int, nint, addI2_int);
            bolt::cl::transform_exclusive_scan(tdv_trf_begin1, tdv_trf_end1, tdvOutVec.begin(), nI2_int, nint, addI2_int);
            /*Compute expected results*/
            std::transform(tsv_trf_begin1, tsv_trf_end1, tstlOut.begin(), nI2_int);
            Serial_scan<int,  bolt::cl::plus< int >, int>(&tstlOut[0], &tstlOut[0], length, addI2_int, false, nint);
            /*Check the results*/
            cmpArrays(tsvOutVec, tstlOut, length);
            cmpArrays(tdvOutVec, tstlOut, length);
        }

        {/*Test case when input is a randomAccessIterator */
            bolt::cl::transform_exclusive_scan(svIn1Vec.begin(), svIn1Vec.end(), svOutVec.begin(), nI2, n, addI2);
            bolt::cl::transform_exclusive_scan(dvIn1Vec.begin(), dvIn1Vec.end(), dvOutVec.begin(), nI2, n, addI2);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), nI2);
            Serial_scan<UDD,  bolt::cl::plus< UDD >, UDD>(&stlOut[0], &stlOut[0], length, addI2, false, n);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when first input is a constant iterator */
            bolt::cl::transform_exclusive_scan(const_itr_begin, const_itr_end, svOutVec.begin(), nI2, n, addI2);
            bolt::cl::transform_exclusive_scan(const_itr_begin, const_itr_end, dvOutVec.begin(), nI2, n, addI2);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,temp);
            std::transform(const_vector.begin(), const_vector.end(), stlOut.begin(), nI2);
            Serial_scan<UDD,  bolt::cl::plus< UDD >, UDD>(&stlOut[0], &stlOut[0], length, addI2, false, n);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the input is a counting iterator */
            bolt::cl::transform_exclusive_scan(count_itr_begin, count_itr_end, svOutVec.begin(), nI2, n, addI2);
            bolt::cl::transform_exclusive_scan(count_itr_begin, count_itr_end, dvOutVec.begin(), nI2, n, addI2);
            /*Compute expected results*/
            std::transform(count_itr_begin,  count_itr_end, stlOut.begin(), nI2);
            Serial_scan<UDD,  bolt::cl::plus< UDD >, UDD>(&stlOut[0], &stlOut[0], length, addI2, false, n);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}


TEST( TransformIterator, ReduceRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svIn2Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;
        add_4 add4;
        bolt::cl::plus<int> plus;
        typedef std::vector< int >::const_iterator                                                         sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                                dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                      counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                      constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>                sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>       dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>                sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator>       dv_trf_itr_add4;    
        
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add4);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add4);
        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;

        {/*Test case when inputs are trf Iterators*/
            int sv_result = bolt::cl::reduce(sv_trf_begin1, sv_trf_end1, 0, plus);
            int dv_result = bolt::cl::reduce(dv_trf_begin1, dv_trf_end1, 0, plus);
            /*Compute expected results*/
            int expected_result = std::accumulate(sv_trf_begin1, sv_trf_end1, 0, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            int sv_result = bolt::cl::reduce(svIn2Vec.begin(), svIn2Vec.end(), 0, plus);
            int dv_result = bolt::cl::reduce(dvIn2Vec.begin(), dvIn2Vec.end(), 0, plus);
            /*Compute expected results*/
            int expected_result = std::accumulate(svIn2Vec.begin(), svIn2Vec.end(), 0, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the first input is trf_itr and the second is a constant iterator */
            int sv_result = bolt::cl::reduce(const_itr_begin, const_itr_end, 0, plus);
            int dv_result = bolt::cl::reduce(const_itr_begin, const_itr_end, 0, plus);
            /*Compute expected results*/
            int expected_result = std::accumulate(const_itr_begin, const_itr_end, 0, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the first input is trf_itr and the second is a counting iterator */
            int sv_result = bolt::cl::reduce(count_itr_begin, count_itr_end, 0, plus);
            int dv_result = bolt::cl::reduce(count_itr_begin, count_itr_end, 0, plus);
            /*Compute expected results*/
            int expected_result = std::accumulate(count_itr_begin, count_itr_end, 0, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, ReduceUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );
        std::vector< UDD > svIn2Vec( length );
        std::vector< UDD > svOutVec( length );
        std::vector< UDD > stlOut( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
		global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), genUDD);
        global_id = 0;

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< UDD > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );

        UDDplus plus;
		UDDmul mul;
		add3UDD_resultUDD sqUDD;
		add4UDD_resultUDD cbUDD;

		squareUDD_result_int sq_int;
		bolt::cl::plus<int> plus_int;

        typedef std::vector< UDD >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator> dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add4UDD_resultUDD, std::vector< UDD >::const_iterator>                sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add4UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>       dv_trf_itr_add4;   

		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>          tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator> tdv_trf_itr_add3;


        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), sqUDD), sv_trf_end1 (svIn1Vec.end(), sqUDD);
		sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), cbUDD);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), sqUDD), dv_trf_end1 (dvIn1Vec.end(), sqUDD);
		dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), cbUDD);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
        tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);


        UDD temp;
		temp.i=1, temp.f=2.5f;

		UDD init;
		init.i=0, init.f=0.0f;


        counting_itr count_itr_begin(init);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;



		UDD UDDzero;
        UDDzero.i = 0;
        UDDzero.f = 0.0f;

		UDD UDDone;
        UDDone.i = 1;
        UDDone.f = 1.0f;

		 {/*Test case when input is trf Iterator and UDD is returning an int*/
            int sv_result = bolt::cl::reduce(tsv_trf_begin1, tsv_trf_end1, 0, plus_int);
            int dv_result = bolt::cl::reduce(tdv_trf_begin1, tdv_trf_end1, 0, plus_int);
            /*Compute expected results*/
            int expected_result = std::accumulate(tsv_trf_begin1, tsv_trf_end1, 0, plus_int);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }

        {/*Test case when input is trf Iterator*/
            UDD sv_result = bolt::cl::reduce(sv_trf_begin1, sv_trf_end1, UDDzero, plus);
            UDD dv_result = bolt::cl::reduce(dv_trf_begin1, dv_trf_end1, UDDzero, plus);
            /*Compute expected results*/
            UDD expected_result = std::accumulate(sv_trf_begin1, sv_trf_end1, UDDzero, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when input is a randomAccessIterator */
            UDD sv_result = bolt::cl::reduce(svIn2Vec.begin(), svIn2Vec.end(), UDDzero, plus);
            UDD dv_result = bolt::cl::reduce(dvIn2Vec.begin(), dvIn2Vec.end(), UDDzero, plus);
            /*Compute expected results*/
            UDD expected_result = std::accumulate(svIn2Vec.begin(), svIn2Vec.end(), UDDzero, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when input is a constant iterator */
            UDD sv_result = bolt::cl::reduce(const_itr_begin, const_itr_end, UDDzero, plus);
            UDD dv_result = bolt::cl::reduce(const_itr_begin, const_itr_end, UDDzero, plus);
            /*Compute expected results*/
            UDD expected_result = std::accumulate(const_itr_begin, const_itr_end, UDDzero, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when input is a counting iterator */
            UDD sv_result = bolt::cl::reduce(count_itr_begin, count_itr_end, UDDzero, plus);
            UDD dv_result = bolt::cl::reduce(count_itr_begin, count_itr_end, UDDzero, plus);
            /*Compute expected results*/
            UDD expected_result = std::accumulate(count_itr_begin, count_itr_end, UDDzero, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }

		//Failing Test Cases
        int mul_test_length = 20;
	    {/*Test case when input is trf Iterator*/
            UDD sv_result = bolt::cl::reduce(sv_trf_begin1, sv_trf_begin1+mul_test_length, UDDone, mul);
            UDD dv_result = bolt::cl::reduce(dv_trf_begin1, dv_trf_begin1+mul_test_length, UDDone, mul);
            /*Compute expected results*/
            UDD expected_result = std::accumulate(sv_trf_begin1, sv_trf_begin1+mul_test_length, UDDone, mul);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when input is a randomAccessIterator */
            UDD sv_result = bolt::cl::reduce(svIn2Vec.begin(), svIn2Vec.begin()+mul_test_length, UDDone, mul);
            UDD dv_result = bolt::cl::reduce(dvIn2Vec.begin(), dvIn2Vec.begin()+mul_test_length, UDDone, mul);
            /*Compute expected results*/
            UDD expected_result = std::accumulate(svIn2Vec.begin(), svIn2Vec.begin()+mul_test_length, UDDone, mul);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when input is a constant iterator */
            UDD sv_result = bolt::cl::reduce(const_itr_begin, const_itr_begin+mul_test_length, UDDone, mul);
            UDD dv_result = bolt::cl::reduce(const_itr_begin, const_itr_begin+mul_test_length, UDDone, mul);
            /*Compute expected results*/
            UDD expected_result = std::accumulate(const_itr_begin, const_itr_begin+mul_test_length, UDDone, mul);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when input is a counting iterator */
            UDD sv_result = bolt::cl::reduce(count_itr_begin, count_itr_begin+mul_test_length, UDDone, mul);
            UDD dv_result = bolt::cl::reduce(count_itr_begin, count_itr_begin+mul_test_length, UDDone, mul);
            /*Compute expected results*/
            UDD expected_result = std::accumulate(count_itr_begin, count_itr_begin+mul_test_length, UDDone, mul);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, TransformReduceRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > stlOut( length );
        

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        
        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        add_3 add3;

        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);

        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;

        int init = (int) rand();
        bolt::cl::plus<int> plus;
        {/*Test case when inputs are trf Iterators*/
            int sv_result = bolt::cl::transform_reduce(sv_trf_begin1, sv_trf_end1, add3, init, plus);
            int dv_result = bolt::cl::transform_reduce(dv_trf_begin1, dv_trf_end1,  add3, init, plus);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, stlOut.begin(), add3);
            int expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }

        {/*Test case when the both are randomAccessIterator */
            int sv_result = bolt::cl::transform_reduce(svIn1Vec.begin(), svIn1Vec.end(), add3, init, plus);
            int dv_result = bolt::cl::transform_reduce(dvIn1Vec.begin(), dvIn1Vec.end(), add3, init, plus);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), add3);
            int expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            int sv_result = bolt::cl::transform_reduce(const_itr_begin, const_itr_end, add3, init, plus);
            int dv_result = bolt::cl::transform_reduce(const_itr_begin, const_itr_end, add3, init, plus);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::transform(const_vector.begin(), const_vector.end(), stlOut.begin(), add3);
            int expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            int sv_result = bolt::cl::transform_reduce(count_itr_begin, count_itr_end, add3, init, plus);
            int dv_result = bolt::cl::transform_reduce(count_itr_begin, count_itr_end, add3, init, plus);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            std::transform(count_vector.begin(), count_vector.end(), stlOut.begin(), add3);
            int expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, TransformReduceUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );
        std::vector< UDD > stlOut( length );

        gen_input_udd genUDD;
        /*Generate inputs*/
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
		global_id = 0;
        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );

        typedef std::vector< UDD >::const_iterator            sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator   dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >         counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >         constant_itr;

		add3UDD_resultUDD add3UDD;

		squareUDD_result_int sq_int;

        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator> dv_trf_itr_add3;

		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>          tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator> tdv_trf_itr_add3;

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3UDD ), sv_trf_end1 (svIn1Vec.end(), add3UDD);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3UDD ), dv_trf_end1 (dvIn1Vec.end(), add3UDD);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int ), tsv_trf_end1 (svIn1Vec.end(), sq_int);
        tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int ), tdv_trf_end1 (dvIn1Vec.end(), sq_int);

		std::vector< int > stlOut_int( length );


        UDD temp;
		temp.i=1, temp.f=2.5f;

		UDD t;
		t.i=0, t.f=0.0f;

        counting_itr count_itr_begin(t);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;



        UDD init;
		init.i = (int) rand();
		init.f = (float) rand();

        bolt::cl::plus<UDD> plus;

		int init_int = rand();
		bolt::cl::plus<int> plus_int;

		{/*Test case when input is a trf Iterator and return type of UDD is int*/
            int sv_result = bolt::cl::transform_reduce(tsv_trf_begin1, tsv_trf_end1, sq_int, init_int, plus_int);
            int dv_result = bolt::cl::transform_reduce(tdv_trf_begin1, tdv_trf_end1, sq_int, init_int, plus_int);
            /*Compute expected results*/
            std::transform(tsv_trf_begin1, tsv_trf_end1, stlOut_int.begin(), sq_int);
            int expected_result = std::accumulate(stlOut_int.begin(), stlOut_int.end(), init_int, plus_int);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }


        {/*Test case when input is a trf Iterator*/
            UDD sv_result = bolt::cl::transform_reduce(sv_trf_begin1, sv_trf_end1, add3UDD, init, plus);
            UDD dv_result = bolt::cl::transform_reduce(dv_trf_begin1, dv_trf_end1, add3UDD, init, plus);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, stlOut.begin(), add3UDD);
            UDD expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }

        {/*Test case when input is a randomAccessIterator */
            UDD sv_result = bolt::cl::transform_reduce(svIn1Vec.begin(), svIn1Vec.end(), add3UDD, init, plus);
            UDD dv_result = bolt::cl::transform_reduce(dvIn1Vec.begin(), dvIn1Vec.end(), add3UDD, init, plus);
            /*Compute expected results*/
            std::transform(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), add3UDD);
            UDD expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }

        {/*Test case when input is constant iterator */
            UDD sv_result = bolt::cl::transform_reduce(const_itr_begin, const_itr_end, add3UDD, init, plus);
            UDD dv_result = bolt::cl::transform_reduce(const_itr_begin, const_itr_end, add3UDD, init, plus);
            /*Compute expected results*/
            //std::vector<UDD> const_vector(length,temp);
            std::transform(const_itr_begin, const_itr_end, stlOut.begin(), add3UDD);
            UDD expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when  input a counting iterator */
            UDD sv_result = bolt::cl::transform_reduce(count_itr_begin, count_itr_end, add3UDD, init, plus);
            UDD dv_result = bolt::cl::transform_reduce(count_itr_begin, count_itr_end, add3UDD, init, plus);
            /*Compute expected results*/
            std::transform(count_itr_begin, count_itr_end, stlOut.begin(), add3UDD);
            UDD expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        global_id = 0; // Reset the global id counter
    }
}


#if 0
TEST( TransformIterator, CopyRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );
        bolt::BCKND::device_vector< int > dvIn1Vec( length );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;

        gen_input gen;
        typedef std::vector< int >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator> dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>          sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator> dv_trf_itr_add4;    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);

        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;

        /*Generate inputs*/
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        bolt::BCKND::generate(dvIn1Vec.begin(), dvIn1Vec.end(), gen);
        global_id = 0;
        {/*Test case when inputs are trf Iterators*/
            bolt::cl::copy(sv_trf_begin1, sv_trf_end1, svOutVec.begin());
            bolt::cl::copy(dv_trf_begin1, dv_trf_end1, dvOutVec.begin());
            /*Compute expected results*/
            std::copy(sv_trf_begin1, sv_trf_end1, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the both are randomAccessIterator */
            bolt::cl::copy(svIn1Vec.begin(), svIn1Vec.end(), svOutVec.begin());
            bolt::cl::copy(dvIn1Vec.begin(), dvIn1Vec.end(), dvOutVec.begin());
            /*Compute expected results*/
            std::copy(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::copy(const_itr_begin, const_itr_end, svOutVec.begin());
            bolt::cl::copy(const_itr_begin, const_itr_end, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::copy(const_vector.begin(), const_vector.end(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            //bolt::cl::copy(count_itr_begin, count_itr_end, svOutVec.begin());
            //bolt::cl::copy(count_itr_begin, count_itr_end, dvOutVec.begin());
            bolt::cl::copy_n(count_itr_begin, length, svOutVec.begin());
            bolt::cl::copy_n(count_itr_begin, length, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            std::copy(count_vector.begin(), count_vector.end(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}
#endif


TEST( TransformIterator, CountRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        //std::vector< int > stlOut( length );
        

        add_3 add3;
        gen_input gen;
        /*Generate inputs*/
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
   
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);

        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;



        int val = (int) rand();

        {/*Test case when inputs are trf Iterators*/
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type sv_result = (int) bolt::cl::count(sv_trf_begin1, sv_trf_end1, val);
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type dv_result = (int) bolt::cl::count(dv_trf_begin1, dv_trf_end1, val);
            /*Compute expected results*/
            std::iterator_traits<std::vector<int>::iterator>::difference_type expected_result = std::count(sv_trf_begin1, sv_trf_end1, val);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the both are randomAccessIterator */
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type sv_result = (int) bolt::cl::count(svIn1Vec.begin(), svIn1Vec.end(), val);
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type dv_result = (int) bolt::cl::count(dvIn1Vec.begin(), dvIn1Vec.end(), val);
            /*Compute expected results*/
            std::iterator_traits<std::vector<int>::iterator>::difference_type expected_result = std::count(svIn1Vec.begin(), svIn1Vec.end(), val);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type sv_result = (int) bolt::cl::count(const_itr_begin, const_itr_end, val);
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type dv_result = (int) bolt::cl::count(const_itr_begin, const_itr_end, val);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::iterator_traits<std::vector<int>::iterator>::difference_type expected_result = std::count(const_vector.begin(), const_vector.end(), val);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type sv_result = (int) bolt::cl::count(count_itr_begin, count_itr_end, val);
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type dv_result = (int) bolt::cl::count(count_itr_begin, count_itr_end, val);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            std::iterator_traits<std::vector<int>::iterator>::difference_type expected_result = std::count(count_vector.begin(), count_vector.end(), val);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, CountUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );

		squareUDD_resultUDD sqUDD;
		squareUDD_result_int sq_int;

        /*Generate inputs*/
        gen_input_udd genUDD;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;
        
        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );

        typedef std::vector< UDD >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< squareUDD_resultUDD, std::vector< UDD >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator> dv_trf_itr_add3;
       
		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>          tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator> tdv_trf_itr_add3;

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), sqUDD), sv_trf_end1 (svIn1Vec.end(), sqUDD);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), sqUDD), dv_trf_end1 (dvIn1Vec.end(), sqUDD);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
        tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);


        UDD temp;
		temp.i=1, temp.f=2.5f;

		UDD init;
		init.i=1, init.f=1.0f;

        counting_itr count_itr_begin(init);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;

        UDD val;
		val.i = rand();
		val.f = (float) rand();

		int val_int = rand();

		{/*Test case when inputs are trf Iterators*/
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type sv_result =  bolt::cl::count(tsv_trf_begin1, tsv_trf_end1, val_int);
            bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type dv_result =  bolt::cl::count(tdv_trf_begin1, tdv_trf_end1, val_int);
            /*Compute expected results*/
            std::iterator_traits<std::vector<int>::iterator>::difference_type expected_result = std::count(tsv_trf_begin1, tsv_trf_end1, val_int);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }


        {/*Test case when inputs are trf Iterators*/
            bolt::cl::iterator_traits<bolt::cl::device_vector<UDD>::iterator>::difference_type sv_result =  bolt::cl::count(sv_trf_begin1, sv_trf_end1, val);
            bolt::cl::iterator_traits<bolt::cl::device_vector<UDD>::iterator>::difference_type dv_result =  bolt::cl::count(dv_trf_begin1, dv_trf_end1, val);
            /*Compute expected results*/
            std::iterator_traits<std::vector<UDD>::iterator>::difference_type expected_result = std::count(sv_trf_begin1, sv_trf_end1, val);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the both are randomAccessIterator */
            bolt::cl::iterator_traits<bolt::cl::device_vector<UDD>::iterator>::difference_type sv_result =  bolt::cl::count(svIn1Vec.begin(), svIn1Vec.end(), val);
            bolt::cl::iterator_traits<bolt::cl::device_vector<UDD>::iterator>::difference_type dv_result =  bolt::cl::count(dvIn1Vec.begin(), dvIn1Vec.end(), val);
            /*Compute expected results*/
            std::iterator_traits<std::vector<UDD>::iterator>::difference_type expected_result = std::count(svIn1Vec.begin(), svIn1Vec.end(), val);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the first input is constant iterator*/
            bolt::cl::iterator_traits<bolt::cl::device_vector<UDD>::iterator>::difference_type sv_result =  bolt::cl::count(const_itr_begin, const_itr_end, val);
            bolt::cl::iterator_traits<bolt::cl::device_vector<UDD>::iterator>::difference_type dv_result =  bolt::cl::count(const_itr_begin, const_itr_end, val);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,temp);
            std::iterator_traits<std::vector<UDD>::iterator>::difference_type expected_result = std::count(const_vector.begin(), const_vector.end(), val);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the first input is counting iterator */
            bolt::cl::iterator_traits<bolt::cl::device_vector<UDD>::iterator>::difference_type sv_result =  bolt::cl::count(count_itr_begin, count_itr_end, val);
            bolt::cl::iterator_traits<bolt::cl::device_vector<UDD>::iterator>::difference_type dv_result =  bolt::cl::count(count_itr_begin, count_itr_end, val);
            /*Compute expected results*/
            std::iterator_traits<std::vector<int>::iterator>::difference_type expected_result = std::count(count_itr_begin, count_itr_end, val);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, InnerProductRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svIn2Vec( length);
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), rand);
        global_id = 0;


        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end());
        
        add_3 add3;
        bolt::cl::plus<int> plus;
        bolt::cl::minus<int> minus;

        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
       
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        sv_trf_itr_add3 sv_trf_begin2 (svIn2Vec.begin(), add3);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        counting_itr count_itr_begin2(10);
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;
        constant_itr const_itr_begin2(5);

        dv_trf_itr_add3 dv_trf_begin2 (dvIn2Vec.begin(), add3);
        int init = (int) rand();

        {/*Test case when both inputs are trf Iterators*/
            int sv_result = bolt::cl::inner_product(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, init, plus, minus);
            int dv_result = bolt::cl::inner_product(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, init, plus, minus);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, stlOut.begin(), minus);
            int expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when the both inputs are randomAccessIterator */
            int sv_result = bolt::cl::inner_product(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), init, plus, minus);
            int dv_result = bolt::cl::inner_product(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), init, plus, minus);
            /*Compute expected results*/
            int expected_result = std::inner_product(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), init, std::plus<int>(), std::minus<int>());
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when both inputs are constant iterator */
            int sv_result = bolt::cl::inner_product(const_itr_begin, const_itr_end, const_itr_begin2, init, plus, minus);
            int dv_result = bolt::cl::inner_product(const_itr_begin, const_itr_end, const_itr_begin2, init, plus, minus);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::vector<int> const_vector2(length,5);
            int expected_result = std::inner_product(const_vector.begin(), const_vector.end(), const_vector2.begin(), init, std::plus<int>(), std::minus<int>());
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when both inputs are counting iterator */
            int sv_result = bolt::cl::inner_product(count_itr_begin, count_itr_end, count_itr_begin2, init, plus, minus);
            int dv_result = bolt::cl::inner_product(count_itr_begin, count_itr_end, count_itr_begin2, init, plus, minus);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            std::vector<int> count_vector2(length);
            for (int index=0;index<length;index++)
            {
                count_vector[index] = index;
                count_vector2[index] = 10 + index;
            }
            int expected_result = std::inner_product(count_vector.begin(), count_vector.end(), count_vector2.begin(), init, std::plus<int>(), std::minus<int>());
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, InnerProductUDDRoutine)
{
    {
        const int length = 1<<8;
        std::vector< UDD > svIn1Vec( length );
        std::vector< UDD > svIn2Vec( length);
		std::vector< int > stlOutVec_int( length );

        std::vector< UDD > stlOut( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
		gen_input_udd2 genUDD2;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), genUDD2);
        global_id = 0;

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< UDD > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end());
		
        bolt::cl::plus<UDD> plus;
        bolt::cl::multiplies<UDD> mul;

    	//UDDminus minus;
        add3UDD_resultUDD sqUDD;

		squareUDD_result_int sq_int;

        typedef std::vector< UDD >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>   dv_trf_itr_add3;
       
		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>          tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator> tdv_trf_itr_add3;

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), sqUDD), sv_trf_end1 (svIn1Vec.end(), sqUDD);
        sv_trf_itr_add3 sv_trf_begin2 (svIn2Vec.begin(), sqUDD), sv_trf_end2 (svIn2Vec.end(), sqUDD);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), sqUDD), dv_trf_end1 (dvIn1Vec.end(),sqUDD);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
		tsv_trf_itr_add3 tsv_trf_begin2 (svIn2Vec.begin(), sq_int), tsv_trf_end2 (svIn2Vec.end(), sq_int);
        tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);

		UDD temp;
		temp.i=1, temp.f=2.5f;
		UDD temp2;
		temp2.i=15, temp2.f=7.5f;
		UDD init1;
		init1.i=1, init1.f=1.0f;
        UDD init2;
		init2.i=10, init2.f=10.0f;

        counting_itr count_itr_begin(init1);
        counting_itr count_itr_end = count_itr_begin + length;
        counting_itr count_itr_begin2(init2);
		counting_itr count_itr_end2 = count_itr_begin2 + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;
        constant_itr const_itr_begin2(temp2);
		constant_itr const_itr_end2 = const_itr_begin2 + length;

        dv_trf_itr_add3 dv_trf_begin2 (dvIn2Vec.begin(), sqUDD);
		tdv_trf_itr_add3 tdv_trf_begin2 (dvIn2Vec.begin(), sq_int);

        global_id = 0;

        UDD init;
		init.i = rand();
		init.f = (float) rand();

		int init_int = rand()%10;

		std::vector< UDD > sv_trf_begin2_copy( sv_trf_begin2, sv_trf_end2);
		std::vector< int> tsv_trf_begin2_copy( tsv_trf_begin2, tsv_trf_end2);

		{/*Test case when both inputs are trf Iterators with UDD returning int*/
		    bolt::cl::multiplies<int> mul_int;
		    bolt::cl::plus<int> plus_int;
            int sv_result = bolt::cl::inner_product(tsv_trf_begin1, tsv_trf_end1, tsv_trf_begin2, init_int, plus_int, mul_int);
            int dv_result = bolt::cl::inner_product(tdv_trf_begin1, tdv_trf_end1, tdv_trf_begin2, init_int, plus_int, mul_int);
            /*Compute expected results*/
            std::transform(tsv_trf_begin1, tsv_trf_end1, tsv_trf_begin2_copy.begin(), stlOutVec_int.begin(), mul_int);
            int expected_result = std::accumulate(stlOutVec_int.begin(), stlOutVec_int.end(), init_int, plus_int);

            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
        {/*Test case when both inputs are trf Iterators*/
            UDD sv_result = bolt::cl::inner_product(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, init, plus, mul);
            UDD dv_result = bolt::cl::inner_product(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, init, plus, mul);
            /*Compute expected results*/
            std::transform(sv_trf_begin1, sv_trf_end1, sv_trf_begin2_copy.begin(), stlOut.begin(), mul);
            UDD expected_result = std::accumulate(stlOut.begin(), stlOut.end(), init, plus);
			//UDD expected_result = bolt::cl::reduce(stlOut.begin(), stlOut.end(), init, plus); 

            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }
		
        {/*Test case when both inputs are constant iterators */
            UDD sv_result = bolt::cl::inner_product(const_itr_begin, const_itr_end, const_itr_begin2, init, plus, mul);
            UDD dv_result = bolt::cl::inner_product(const_itr_begin, const_itr_end, const_itr_begin2, init, plus, mul);
            /*Compute expected results*/
            std::vector<UDD> const_vector2(const_itr_begin2, const_itr_end2);
            UDD expected_result = std::inner_product(const_itr_begin, const_itr_end, const_vector2.begin(), init, plus, mul);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }

        {/*Test case when the both inputs are randomAccessIterator */
            UDD sv_result = bolt::cl::inner_product(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), init, plus, mul);
            UDD dv_result = bolt::cl::inner_product(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), init, plus, mul);
            /*Compute expected results*/
			//std::transform(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svOutVec.begin(), minus);
			//UDD expected_result = bolt::cl::reduce(svOutVec.begin(), svOutVec.end(), init, mul);
            UDD expected_result = std::inner_product(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), init, plus, mul);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }

        {/*Test case when both inputs are counting iterators */
            UDD sv_result = bolt::cl::inner_product(count_itr_begin, count_itr_end, count_itr_begin2, init, plus, mul);
            UDD dv_result = bolt::cl::inner_product(count_itr_begin, count_itr_end, count_itr_begin2, init, plus, mul);
            /*Compute expected results*/
            std::vector<UDD> count_vector2(count_itr_begin2, count_itr_end2); 
			//std::transform(count_itr_begin, count_itr_end, count_vector2.begin(), svOutVec.begin(), minus);
			//UDD expected_result = bolt::cl::reduce(svOutVec.begin(), svOutVec.end(), init, mul);
            UDD expected_result = std::inner_product(count_itr_begin, count_itr_end, count_vector2.begin(), init, plus, mul);
            /*Check the results*/
            EXPECT_EQ( expected_result, sv_result );
            EXPECT_EQ( expected_result, dv_result );
        }

        global_id = 0; // Reset the global id counter
    }
}


TEST( TransformIterator, ScatterRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svIn2Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;
		add_0 add0;

        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add0);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add0);
        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;

        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::scatter(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec.begin());
            bolt::cl::scatter(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dvOutVec.begin());
            /*Compute expected results*/
            Serial_scatter(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::scatter(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svOutVec.begin());
            bolt::cl::scatter(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            Serial_scatter(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::scatter(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svOutVec.begin());
            bolt::cl::scatter(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dvOutVec.begin());
            /*Compute expected results*/
            Serial_scatter(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the both are randomAccessIterator */
            bolt::cl::scatter(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svOutVec.begin());
            bolt::cl::scatter(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            Serial_scatter(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        // Map cannot be a constant iterator! 

        //{/*Test case when the first input is trf_itr and the second is a constant iterator */
        //    bolt::cl::scatter(sv_trf_begin1, sv_trf_end1, const_itr_begin, svOutVec.begin());
        //    bolt::cl::scatter(dv_trf_begin1, dv_trf_end1, const_itr_begin, dvOutVec.begin());
        //    /*Compute expected results*/
        //    std::vector<int> const_vector(length,1);
        //    Serial_scatter(sv_trf_begin1, sv_trf_end1, const_vector.begin(), stlOut.begin());
        //    /*Check the results*/
        //    cmpArrays(svOutVec, stlOut, length);
        //    cmpArrays(dvOutVec, stlOut, length);
        //}

        {/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::scatter(sv_trf_begin1, sv_trf_end1, count_itr_begin, svOutVec.begin());
            bolt::cl::scatter(dv_trf_begin1, dv_trf_end1, count_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            Serial_scatter(sv_trf_begin1, sv_trf_end1, count_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::scatter(const_itr_begin, const_itr_end, svIn2Vec.begin(), svOutVec.begin());
            bolt::cl::scatter(const_itr_begin, const_itr_end, dvIn2Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);          
            Serial_scatter(const_vector.begin(), const_vector.end(), svIn2Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::scatter(const_itr_begin, const_itr_end, count_itr_begin, svOutVec.begin());
            bolt::cl::scatter(const_itr_begin, const_itr_end, count_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_scatter(const_vector.begin(), const_vector.end(), count_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		 {/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::scatter(svIn1Vec.begin(), svIn1Vec.end(), count_itr_begin, svOutVec.begin());
            bolt::cl::scatter(dvIn1Vec.begin(), dvIn1Vec.end(), count_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_scatter(svIn1Vec.begin(), svIn1Vec.end(), count_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        global_id = 0; // Reset the global id counter
    }
 }
 
TEST( TransformIterator, ScatterUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );
        std::vector< int > svIn2Vec( length );
        std::vector< UDD > svOutVec( length );
        std::vector< int > tsvOutVec( length );
        std::vector< UDD > stlOut( length );
        std::vector< int > tstlOut( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
		gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );
        bolt::BCKND::device_vector< int > tdvOutVec( length );

        add3UDD_resultUDD add3;
		add_0 add0;

		squareUDD_result_int sq_int;

        typedef std::vector< UDD >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< UDD  >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD  >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;  

		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>            tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator>   tdv_trf_itr_add3;


        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add0);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add0);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
		tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);

		UDD temp;
		temp.i = rand()%10, temp.f = (float) rand();

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;

		{/*Test case when both inputs are trf Iterators and UDD returns an int*/
            bolt::cl::scatter(tsv_trf_begin1, tsv_trf_end1, sv_trf_begin2, tsvOutVec.begin());
            bolt::cl::scatter(tdv_trf_begin1, tdv_trf_end1, dv_trf_begin2, tdvOutVec.begin());
            /*Compute expected results*/
            Serial_scatter(tsv_trf_begin1, tsv_trf_end1, sv_trf_begin2, tstlOut.begin());
            /*Check the results*/
            cmpArrays(tsvOutVec, tstlOut, length);
            cmpArrays(tdvOutVec, tstlOut, length);
        }

        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::scatter(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec.begin());
            bolt::cl::scatter(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dvOutVec.begin());
            /*Compute expected results*/
            Serial_scatter(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::scatter(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svOutVec.begin());
            bolt::cl::scatter(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            Serial_scatter(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::scatter(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svOutVec.begin());
            bolt::cl::scatter(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dvOutVec.begin());
            /*Compute expected results*/
            Serial_scatter(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the both are randomAccessIterator */
            bolt::cl::scatter(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svOutVec.begin());
            bolt::cl::scatter(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            Serial_scatter(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::scatter(sv_trf_begin1, sv_trf_end1, count_itr_begin, svOutVec.begin());
            bolt::cl::scatter(dv_trf_begin1, dv_trf_end1, count_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
			{
                count_vector[index] = index;
			}
            Serial_scatter(sv_trf_begin1, sv_trf_end1, count_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::scatter(const_itr_begin, const_itr_end, svIn2Vec.begin(), svOutVec.begin());
            bolt::cl::scatter(const_itr_begin, const_itr_end, dvIn2Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,temp);          
            Serial_scatter(const_vector.begin(), const_vector.end(), svIn2Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::scatter(const_itr_begin, const_itr_end, count_itr_begin, svOutVec.begin());
            bolt::cl::scatter(const_itr_begin, const_itr_end, count_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,temp);
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
			{
                count_vector[index] = index;
			}          
            Serial_scatter(const_vector.begin(), const_vector.end(), count_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		 {/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::scatter(svIn1Vec.begin(), svIn1Vec.end(), count_itr_begin, svOutVec.begin());
            bolt::cl::scatter(dvIn1Vec.begin(), dvIn1Vec.end(), count_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
			{
                count_vector[index] = index;
			}           
            Serial_scatter(svIn1Vec.begin(), svIn1Vec.end(), count_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        global_id = 0; // Reset the global id counter
    }
 }

TEST( TransformIterator, ScatterIfRoutine)
 {
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length ); // input
        std::vector< int > svIn2Vec( length ); // map
		std::vector< int > svIn3Vec( length ); // stencil
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

		for(int i=0; i<length; i++)
		{
			if(i%2 == 0)
				svIn3Vec[i] = 0;
			else
				svIn3Vec[i] = 1;
		}
		bolt::BCKND::device_vector< int > dvIn3Vec( svIn3Vec.begin(), svIn3Vec.end() );

        add_3 add3;
		add_0 add0;

        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;  
		typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add0;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add0; 

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add0);
		sv_trf_itr_add0 sv_trf_begin3 (svIn3Vec.begin(), add0);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add0);
		dv_trf_itr_add0 dv_trf_begin3 (dvIn3Vec.begin(), add0);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;
		constant_itr stencil_itr_begin(1);
        constant_itr stencil_itr_end = stencil_itr_begin + length;

		is_even iepred;

        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, sv_trf_begin3, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dv_trf_begin3, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, sv_trf_begin3, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), sv_trf_begin3, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dv_trf_begin3, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), sv_trf_begin3, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, sv_trf_begin3, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dv_trf_begin3, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, sv_trf_begin3, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, count_itr_begin, svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, count_itr_begin, dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, count_vector.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, count_itr_begin, stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, count_itr_begin, stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length), stencil_vector(length);
            for (int index=0;index<length;index++)
			{
                count_vector[index] = index;
				stencil_vector[index] = 1;
			}
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, count_vector.begin(), stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the both are randomAccessIterator */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
       
		{/*Test case when the both are randomAccessIterator */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
			}
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, svIn2Vec.begin(), svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, dvIn2Vec.begin(), dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);          
            Serial_scatter_if(const_vector.begin(), const_vector.end(), svIn2Vec.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, svIn2Vec.begin(), stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, dvIn2Vec.begin(), stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1); 
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
			}
            Serial_scatter_if(const_vector.begin(), const_vector.end(), svIn2Vec.begin(), stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, count_itr_begin, svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, count_itr_begin, dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_scatter_if(const_vector.begin(), const_vector.end(), count_vector.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, count_itr_begin, stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, count_itr_begin, stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::vector<int> count_vector(length);
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
				count_vector[index] = index; 
			}
                           
            Serial_scatter_if(const_vector.begin(), const_vector.end(), count_vector.begin(),  stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), count_itr_begin, svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), count_itr_begin, dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), count_vector.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), count_itr_begin, stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), count_itr_begin, stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
				count_vector[index] = index; 
			}
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), count_vector.begin(), stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        global_id = 0; // Reset the global id counter
    }
 }

TEST( TransformIterator, ScatterIfUDDRoutine)
 {
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length ); // input
        std::vector< int > svIn2Vec( length ); // map
		std::vector< int > svIn3Vec( length ); // stencil
        std::vector< UDD > svOutVec( length );
        std::vector< UDD > stlOut( length );
        std::vector< int > tsvOutVec( length );
        std::vector< int > tstlOut( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
		gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;


        bolt::BCKND::device_vector< UDD> dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );
        bolt::BCKND::device_vector< int > tdvOutVec( length );

		for(int i=0; i<length; i++)
		{
			if(i%2 == 0)
				svIn3Vec[i] = 0;
			else
				svIn3Vec[i] = 1;
		}
		bolt::BCKND::device_vector< int > dvIn3Vec( svIn3Vec.begin(), svIn3Vec.end() );

		add3UDD_resultUDD add3;
		add_0 add0;

		squareUDD_result_int sq_int;

        typedef std::vector< UDD >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                  constant_itr;
		typedef bolt::BCKND::constant_iterator< int >                                                  const_itr;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;  
		typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add0;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add0; 

		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>            tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator>   tdv_trf_itr_add3;

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add0);
		sv_trf_itr_add0 sv_trf_begin3 (svIn3Vec.begin(), add0);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add0);
		dv_trf_itr_add0 dv_trf_begin3 (dvIn3Vec.begin(), add0);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
		tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);

		UDD t;
		t.i = (int)rand()%10;
		t.f = (float)rand();

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(t);
        constant_itr const_itr_end = const_itr_begin + length;
		const_itr stencil_itr_begin(1);
        const_itr stencil_itr_end = stencil_itr_begin + length;

		is_even iepred;
		
        {/*Test case when both inputs are trf Iterators and UDD is returning an int*/
            bolt::cl::scatter_if(tsv_trf_begin1, tsv_trf_end1, sv_trf_begin2, sv_trf_begin3, tsvOutVec.begin(), iepred);
            bolt::cl::scatter_if(tdv_trf_begin1, tdv_trf_end1, dv_trf_begin2, dv_trf_begin3, tdvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(tsv_trf_begin1, tsv_trf_end1, sv_trf_begin2, sv_trf_begin3, tstlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(tsvOutVec, tstlOut, length);
            cmpArrays(tdvOutVec, tstlOut, length);
        }

        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, sv_trf_begin3, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dv_trf_begin3, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, sv_trf_begin3, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), sv_trf_begin3, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dv_trf_begin3, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), sv_trf_begin3, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, sv_trf_begin3, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dv_trf_begin3, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, sv_trf_begin3, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, count_itr_begin, svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, count_itr_begin, dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, count_vector.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, count_itr_begin, stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, count_itr_begin, stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length), stencil_vector(length);
            for (int index=0;index<length;index++)
			{
                count_vector[index] = index;
				stencil_vector[index] = 1;
			}
            Serial_scatter_if(sv_trf_begin1, sv_trf_end1, count_vector.begin(), stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the both are randomAccessIterator */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
       
		{/*Test case when the both are randomAccessIterator */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
			}
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, svIn2Vec.begin(), svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, dvIn2Vec.begin(), dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t);          
            Serial_scatter_if(const_vector.begin(), const_vector.end(), svIn2Vec.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, svIn2Vec.begin(), stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, dvIn2Vec.begin(), stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t); 
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
			}
            Serial_scatter_if(const_vector.begin(), const_vector.end(), svIn2Vec.begin(), stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, count_itr_begin, svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, count_itr_begin, dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t);
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_scatter_if(const_vector.begin(), const_vector.end(), count_vector.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, count_itr_begin, stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(const_itr_begin, const_itr_end, count_itr_begin, stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t);
            std::vector<int> count_vector(length);
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
				count_vector[index] = index; 
			}
                           
            Serial_scatter_if(const_vector.begin(), const_vector.end(), count_vector.begin(),  stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), count_itr_begin, svIn3Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), count_itr_begin, dvIn3Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), count_vector.begin(), svIn3Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::scatter_if(svIn1Vec.begin(), svIn1Vec.end(), count_itr_begin, stencil_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::scatter_if(dvIn1Vec.begin(), dvIn1Vec.end(), count_itr_begin, stencil_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
				count_vector[index] = index; 
			}
            Serial_scatter_if(svIn1Vec.begin(), svIn1Vec.end(), count_vector.begin(), stencil_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        global_id = 0; // Reset the global id counter
    }
 }

TEST( TransformIterator, GatherRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svIn2Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;
        
        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;
		add_0 add0;

        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;    

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3); 
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add0), sv_trf_end2 (svIn2Vec.end(), add0);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add0), dv_trf_end2 (dvIn2Vec.end(), add0);
        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;

        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::gather(sv_trf_begin2, sv_trf_end2, sv_trf_begin1, svOutVec.begin());
            bolt::cl::gather(dv_trf_begin2, dv_trf_end2, dv_trf_begin1, dvOutVec.begin());
            /*Compute expected results*/
            Serial_gather(sv_trf_begin2, sv_trf_end2, sv_trf_begin1, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and map is a randomAccessIterator */
            bolt::cl::gather(sv_trf_begin2, sv_trf_end2, svIn1Vec.begin(), svOutVec.begin());
            bolt::cl::gather(dv_trf_begin2, dv_trf_end2, dvIn1Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            Serial_gather(sv_trf_begin2, sv_trf_end2, svIn1Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is a randomAccessIterator  and map is a trf_itr */
            bolt::cl::gather(svIn2Vec.begin(), svIn2Vec.end(), sv_trf_begin1, svOutVec.begin());
            bolt::cl::gather(dvIn2Vec.begin(), dvIn2Vec.end(), dv_trf_begin1, dvOutVec.begin());
            /*Compute expected results*/
            Serial_gather(svIn2Vec.begin(), svIn2Vec.end(), sv_trf_begin1, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the both are randomAccessIterator */
            bolt::cl::gather(svIn2Vec.begin(), svIn2Vec.end(), svIn1Vec.begin(), svOutVec.begin());
            bolt::cl::gather(dvIn2Vec.begin(), dvIn2Vec.end(), dvIn1Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            Serial_gather(svIn2Vec.begin(), svIn2Vec.end(), svIn1Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and map is a counting iterator */
            bolt::cl::gather(count_itr_begin, count_itr_end, sv_trf_begin1, svOutVec.begin());
            bolt::cl::gather(count_itr_begin, count_itr_end, dv_trf_begin1, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            Serial_gather(count_vector.begin(), count_vector.end(), sv_trf_begin1, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is constant iterator and map is a randomAccessIterator */
            bolt::cl::gather(svIn2Vec.begin(), svIn2Vec.end(), const_itr_begin, svOutVec.begin());
            bolt::cl::gather(dvIn2Vec.begin(), dvIn2Vec.end(), const_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);          
            Serial_gather(svIn2Vec.begin(), svIn2Vec.end(), const_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the first input is constant iterator and map is a counting iterator */
            bolt::cl::gather(count_itr_begin,  count_itr_end, const_itr_begin, svOutVec.begin());
            bolt::cl::gather(count_itr_begin,  count_itr_end, const_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_gather( count_vector.begin(), count_vector.end(), const_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		 {/*Test case when the first input is a randomAccessIterator and map is a counting iterator */
            bolt::cl::gather(count_itr_begin, count_itr_end, svIn1Vec.begin(), svOutVec.begin());
            bolt::cl::gather(count_itr_begin, count_itr_end, dvIn1Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_gather(count_vector.begin(), count_vector.end(), svIn1Vec.begin(),stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        global_id = 0; // Reset the global id counter
    }
 }

TEST( TransformIterator, GatherUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD> svIn1Vec( length );
        std::vector< int > svIn2Vec( length );
        std::vector< UDD > svOutVec( length );
        std::vector< UDD > stlOut( length );

        gen_input_udd genUDD;
		gen_input gen;
        /*Generate inputs*/
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );

        add3UDD_resultUDD add3;
		add_0 add0;

		squareUDD_result_int sq_int;

        typedef std::vector< UDD >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4; 

		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>           tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator>  tdv_trf_itr_add3;

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3); 
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add0), sv_trf_end2 (svIn2Vec.end(), add0);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add0), dv_trf_end2 (dvIn2Vec.end(), add0);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int); 
		tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int);

		UDD t;
		t.i = (int)rand()%10;
		t.f = (float) rand();

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(t);
        constant_itr const_itr_end = const_itr_begin + length;

		 {/*Test case when both inputs are trf Iterators and UDD returns int*/
            bolt::cl::gather(sv_trf_begin2, sv_trf_end2, tsv_trf_begin1, svOutVec.begin());
            bolt::cl::gather(dv_trf_begin2, dv_trf_end2, tdv_trf_begin1, dvOutVec.begin());
            /*Compute expected results*/
            Serial_gather(sv_trf_begin2, sv_trf_end2, tsv_trf_begin1, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::gather(sv_trf_begin2, sv_trf_end2, sv_trf_begin1, svOutVec.begin());
            bolt::cl::gather(dv_trf_begin2, dv_trf_end2, dv_trf_begin1, dvOutVec.begin());
            /*Compute expected results*/
            Serial_gather(sv_trf_begin2, sv_trf_end2, sv_trf_begin1, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and map is a randomAccessIterator */
            bolt::cl::gather(sv_trf_begin2, sv_trf_end2, svIn1Vec.begin(), svOutVec.begin());
            bolt::cl::gather(dv_trf_begin2, dv_trf_end2, dvIn1Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            Serial_gather(sv_trf_begin2, sv_trf_end2, svIn1Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is a randomAccessIterator  and map is a trf_itr */
            bolt::cl::gather(svIn2Vec.begin(), svIn2Vec.end(), sv_trf_begin1, svOutVec.begin());
            bolt::cl::gather(dvIn2Vec.begin(), dvIn2Vec.end(), dv_trf_begin1, dvOutVec.begin());
            /*Compute expected results*/
            Serial_gather(svIn2Vec.begin(), svIn2Vec.end(), sv_trf_begin1, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the both are randomAccessIterator */
            bolt::cl::gather(svIn2Vec.begin(), svIn2Vec.end(), svIn1Vec.begin(), svOutVec.begin());
            bolt::cl::gather(dvIn2Vec.begin(), dvIn2Vec.end(), dvIn1Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            Serial_gather(svIn2Vec.begin(), svIn2Vec.end(), svIn1Vec.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and map is a counting iterator */
            bolt::cl::gather(count_itr_begin, count_itr_end, sv_trf_begin1, svOutVec.begin());
            bolt::cl::gather(count_itr_begin, count_itr_end, dv_trf_begin1, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            Serial_gather(count_vector.begin(), count_vector.end(), sv_trf_begin1, stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is constant iterator and map is a randomAccessIterator */
            bolt::cl::gather(svIn2Vec.begin(), svIn2Vec.end(), const_itr_begin, svOutVec.begin());
            bolt::cl::gather(dvIn2Vec.begin(), dvIn2Vec.end(), const_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t);          
            Serial_gather(svIn2Vec.begin(), svIn2Vec.end(), const_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the first input is constant iterator and map is a counting iterator */
            bolt::cl::gather(count_itr_begin,  count_itr_end, const_itr_begin, svOutVec.begin());
            bolt::cl::gather(count_itr_begin,  count_itr_end, const_itr_begin, dvOutVec.begin());
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t);
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_gather( count_vector.begin(), count_vector.end(), const_vector.begin(), stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		 {/*Test case when the first input is a randomAccessIterator and map is a counting iterator */
            bolt::cl::gather(count_itr_begin, count_itr_end, svIn1Vec.begin(), svOutVec.begin());
            bolt::cl::gather(count_itr_begin, count_itr_end, dvIn1Vec.begin(), dvOutVec.begin());
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_gather(count_vector.begin(), count_vector.end(), svIn1Vec.begin(),stlOut.begin());
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        global_id = 0; // Reset the global id counter
    }
 }

TEST( TransformIterator, GatherIfRoutine)
 {
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length ); // input
        std::vector< int > svIn2Vec( length ); // map
		std::vector< int > svIn3Vec( length ); // stencil
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

		for(int i=0; i<length; i++)
		{
			if(i%2 == 0)
				svIn3Vec[i] = 0;
			else
				svIn3Vec[i] = 1;
		}
		bolt::BCKND::device_vector< int > dvIn3Vec( svIn3Vec.begin(), svIn3Vec.end() );

        add_3 add3;
		add_0 add0;

        typedef std::vector< int >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                  constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;  
		typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add0;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add0; 

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3); // Input
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add0), sv_trf_end2 (svIn2Vec.end(), add0); //Map
		sv_trf_itr_add0 sv_trf_begin3 (svIn3Vec.begin(), add0); // Stencil
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add0), dv_trf_end2 (dvIn2Vec.end(), add0);
		dv_trf_itr_add0 dv_trf_begin3 (dvIn3Vec.begin(), add0);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;
		constant_itr stencil_itr_begin(1);
        constant_itr stencil_itr_end = stencil_itr_begin + length;


		is_even iepred;

        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dv_trf_begin2, dv_trf_end2, dv_trf_begin3, dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), sv_trf_begin3, sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), dv_trf_begin3, dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), sv_trf_begin3, sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
		{/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), dvIn3Vec.begin(), dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if( dv_trf_begin2, dv_trf_end2, dv_trf_begin3, dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::gather_if(sv_trf_begin2, sv_trf_end2, svIn3Vec.begin(), svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if(dv_trf_begin2, dv_trf_end2, dvIn3Vec.begin(), dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(sv_trf_begin2, sv_trf_end2, svIn3Vec.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::gather_if(count_itr_begin, count_itr_end, svIn3Vec.begin(), sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(count_itr_begin, count_itr_end, dvIn3Vec.begin(), dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            Serial_gather_if(count_vector.begin(), count_vector.end(), svIn3Vec.begin(), sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length), stencil_vector(length);
            for (int index=0;index<length;index++)
			{
                count_vector[index] = index;
				stencil_vector[index] = 1;
			}
            Serial_gather_if(count_vector.begin(), count_vector.end(), stencil_vector.begin(), sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the both are randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), dvIn3Vec.begin(), dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
       
		{/*Test case when the both are randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), stencil_itr_begin, svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), stencil_itr_begin,  dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
			}
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), stencil_vector.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), const_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), dvIn3Vec.begin(), const_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);          
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), const_vector.begin(),stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), stencil_itr_begin, const_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), stencil_itr_begin, const_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1); 
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
			}
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), stencil_vector.begin(), const_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::gather_if( count_itr_begin, count_itr_end, svIn3Vec.begin(), const_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::gather_if( count_itr_begin, count_itr_end, dvIn3Vec.begin(), const_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_gather_if(count_vector.begin(), count_vector.end(), svIn3Vec.begin(), const_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, const_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, const_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::vector<int> count_vector(length);
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
				count_vector[index] = index; 
			}
                           
            Serial_gather_if(count_vector.begin(), count_vector.end(), stencil_vector.begin(), const_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::gather_if( count_itr_begin, count_itr_end, svIn3Vec.begin(), svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if( count_itr_begin, count_itr_end, dvIn3Vec.begin(), dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_gather_if(count_vector.begin(), count_vector.end(), svIn3Vec.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
				count_vector[index] = index; 
			}
            Serial_gather_if(count_vector.begin(), count_vector.end(), stencil_vector.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        global_id = 0; // Reset the global id counter
    }
 }

TEST( TransformIterator, GatherIfUDDRoutine)
 {
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length ); // input
        std::vector< int > svIn2Vec( length ); // map
		std::vector< int > svIn3Vec( length ); // stencil
        std::vector< UDD > svOutVec( length );
        std::vector< UDD > stlOut( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
		gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );

		for(int i=0; i<length; i++)
		{
			if(i%2 == 0)
				svIn3Vec[i] = 0;
			else
				svIn3Vec[i] = 1;
		}
		bolt::BCKND::device_vector< int > dvIn3Vec( svIn3Vec.begin(), svIn3Vec.end() );

        add3UDD_resultUDD add3;
		add_0 add0;


        squareUDD_result_int sq_int;

		typedef std::vector< UDD >::const_iterator                                                     sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                            dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                  counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                  constant_itr;
		typedef bolt::BCKND::constant_iterator< int >                                                  const_itr;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>            sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>   dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add4;  
		typedef bolt::BCKND::transform_iterator< add_0, std::vector< int >::const_iterator>            sv_trf_itr_add0;
        typedef bolt::BCKND::transform_iterator< add_0, bolt::BCKND::device_vector< int >::iterator>   dv_trf_itr_add0; 

		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>            tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator>   tdv_trf_itr_add3;

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3); // Input
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add0), sv_trf_end2 (svIn2Vec.end(), add0); //Map
		sv_trf_itr_add0 sv_trf_begin3 (svIn3Vec.begin(), add0); // Stencil
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add0), dv_trf_end2 (dvIn2Vec.end(), add0);
		dv_trf_itr_add0 dv_trf_begin3 (dvIn3Vec.begin(), add0);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int); // Input
		tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int);

		UDD t;
		t.i = (int)rand()%10;
		t.f = (float)rand();

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(t);
        constant_itr const_itr_end = const_itr_begin + length;
		const_itr stencil_itr_begin(1);
        const_itr stencil_itr_end = stencil_itr_begin + length;

		is_even iepred;

		{/*Test case when both inputs are trf Iterators with UDD returning int*/
            bolt::cl::gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, tsv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dv_trf_begin2, dv_trf_end2, dv_trf_begin3, tdv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, tsv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when both inputs are trf Iterators*/
            bolt::cl::gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dv_trf_begin2, dv_trf_end2, dv_trf_begin3, dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), sv_trf_begin3, sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), dv_trf_begin3, dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), sv_trf_begin3, sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
		{/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), dvIn3Vec.begin(), dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        {/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if( dv_trf_begin2, dv_trf_end2, dv_trf_begin3, dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(sv_trf_begin2, sv_trf_end2, sv_trf_begin3, svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is a randomAccessIterator  and second is a trf_itr */
            bolt::cl::gather_if(sv_trf_begin2, sv_trf_end2, svIn3Vec.begin(), svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if(dv_trf_begin2, dv_trf_end2, dvIn3Vec.begin(), dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(sv_trf_begin2, sv_trf_end2, svIn3Vec.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::gather_if(count_itr_begin, count_itr_end, svIn3Vec.begin(), sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(count_itr_begin, count_itr_end, dvIn3Vec.begin(), dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
            Serial_gather_if(count_vector.begin(), count_vector.end(), svIn3Vec.begin(), sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is trf_itr and the second is a counting iterator */
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, sv_trf_begin1, svOutVec.begin(), iepred);
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, dv_trf_begin1, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length), stencil_vector(length);
            for (int index=0;index<length;index++)
			{
                count_vector[index] = index;
				stencil_vector[index] = 1;
			}
            Serial_gather_if(count_vector.begin(), count_vector.end(), stencil_vector.begin(), sv_trf_begin1, stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the both are randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), dvIn3Vec.begin(), dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
       
		{/*Test case when the both are randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), stencil_itr_begin, svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), stencil_itr_begin,  dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
			}
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), stencil_vector.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), const_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), dvIn3Vec.begin(), const_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t);          
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), svIn3Vec.begin(), const_vector.begin(),stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a randomAccessIterator */
            bolt::cl::gather_if(svIn2Vec.begin(), svIn2Vec.end(), stencil_itr_begin, const_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::gather_if(dvIn2Vec.begin(), dvIn2Vec.end(), stencil_itr_begin, const_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t); 
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
			}
            Serial_gather_if(svIn2Vec.begin(), svIn2Vec.end(), stencil_vector.begin(), const_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


        {/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::gather_if( count_itr_begin, count_itr_end, svIn3Vec.begin(), const_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::gather_if( count_itr_begin, count_itr_end, dvIn3Vec.begin(), const_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t);
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_gather_if(count_vector.begin(), count_vector.end(), svIn3Vec.begin(), const_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is constant iterator and the second is a counting iterator */
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, const_itr_begin, svOutVec.begin(), iepred);
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, const_itr_begin, dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,t);
            std::vector<int> count_vector(length);
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
				count_vector[index] = index; 
			}
                           
            Serial_gather_if(count_vector.begin(), count_vector.end(), stencil_vector.begin(), const_vector.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }


		{/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::gather_if( count_itr_begin, count_itr_end, svIn3Vec.begin(), svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if( count_itr_begin, count_itr_end, dvIn3Vec.begin(), dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;            
            Serial_gather_if(count_vector.begin(), count_vector.end(), svIn3Vec.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

		{/*Test case when the first input is a randomAccessIterator and the second is a counting iterator */
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, svIn1Vec.begin(), svOutVec.begin(), iepred);
            bolt::cl::gather_if(count_itr_begin, count_itr_end, stencil_itr_begin, dvIn1Vec.begin(), dvOutVec.begin(), iepred);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
			std::vector<int> stencil_vector(length);
            for (int index=0;index<length;index++)
			{
				stencil_vector[index] = 1;
				count_vector[index] = index; 
			}
            Serial_gather_if(count_vector.begin(), count_vector.end(), stencil_vector.begin(), svIn1Vec.begin(), stlOut.begin(), iepred);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }

        global_id = 0; // Reset the global id counter
    }
 }



TEST( TransformIterator, ReduceByKeyRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svIn2Vec( length );
        std::vector< int > svOutVec1( length );
		std::vector< int > svOutVec2( length );
        std::vector< int > stlOut1( length );
		std::vector< int > stlOut2( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec1( length );
		bolt::BCKND::device_vector< int > dvOutVec2( length );

        add_3 add3;
        add_4 add4;
        bolt::cl::equal_to<int> binary_predictor;
        bolt::cl::plus<int> binary_operator;


        typedef std::vector< int >::const_iterator                                                         sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                                dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                      counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                      constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>                sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator>       dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>                sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator>       dv_trf_itr_add4;    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), add4);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), add4);
        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;



		std::vector< int > testInput1( svIn1Vec.begin(), svIn1Vec.end() );
		std::vector< int > testInput2( svIn2Vec.begin(), svIn2Vec.end() );
		for(int i=0; i<length; i++)
		{
			testInput1[i] = testInput1[i] + 3;
			testInput2[i] = testInput2[i] + 4;
		}

		std::vector< int > constVector(length, 1);
		std::vector< int > countVector(length);
		for(int i=0; i<length; i++)
		{
			countVector[i]=i;
		}

        {/*Test case when inputs are trf Iterators*/
            auto sv_result = bolt::cl::reduce_by_key(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&testInput1[0], &testInput2[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }
        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            auto sv_result = bolt::cl::reduce_by_key(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&testInput1[0], &svIn2Vec[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		 {/*Test case when the first input is randomAccessIterator and the second is a trf_itr*/
            auto sv_result = bolt::cl::reduce_by_key(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&svIn1Vec[0], &testInput2[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

        {/*Test case when the first input is trf_itr and the second is a constant iterator */
            auto sv_result = bolt::cl::reduce_by_key(sv_trf_begin1, sv_trf_end1, const_itr_begin, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dv_trf_begin1, dv_trf_end1, const_itr_begin, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&testInput1[0], &constVector[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		{/*Test case when the first input is constant iterator and the second is a  trf_itr */
            auto sv_result = bolt::cl::reduce_by_key(const_itr_begin, const_itr_end, sv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(const_itr_begin, const_itr_end, dv_trf_begin2, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&constVector[0], &testInput2[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

        {/*Test case when the first input is trf_itr and the second is a counting iterator */      
			auto sv_result = bolt::cl::reduce_by_key(sv_trf_begin1, sv_trf_end1, count_itr_begin, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dv_trf_begin1, dv_trf_end1, count_itr_begin, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&testInput1[0], &countVector[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		{/*Test case when the first input is counting iterator and the second is a trf_itr */
            auto sv_result = bolt::cl::reduce_by_key(count_itr_begin, count_itr_end, sv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(count_itr_begin, count_itr_end, dv_trf_begin2, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&countVector[0], &testInput2[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }


		 {/*Test case when the both inputs are randomAccessIterators*/
            auto sv_result = bolt::cl::reduce_by_key(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&svIn1Vec[0], &svIn2Vec[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		{/*Test case when the first input is constant iterator and the second is a counting iterator */
            auto sv_result = bolt::cl::reduce_by_key(const_itr_begin, const_itr_end, count_itr_begin, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(const_itr_begin, const_itr_end, count_itr_begin, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&constVector[0], &countVector[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		 {/*Test case when the first input is counting iterator and the second is a constant iterator */
            auto sv_result = bolt::cl::reduce_by_key(count_itr_begin, count_itr_end, const_itr_begin, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(count_itr_begin, count_itr_end, const_itr_begin, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&countVector[0], &constVector[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, ReduceByKeyUDDRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );
        std::vector< UDD > svIn2Vec( length );
        std::vector< UDD > svOutVec1( length );
		std::vector< UDD > svOutVec2( length );
        std::vector< UDD > stlOut1( length );
		std::vector< UDD > stlOut2( length );

        /*Generate inputs*/
        gen_input_udd genUDD;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), genUDD);
        global_id = 0;

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< UDD > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec1( length );
		bolt::BCKND::device_vector< UDD > dvOutVec2( length );

        bolt::cl::equal_to<UDD> binary_predictor;
        bolt::cl::plus<UDD> binary_operator;
		add3UDD_resultUDD sqUDD;
		add4UDD_resultUDD cbUDD;
        

		squareUDD_result_int sq_int;
		cubeUDD_result_int cb_int;
#if 0
		bolt::cl::equal_to<int> binary_predictor_int;
        bolt::cl::plus<int> binary_operator_int;
#endif
		std::vector< int > stlOut1_int( length );
		std::vector< int > stlOut2_int( length );

        typedef std::vector< UDD >::const_iterator                                                         sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                                dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >                                                      counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                      constant_itr;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, std::vector< UDD >::const_iterator>                sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add3UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>       dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add4UDD_resultUDD, std::vector< UDD >::const_iterator>                sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add4UDD_resultUDD, bolt::BCKND::device_vector< UDD >::iterator>       dv_trf_itr_add4;    

		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>              tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator>     tdv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< cubeUDD_result_int, std::vector< UDD >::const_iterator>                tsv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< cubeUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator>       tdv_trf_itr_add4;


        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), sqUDD), sv_trf_end1 (svIn1Vec.end(), sqUDD);
        sv_trf_itr_add4 sv_trf_begin2 (svIn2Vec.begin(), cbUDD);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), sqUDD), dv_trf_end1 (dvIn1Vec.end(), sqUDD);
        dv_trf_itr_add4 dv_trf_begin2 (dvIn2Vec.begin(), cbUDD);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
        tsv_trf_itr_add4 tsv_trf_begin2 (svIn2Vec.begin(), cb_int);
        tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);
        tdv_trf_itr_add4 tdv_trf_begin2 (dvIn2Vec.begin(), cb_int);

		UDD temp;
		temp.i = (int) rand()%10;
		temp.f = (float) rand();
		UDD t;
		t.i = 0;
		t.f = 0.0f;

        counting_itr count_itr_begin(t);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;



		std::vector< UDD > testInput1( svIn1Vec.begin(), svIn1Vec.end() );
		std::vector< UDD > testInput2( svIn2Vec.begin(), svIn2Vec.end() );
		for(int i=0; i<length; i++)
		{
			testInput1[i].i = testInput1[i].i + 3;   
			testInput1[i].f = testInput1[i].f + 3.f; 
			testInput2[i].i = testInput2[i].i + 4;   
			testInput2[i].f = testInput2[i].f + 4.f; 
		}

		std::vector< int > ttestInput1( length);
		std::vector< int > ttestInput2( length);
		for(int i=0; i<length; i++)
		{
			ttestInput1[i] = svIn1Vec[i].i + (int) svIn1Vec[i].f;   
			ttestInput2[i] = svIn2Vec[i].i + (int) svIn2Vec[i].f + 3;   
		}

		std::vector< UDD > constVector(const_itr_begin, const_itr_end);
		std::vector< UDD > countVector(count_itr_begin, count_itr_end);
	
#if 0	
        {/*Test case when inputs are trf Iterators and return type of UDD is int*/
            auto sv_result = bolt::cl::reduce_by_key(tsv_trf_begin1, tsv_trf_end1, tsv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), binary_predictor_int, binary_operator_int);
            auto dv_result = bolt::cl::reduce_by_key(tdv_trf_begin1, tdv_trf_end1, tdv_trf_begin2, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor_int, binary_operator_int);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&ttestInput1[0], &ttestInput2[0], &stlOut1_int[0], &stlOut2_int[0], binary_operator_int, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1_int, length);
			cmpArrays(svOutVec2, stlOut2_int, length);
            cmpArrays(dvOutVec1, stlOut1_int, length);
			cmpArrays(dvOutVec2, stlOut2_int, length);
        }
#endif

        {/*Test case when inputs are trf Iterators*/
            auto sv_result = bolt::cl::reduce_by_key(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD >> (&testInput1[0], &testInput2[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }
        {/*Test case when the first input is trf_itr and the second is a randomAccessIterator */
            auto sv_result = bolt::cl::reduce_by_key(sv_trf_begin1, sv_trf_end1, svIn2Vec.begin(), svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dv_trf_begin1, dv_trf_end1, dvIn2Vec.begin(), dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD >> (&testInput1[0], &svIn2Vec[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		 {/*Test case when the first input is randomAccessIterator and the second is a trf_itr*/
            auto sv_result = bolt::cl::reduce_by_key(svIn1Vec.begin(), svIn1Vec.end(), sv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), dv_trf_begin2, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD >> (&svIn1Vec[0], &testInput2[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

        {/*Test case when the first input is trf_itr and the second is a constant iterator */
            auto sv_result = bolt::cl::reduce_by_key(sv_trf_begin1, sv_trf_end1, const_itr_begin, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dv_trf_begin1, dv_trf_end1, const_itr_begin, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD>> (&testInput1[0], &constVector[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		{/*Test case when the first input is constant iterator and the second is a  trf_itr */
            auto sv_result = bolt::cl::reduce_by_key(const_itr_begin, const_itr_end, sv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(const_itr_begin, const_itr_end, dv_trf_begin2, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD>> (&constVector[0], &testInput2[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

        {/*Test case when the first input is trf_itr and the second is a counting iterator */      
			auto sv_result = bolt::cl::reduce_by_key(sv_trf_begin1, sv_trf_end1, count_itr_begin, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dv_trf_begin1, dv_trf_end1, count_itr_begin, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD>> (&testInput1[0], &countVector[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		{/*Test case when the first input is counting iterator and the second is a trf_itr */
            auto sv_result = bolt::cl::reduce_by_key(count_itr_begin, count_itr_end, sv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(count_itr_begin, count_itr_end, dv_trf_begin2, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD>> (&countVector[0], &testInput2[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }


		 {/*Test case when the both inputs are randomAccessIterators*/
            auto sv_result = bolt::cl::reduce_by_key(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD>> (&svIn1Vec[0], &svIn2Vec[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		{/*Test case when the first input is constant iterator and the second is a counting iterator */
            auto sv_result = bolt::cl::reduce_by_key(const_itr_begin, const_itr_end, count_itr_begin, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(const_itr_begin, const_itr_end, count_itr_begin, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD>> (&constVector[0], &countVector[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		 {/*Test case when the first input is counting iterator and the second is a constant iterator */
            auto sv_result = bolt::cl::reduce_by_key(count_itr_begin, count_itr_end, const_itr_begin, svOutVec1.begin(), svOutVec2.begin(), binary_predictor, binary_operator);
            auto dv_result = bolt::cl::reduce_by_key(count_itr_begin, count_itr_end, const_itr_begin, dvOutVec1.begin(), dvOutVec2.begin(), binary_predictor, binary_operator);
            /*Compute expected results*/
            unsigned int n= Serial_reduce_by_key<UDD, UDD, UDD, UDD, bolt::cl::plus< UDD>> (&countVector[0], &constVector[0], &stlOut1[0], &stlOut2[0], binary_operator, length);
            /*Check the results*/
            cmpArrays(svOutVec1, stlOut1, length);
			cmpArrays(svOutVec2, stlOut2, length);
            cmpArrays(dvOutVec1, stlOut1, length);
			cmpArrays(dvOutVec2, stlOut2, length);
        }

		global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, InclusiveScanRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), gen);
        global_id = 0;


        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;
        bolt::cl::plus<int> addI2;

        typedef std::vector< int >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator> dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>          sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator> dv_trf_itr_add4;    
        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);

        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
		constant_itr const_itr_end = const_itr_begin + length;
		

        {/*Test case when input is trf Iterator*/
            bolt::cl::inclusive_scan(sv_trf_begin1, sv_trf_end1, svOutVec.begin(), addI2);
            bolt::cl::inclusive_scan(dv_trf_begin1, dv_trf_end1, dvOutVec.begin(), addI2);
            /*Compute expected results*/

            std::partial_sum(sv_trf_begin1, sv_trf_end1, stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is randomAccessIterator */
            bolt::cl::inclusive_scan(svIn1Vec.begin(), svIn1Vec.end(), svOutVec.begin(),  addI2);
            bolt::cl::inclusive_scan(dvIn1Vec.begin(), dvIn1Vec.end(), dvOutVec.begin(),  addI2);
            /*Compute expected results*/
            std::partial_sum(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a constant iterator  */
            bolt::cl::inclusive_scan(const_itr_begin, const_itr_end, svOutVec.begin(), addI2);
            bolt::cl::inclusive_scan(const_itr_begin, const_itr_end, dvOutVec.begin(), addI2);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);
            std::partial_sum(const_vector.begin(), const_vector.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a counting iterator */
            bolt::cl::inclusive_scan(count_itr_begin, count_itr_end, svOutVec.begin(), addI2);
            bolt::cl::inclusive_scan(count_itr_begin, count_itr_end, dvOutVec.begin(), addI2);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;
			std::partial_sum(count_vector.begin(), count_vector.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, InclusiveScanbykeyRoutine)
{
    {
        const int length = 1<<10;
        std::vector< int > svIn1Vec( length );
        std::vector< int > svIn2Vec( length );
        std::vector< int > svOutVec( length );
        std::vector< int > stlOut( length );

        /*Generate inputs*/
        gen_input gen;
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;
        /*Create Iterators*/
		int segmentLength = 0;
		int segmentIndex = 0;
		std::vector<int> key(1);
		key[0] = 0;

		for (int i = 0; i < length; i++)
		{
			// start over, i.e., begin assigning new key
			if (segmentIndex == segmentLength)
			{
				segmentLength++;
				segmentIndex = 0;
				key[0] = key[0]+1 ; 
			}
			svIn1Vec[i] = key[0];
			segmentIndex++;
		}

        bolt::BCKND::device_vector< int > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
		bolt::BCKND::device_vector< int > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< int > dvOutVec( length );

        add_3 add3;
		bolt::cl::equal_to<int> equal_to;
        bolt::cl::plus<int> addI2;


        typedef std::vector< int >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< int >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< int >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< int >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< add_3, std::vector< int >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_3, bolt::BCKND::device_vector< int >::iterator> dv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< add_4, std::vector< int >::const_iterator>          sv_trf_itr_add4;
        typedef bolt::BCKND::transform_iterator< add_4, bolt::BCKND::device_vector< int >::iterator> dv_trf_itr_add4;    


        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
		sv_trf_itr_add3 sv_trf_begin2 (svIn2Vec.begin(), add3), sv_trf_end2 (svIn2Vec.end(), add3);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);
        dv_trf_itr_add3 dv_trf_begin2 (dvIn2Vec.begin(), add3), dv_trf_end2 (dvIn2Vec.end(), add3);

        counting_itr count_itr_begin(0);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(1);
        constant_itr const_itr_end = const_itr_begin + length;

        {/*Test case when inputs are trf Iterator*/
            bolt::cl::inclusive_scan_by_key(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec.begin(), equal_to, addI2);
            bolt::cl::inclusive_scan_by_key(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dvOutVec.begin(), equal_to, addI2);
            /*Compute expected results*/
			bolt::cl::control ctl = bolt::cl::control::getDefault( );
			ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

            bolt::cl::inclusive_scan_by_key(ctl, sv_trf_begin1, sv_trf_end1, sv_trf_begin2, stlOut.begin(), equal_to, addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
            ctl.setForceRunMode(bolt::cl::control::Automatic); 
        }
        {/*Test case when inputs are randomAccessIterator */
            bolt::cl::inclusive_scan_by_key(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svOutVec.begin(), equal_to, addI2);
            bolt::cl::inclusive_scan_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvOutVec.begin(), equal_to, addI2);
            /*Compute expected results*/
			bolt::cl::control ctl = bolt::cl::control::getDefault( );
			//ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

            bolt::cl::inclusive_scan_by_key(ctl, svIn1Vec.begin(), svIn1Vec.end(),  svIn2Vec.begin(), stlOut.begin(), equal_to, addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when value input is a constant iterator while key input is stil randomAccessIterator  */
            bolt::cl::inclusive_scan_by_key(svIn1Vec.begin(), svIn1Vec.end(), const_itr_begin, svOutVec.begin(), equal_to, addI2);
            bolt::cl::inclusive_scan_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), const_itr_begin, dvOutVec.begin(), equal_to, addI2);
            /*Compute expected results*/
            std::vector<int> const_vector(length,1);

			bolt::cl::control ctl = bolt::cl::control::getDefault( );
			//ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

            bolt::cl::inclusive_scan_by_key(ctl, svIn1Vec.begin(), svIn1Vec.end(), const_vector.begin(), stlOut.begin(), equal_to, addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a counting iterator */
            bolt::cl::inclusive_scan_by_key(svIn1Vec.begin(), svIn1Vec.end(), count_itr_begin, svOutVec.begin(), equal_to, addI2);
            bolt::cl::inclusive_scan_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), count_itr_begin, dvOutVec.begin(), equal_to, addI2);
            /*Compute expected results*/
            std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;

			bolt::cl::control ctl = bolt::cl::control::getDefault( );
			//ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

            bolt::cl::inclusive_scan_by_key(ctl, svIn1Vec.begin(), svIn1Vec.end(), count_vector.begin(), stlOut.begin(), equal_to, addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

TEST( TransformIterator, UDDInclusiveScanRoutine)
{  
        const int length = 10;
        std::vector< UDD > svIn1Vec( length);
        std::vector< UDD > svOutVec( length );
        std::vector< int > tsvOutVec( length );
        std::vector< UDD > stlOut( length );
        std::vector< int > tstlOut( length );
        /*Generate inputs*/
        gen_input_udd genUDD;
        global_id = 0;
        std::generate(svIn1Vec.begin(), svIn1Vec.end(), genUDD);
        global_id = 0;

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );
        bolt::BCKND::device_vector< int > tdvOutVec( length );

        UDDadd_3 add3;
        bolt::cl::plus<UDD> addI2;

		UDD temp;
		temp.i=1, temp.f=2.5f;

		UDD init;
		init.i=0, init.f=0.0f;

		squareUDD_result_int sq_int;

        typedef std::vector< UDD >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< UDDadd_3, std::vector< UDD >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< UDDadd_3, bolt::BCKND::device_vector< UDD >::iterator> dv_trf_itr_add3;
          
		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>          tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator> tdv_trf_itr_add3;

        /*Create Iterators*/
        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
        tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);


        counting_itr count_itr_begin(init);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
		constant_itr const_itr_end = const_itr_begin + length;
		
#if 0
        //TODO - 
		{/*Test case when input is trf Iterato and UDD is returning an intr*/
		    bolt::cl::plus<int> addI2_int;
            bolt::cl::inclusive_scan(tsv_trf_begin1, tsv_trf_end1, tsvOutVec.begin(), addI2_int);
            bolt::cl::inclusive_scan(tdv_trf_begin1, tdv_trf_end1, tdvOutVec.begin(), addI2_int);
            /*Compute expected results*/
            std::partial_sum(tsv_trf_begin1, tsv_trf_end1, tstlOut.begin(), addI2_int);
            /*Check the results*/
            cmpArrays(tsvOutVec, tstlOut, length);
            cmpArrays(tdvOutVec, tstlOut, length);
        }
#endif

        {/*Test case when input is trf Iterator*/
            bolt::cl::inclusive_scan(sv_trf_begin1, sv_trf_end1, svOutVec.begin(), addI2);
            bolt::cl::inclusive_scan(dv_trf_begin1, dv_trf_end1, dvOutVec.begin(), addI2);
            /*Compute expected results*/
            std::partial_sum(sv_trf_begin1, sv_trf_end1, stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is randomAccessIterator */
            bolt::cl::inclusive_scan(svIn1Vec.begin(), svIn1Vec.end(), svOutVec.begin(),  addI2);
            bolt::cl::inclusive_scan(dvIn1Vec.begin(), dvIn1Vec.end(), dvOutVec.begin(),  addI2);
            /*Compute expected results*/
            std::partial_sum(svIn1Vec.begin(), svIn1Vec.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a constant iterator  */
            bolt::cl::inclusive_scan(const_itr_begin, const_itr_end, svOutVec.begin(), addI2);
            bolt::cl::inclusive_scan(const_itr_begin, const_itr_end, dvOutVec.begin(), addI2);
            /*Compute expected results*/
            std::vector<UDD> const_vector(length,temp);
            std::partial_sum(const_vector.begin(), const_vector.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a counting iterator */
            bolt::cl::inclusive_scan(count_itr_begin, count_itr_end, svOutVec.begin(), addI2);
            bolt::cl::inclusive_scan(count_itr_begin, count_itr_end, dvOutVec.begin(), addI2);

            /*Compute expected results*/			
            /*std::vector<UDD> rk(count_itr_begin, count_itr_end);
			std::vector<UDD> count_vector(length);	                
            for (int index=0;index<length;index++)
			{
				count_vector[index] = index;
				std::cout<<count_vector[index].i <<"	"<<count_vector[index].f;
				std::cout<<"			"<<rk[index].i <<"	"<<rk[index].f<<"\n";
			}*/  // Why counting iterator's values are (WRONG) different than the count_vector - need to be looked
			std::vector<UDD> count_vector(count_itr_begin, count_itr_end);
			std::partial_sum(count_vector.begin(), count_vector.end(), stlOut.begin(), addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
}

TEST( TransformIterator, UDDInclusiveScanbykeyRoutine)
{
    {
        const int length = 1<<10;
        std::vector< UDD > svIn1Vec( length );
        std::vector< UDD > svIn2Vec( length );
        std::vector< UDD > svOutVec( length );
        std::vector< UDD > stlOut( length );

        /*Generate inputs*/
        gen_input_udd gen;
        global_id = 0;
        std::generate(svIn2Vec.begin(), svIn2Vec.end(), gen);
        global_id = 0;

        /*Create Iterators*/
			UDD Zero;
			Zero.i=0, Zero.f=0.0f; 

			int segmentLength = 0;
			int segmentIndex = 0;
			std::vector<UDD> key(1);
			key[0] = Zero;

			for (int i = 0; i < length; i++)
			{
				// start over, i.e., begin assigning new key
				if (segmentIndex == segmentLength)
				{
					segmentLength++;
					segmentIndex = 0;
					key[0] = key[0]+1 ; 
				}
				svIn1Vec[i] = key[0];
				segmentIndex++;
			}

        bolt::BCKND::device_vector< UDD > dvIn1Vec( svIn1Vec.begin(), svIn1Vec.end() );
		bolt::BCKND::device_vector< UDD > dvIn2Vec( svIn2Vec.begin(), svIn2Vec.end() );
        bolt::BCKND::device_vector< UDD > dvOutVec( length );

        UDDadd_3 add3;
		bolt::cl::equal_to<UDD> equal_to;
        bolt::cl::plus<UDD> addI2;

		squareUDD_result_int sq_int;

        typedef std::vector< UDD >::const_iterator                                                   sv_itr;
        typedef bolt::BCKND::device_vector< UDD >::iterator                                          dv_itr;
        typedef bolt::BCKND::counting_iterator< UDD >                                                counting_itr;
        typedef bolt::BCKND::constant_iterator< UDD >                                                constant_itr;
        typedef bolt::BCKND::transform_iterator< UDDadd_3, std::vector< UDD >::const_iterator>          sv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< UDDadd_3, bolt::BCKND::device_vector< UDD >::iterator> dv_trf_itr_add3;

		typedef bolt::BCKND::transform_iterator< squareUDD_result_int, std::vector< UDD >::const_iterator>          tsv_trf_itr_add3;
        typedef bolt::BCKND::transform_iterator< squareUDD_result_int, bolt::BCKND::device_vector< UDD >::iterator> tdv_trf_itr_add3;




        sv_trf_itr_add3 sv_trf_begin1 (svIn1Vec.begin(), add3), sv_trf_end1 (svIn1Vec.end(), add3);
		sv_trf_itr_add3 sv_trf_begin2 (svIn2Vec.begin(), add3), sv_trf_end2 (svIn2Vec.end(), add3);
        dv_trf_itr_add3 dv_trf_begin1 (dvIn1Vec.begin(), add3), dv_trf_end1 (dvIn1Vec.end(), add3);
        dv_trf_itr_add3 dv_trf_begin2 (dvIn2Vec.begin(), add3), dv_trf_end2 (dvIn2Vec.end(), add3);

		tsv_trf_itr_add3 tsv_trf_begin1 (svIn1Vec.begin(), sq_int), tsv_trf_end1 (svIn1Vec.end(), sq_int);
		tdv_trf_itr_add3 tdv_trf_begin1 (dvIn1Vec.begin(), sq_int), tdv_trf_end1 (dvIn1Vec.end(), sq_int);
		tsv_trf_itr_add3 tsv_trf_begin2 (svIn2Vec.begin(), sq_int), tsv_trf_end2 (svIn2Vec.end(), sq_int);
		tdv_trf_itr_add3 tdv_trf_begin2 (dvIn2Vec.begin(), sq_int), tdv_trf_end2 (dvIn2Vec.end(), sq_int);

		UDD temp;
		temp.i=1, temp.f=2.5f;

		UDD init;
		init.i=0, init.f=0.0f;

        counting_itr count_itr_begin(init);
        counting_itr count_itr_end = count_itr_begin + length;
        constant_itr const_itr_begin(temp);
        constant_itr const_itr_end = const_itr_begin + length;


#if 0
		{/*Test case when inputs are trf Iterators and return type of UDD is int*/
    		bolt::cl::equal_to<int> equal_to_int;
            bolt::cl::plus<int> addI2_int;
            bolt::cl::inclusive_scan_by_key(tsv_trf_begin1, tsv_trf_end1, tsv_trf_begin2, svOutVec.begin(), equal_to_int, addI2_int);
            bolt::cl::inclusive_scan_by_key(tdv_trf_begin1, tdv_trf_end1, tdv_trf_begin2, dvOutVec.begin(), equal_to_int, addI2_int);
            /*Compute expected results*/

            bolt::cl::inclusive_scan_by_key(tsv_trf_begin1, tsv_trf_end1, tsv_trf_begin2, stlOut.begin(), equal_to_int, addI2_int);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
#endif
        {/*Test case when inputs are trf Iterator*/
            bolt::cl::inclusive_scan_by_key(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec.begin(), equal_to, addI2);
            bolt::cl::inclusive_scan_by_key(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dvOutVec.begin(), equal_to, addI2);
            /*Compute expected results*/
			bolt::cl::control ctl = bolt::cl::control::getDefault( );
			//ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

            bolt::cl::inclusive_scan_by_key(ctl, sv_trf_begin1, sv_trf_end1, sv_trf_begin2, stlOut.begin(), equal_to, addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when inputs are randomAccessIterator */
            bolt::cl::inclusive_scan_by_key(svIn1Vec.begin(), svIn1Vec.end(), svIn2Vec.begin(), svOutVec.begin(), equal_to, addI2);
            bolt::cl::inclusive_scan_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), dvIn2Vec.begin(), dvOutVec.begin(), equal_to, addI2);
            /*Compute expected results*/
			bolt::cl::control ctl = bolt::cl::control::getDefault( );
			//ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

            bolt::cl::inclusive_scan_by_key(ctl, svIn1Vec.begin(), svIn1Vec.end(),  svIn2Vec.begin(), stlOut.begin(), equal_to, addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when value input is a constant iterator while key input is stil randomAccessIterator  */
            bolt::cl::inclusive_scan_by_key(svIn1Vec.begin(), svIn1Vec.end(), const_itr_begin, svOutVec.begin(), equal_to, addI2);
            bolt::cl::inclusive_scan_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), const_itr_begin, dvOutVec.begin(), equal_to, addI2);
            /*Compute expected results*/
            std::vector<UDD> const_vector(const_itr_begin, const_itr_end);

			bolt::cl::control ctl = bolt::cl::control::getDefault( );
			//ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

            bolt::cl::inclusive_scan_by_key(ctl, svIn1Vec.begin(), svIn1Vec.end(), const_vector.begin(), stlOut.begin(), equal_to, addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        {/*Test case when input is a counting iterator */
            bolt::cl::inclusive_scan_by_key(svIn1Vec.begin(), svIn1Vec.end(), count_itr_begin, svOutVec.begin(), equal_to, addI2);
            bolt::cl::inclusive_scan_by_key(dvIn1Vec.begin(), dvIn1Vec.end(), count_itr_begin, dvOutVec.begin(), equal_to, addI2);
            /*Compute expected results*/
            /*std::vector<int> count_vector(length);
            for (int index=0;index<length;index++)
                count_vector[index] = index;*/
			std::vector<UDD> count_vector(count_itr_begin , count_itr_end);
			bolt::cl::control ctl = bolt::cl::control::getDefault( );
			//ctl.setForceRunMode(bolt::cl::control::SerialCpu); 

            bolt::cl::inclusive_scan_by_key(ctl, svIn1Vec.begin(), svIn1Vec.end(), count_vector.begin(), stlOut.begin(), equal_to, addI2);
            /*Check the results*/
            cmpArrays(svOutVec, stlOut, length);
            cmpArrays(dvOutVec, stlOut, length);
        }
        global_id = 0; // Reset the global id counter
    }
}

//BUGS

BOLT_FUNCTOR(UDD_trans,
struct UDD_trans
{
    int		i	;
    float	f	;
	UDD_trans ()  
		{
		}; 

	UDD_trans (int val1)  
		{
			i =  val1 ;
		
		}; 

	UDD_trans operator() ()  const 
		{ 
			UDD_trans temp ;
			int a = 0;
			temp.i =  a++;
			return temp ;
			//return get_global_id(0); 
		}
    
};
);

BOLT_FUNCTOR(add_UDD,
  struct add_UDD
    {
        int operator() (const UDD_trans x)  const { return x.i + 3; }
        typedef int result_type;
    };
  );
//  int global_id;

BOLT_TEMPLATE_REGISTER_NEW_ITERATOR( bolt::cl::device_vector, int, UDD_trans);

BOLT_TEMPLATE_REGISTER_NEW_TRANSFORM_ITERATOR( bolt::cl::transform_iterator, add_UDD, UDD_trans);




TEST (TransformIterator, BUG399572)
{
	int length = 5;

	std::vector< UDD_trans > svInVec1( length ); // Input
	std::vector< UDD_trans > svInVec2( length ); // Map
	std::vector< UDD_trans > svInVec3( length ); // Stencil

	std::vector< int > svOutVec( length );
	std::vector< UDD_trans > stlOut(length);

    bolt::cl::device_vector< UDD_trans > dvInVec1( length ); //Input
    bolt::cl::device_vector< UDD_trans > dvInVec2( length ); //Map

	bolt::cl::device_vector< int > dvOutVec1( length );
	bolt::cl::device_vector< UDD_trans > dvOutVec2( length );

	for(int i=0; i<length; i++)
		{
			if(i%2 == 0)
				svInVec3[i] = 0;
			else
				svInVec3[i] = 1;
		}
		bolt::cl::device_vector< UDD_trans > dvInVec3( svInVec3.begin(), svInVec3.end() );


	add_UDD add1;
	UDD_trans gen_udd(0) ;


	// ADD
    bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>::value_type t_v_t;
    t_v_t = 90;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		sv_trf_begin1 (svInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>		sv_trf_end1   (svInVec1.end(),   add1) ;

	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		sv_trf_begin2 (svInVec2.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>		sv_trf_end2   (svInVec2.end(),   add1) ;

	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		sv_trf_begin3 (svInVec3.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>		sv_trf_end3   (svInVec3.end(),   add1) ;

	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  dv_trf_begin1 (dvInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>	dv_trf_end1   (dvInVec1.end(),   add1) ;

	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  dv_trf_begin2 (dvInVec2.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>	dv_trf_end2   (dvInVec2.end(),   add1) ;

	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  dv_trf_begin3 (dvInVec3.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>	dv_trf_end3   (dvInVec3.end(),   add1) ;

	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		t_sv_trf_begin1 (svInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		t_sv_trf_begin2 (svInVec2.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		t_sv_trf_begin3 (svInVec3.begin(), add1) ;

	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  t_sv_trf_begin4 (dvInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  t_sv_trf_begin5 (dvInVec2.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  t_sv_trf_begin6 (dvInVec3.begin(), add1) ;


	std::generate(svInVec1.begin(), svInVec1.end(), gen_udd);

	std::generate(svInVec2.begin(), svInVec2.end(), gen_udd);

	bolt::cl::generate(dvInVec1.begin(), dvInVec1.end(), gen_udd);

	bolt::cl::generate(dvInVec2.begin(), dvInVec2.end(), gen_udd);

	is_even iepred;

	t_sv_trf_begin1 = sv_trf_begin1 ;
	t_sv_trf_begin2 = sv_trf_begin2 ;
	t_sv_trf_begin3 = sv_trf_begin3 ;

	t_sv_trf_begin4 = dv_trf_begin1 ;
	t_sv_trf_begin5 = dv_trf_begin2 ;
	t_sv_trf_begin6 = dv_trf_begin3 ;


	bolt::cl::control ctrl = bolt::cl::control::getDefault();

	bolt::cl::scatter_if(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, sv_trf_begin3, svOutVec.begin(), iepred);
    bolt::cl::scatter_if(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, dv_trf_begin3, dvOutVec1.begin(), iepred);

	for (int i=0; i<length;  i++ )
	{
	    //std::cout << "Val = " << svOutVec[i].i <<  "\n" ;
	    std::cout << "Val --->" << *t_sv_trf_begin1++ << "     "  << *t_sv_trf_begin2++ <<"   " << *t_sv_trf_begin3++ <<"   " << *t_sv_trf_begin4++ << "   " << *t_sv_trf_begin5++ <<"  " << *t_sv_trf_begin6++ <<"\n" ;

	}

}



TEST(transform_iterator, BUG400107)
{
//	int length =  1<<20;
	int length =  5;
    
	std::vector< UDD_trans > svInVec( length );
	std::vector< int > svOutVec( length );
	std::vector< int > stlOut(length);

    bolt::cl::device_vector< UDD_trans > svInVec1( length );
    bolt::cl::device_vector< UDD_trans > svInVec2( length );

	bolt::cl::device_vector< int > svOutVec1( length );
	bolt::cl::device_vector< int > svOutVec2( length );

	add_UDD add1;
	UDD_trans gen_udd(0) ;

	// ADD
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		sv_trf_begin1 (svInVec.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>		sv_trf_end1   (svInVec.end(),   add1) ;
																			 
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  	sv_trf_begin2 (svInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>		sv_trf_end2   (svInVec1.end(),   add1) ;
	
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  			t_sv_trf_begin1 (svInVec.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  	t_sv_trf_begin2 (svInVec1.begin(), add1) ;


	global_id = 0;
	std::generate(svInVec.begin(), svInVec.end(), gen_udd);

	global_id = 0;
	bolt::cl::generate(svInVec1.begin(), svInVec1.end(), gen_udd);

	//bolt::cl::square<int> sq;
	bolt::cl::plus<int> pls;
	//bolt::cl::minus<int> mi;
	bolt::cl::negate<int> neg;

	t_sv_trf_begin1 = sv_trf_begin1 ;
	t_sv_trf_begin2 = sv_trf_begin2 ;


	bolt::cl::control ctrl = bolt::cl::control::getDefault();
	global_id = 0;
		

	bolt::cl::transform_inclusive_scan(	  sv_trf_begin1, sv_trf_end1,  svOutVec.begin(), neg, pls);
	bolt::cl::transform_inclusive_scan(ctrl, sv_trf_begin2, sv_trf_end2,  svOutVec1.begin(), neg, pls);
	
	//STD_TRANSFORM_SCAN
	std::transform(sv_trf_begin1, sv_trf_end1, stlOut.begin(), std::negate<int>());
	std::partial_sum(stlOut.begin(), stlOut.end(), stlOut.begin(), std::plus<int>());

	for (int i=0; i<length;  i++ )
	{
		std::cout << "Val " << svInVec[i].i << "     "  << *t_sv_trf_begin1++ <<"   " << *t_sv_trf_begin2++ <<  "Out VAL ---> " << "  " << svOutVec[i] <<"   " << stlOut[i] << "\n" ;
	} 
	  
	for(int i =0; i< length; i++)
    {
        EXPECT_EQ( svOutVec[i], stlOut[i] );
        EXPECT_EQ( svOutVec1[i], stlOut[i] );
	}
	  
}


TEST (transform_iterator, BUG400103){
	bolt::cl::square<int> sqInt; 
	bolt::cl::plus<int> plInt; 
	int a[10] = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10}; 
	int b[10] = {1, 5, 14, 30, 55, 91, 140, 204, 285, 385};
	
	bolt::cl::transform_inclusive_scan( a, a+10, a, sqInt, plInt ); 

	for (int i = 0 ; i < 10 ; ++i){
		EXPECT_EQ(b[i], a[i])<<std::endl;
	}
}

TEST (transform_iterator, BUG400294){
 const int N = 7;
  int A[N] = {1, 3, 3, 3, 2, 2, 1}; // input keys
  int B[N] = {9, 8, 7, 6, 5, 4, 3}; // input values
  int C[N];                         // output keys
  int D[N];                         // output values

  bolt::cl::pair<int*,int*> new_end;
  bolt::cl::equal_to<int> binary_pred;
  bolt::cl::plus<int> binary_op;

  //new_end = bolt::cl::reduce_by_key(A, A + N, B, C, D, binary_pred, binary_op);
  bolt::cl::reduce_by_key(A, A + N, B, C, D, binary_pred, binary_op);
  
  int C_exp[N] = {1, 3, 2, 1};
  int D_exp[N] = {9, 21, 9, 3};
}

TEST ( transform_iterator, BUG400110)
{
//	int length =  1<<20;
	int length =  5;
    
	std::vector< UDD_trans > svInVec1( length );
	std::vector< UDD_trans > svInVec2( length );

	std::vector< int > svOutVec1( length );
	std::vector< int > svOutVec2(length);
				 
	std::vector< int > stlOut1( length );
	std::vector< int > stlOut2( length );

    bolt::cl::device_vector< UDD_trans > dvInVec1( length );
    bolt::cl::device_vector< UDD_trans > dvInVec2( length );
							 
	bolt::cl::device_vector< int > dvOutVec1( length );
	bolt::cl::device_vector< int > dvOutVec2( length );

	add_UDD add1;
	UDD_trans gen_udd(0) ;
	
	// ADD
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		sv_trf_begin1 (svInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>		sv_trf_end1   (svInVec1.end(),   add1) ;
	
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		sv_trf_begin2 (svInVec2.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>		sv_trf_end2   (svInVec2.end(),   add1) ;

	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  sv_trf_begin3 (dvInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>	sv_trf_end3   (dvInVec1.end(),   add1) ;
	
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  sv_trf_begin4 (dvInVec2.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>	sv_trf_end4   (dvInVec2.end(),   add1) ;

	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		t_sv_trf_begin1 (svInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		t_sv_trf_begin2 (svInVec2.begin(), add1) ;
	
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  t_sv_trf_begin3 (dvInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  t_sv_trf_begin4 (dvInVec2.begin(), add1) ;


	global_id = 0;
	std::generate(svInVec1.begin(), svInVec1.end(), gen_udd);

	global_id = 0;
	std::generate(svInVec2.begin(), svInVec2.end(), gen_udd);
	
	global_id = 0;
	bolt::cl::generate(dvInVec1.begin(), dvInVec1.end(), gen_udd);

	global_id = 0;
	bolt::cl::generate(dvInVec2.begin(), dvInVec2.end(), gen_udd);

	//bolt::cl::square<int> sq;
	bolt::cl::plus<int> pls;
	//bolt::cl::minus<int> mi;
	//bolt::cl::negate<int> neg;
	 bolt::cl::equal_to<int> eql;

	t_sv_trf_begin1 = sv_trf_begin1 ;
	t_sv_trf_begin2 = sv_trf_begin2 ;

	t_sv_trf_begin3 = sv_trf_begin3 ;
	t_sv_trf_begin4 = sv_trf_begin4 ;

	std::vector< UDD_trans > testInput1( svInVec1.begin(), svInVec1.end() );
    std::vector< int > testInput_int1( length );
	std::vector< UDD_trans > testInput2( svInVec2.begin(), svInVec2.end() );
    std::vector< int > testInput_int2( length );
	for(int i=0; i<length; i++)
	{
		testInput_int1[i] = testInput1[i].i + 3;
		testInput_int2[i] = testInput2[i].i + 3;
	}

	bolt::cl::control ctrl = bolt::cl::control::getDefault();
	global_id = 0;
	
    bolt::cl::reduce_by_key(		sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec1.begin(), svOutVec2.begin(), eql, pls);
    bolt::cl::reduce_by_key(ctrl, sv_trf_begin3, sv_trf_end3, sv_trf_begin4, dvOutVec1.begin(), dvOutVec2.begin(), eql, pls);

	//STD_REDUCED
	unsigned int n= Serial_reduce_by_key<int, int, int, int, bolt::cl::plus< int >> (&testInput_int1[0], &testInput_int2[0], &stlOut1[0], &stlOut2[0], pls, length);

	for(int i =0; i< length; i++)
    {
        EXPECT_EQ( svOutVec1[i], stlOut1[i] );
        EXPECT_EQ( svOutVec2[i], stlOut2[i] );
        EXPECT_EQ( dvOutVec1[i], stlOut1[i] );
        EXPECT_EQ( dvOutVec2[i], stlOut2[i] );	
        EXPECT_EQ( svOutVec1[i], dvOutVec1[i] );
        EXPECT_EQ( svOutVec2[i], dvOutVec2[i] );
	}
}

TEST (transform_iterator, BUG400109){

    int length =  1<<8;
    
	std::vector< UDD_trans > svInVec1( length );
	std::vector< UDD_trans > svInVec2( length );
	
	std::vector< int > svOutVec( length );
	std::vector< int > stlOut(length);

    
							 
	bolt::cl::device_vector< int > dvOutVec1( length );
	bolt::cl::device_vector< int > dvOutVec2( length );

	add_UDD add1;
	UDD_trans gen_udd(0) ;
	int init = 0;

	// ADD
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		sv_trf_begin1 (svInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>		sv_trf_end1   (svInVec1.end(),   add1) ;

	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		sv_trf_begin2 (svInVec2.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>		sv_trf_end2   (svInVec2.end(),   add1) ;
								 
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		t_sv_trf_begin1 (svInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, std::vector< UDD_trans >::const_iterator>  		t_sv_trf_begin2 (svInVec2.begin(), add1) ;
								  
	
	global_id = 0;
	std::generate(svInVec1.begin(), svInVec1.end(), gen_udd);

	global_id = 0;
	std::generate(svInVec2.begin(), svInVec2.end(), rand);

	global_id = 0;
	bolt::cl::device_vector< UDD_trans > dvInVec1( svInVec1.begin(), svInVec1.end());

	global_id = 0;
	bolt::cl::device_vector< UDD_trans > dvInVec2( svInVec2.begin(), svInVec2.end() );

	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator> 	    dv_trf_begin1 (dvInVec1.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>	    dv_trf_end1   (dvInVec1.end(),   add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  	t_sv_trf_begin3 (dvInVec1.begin(), add1) ;

	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  	dv_trf_begin2 (dvInVec2.begin(), add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>		dv_trf_end2   (dvInVec2.end(),   add1) ;
	bolt::cl::transform_iterator< add_UDD, bolt::cl::device_vector< UDD_trans >::iterator>  	t_sv_trf_begin4 (dvInVec2.begin(), add1) ;

	bolt::cl::plus<int> pls;
	bolt::cl::minus<int> mi;

	t_sv_trf_begin1 = sv_trf_begin1 ;
	t_sv_trf_begin2 = sv_trf_begin2 ;
	t_sv_trf_begin3 = dv_trf_begin1 ;
	t_sv_trf_begin4 = dv_trf_begin2 ;

	global_id = 0;
		
	
	int sv_result = bolt::cl::inner_product(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, init, pls, mi);
    int dv_result = bolt::cl::inner_product(dv_trf_begin1, dv_trf_end1, dv_trf_begin2, init, pls, mi);
	
	std::transform(sv_trf_begin1, sv_trf_end1, sv_trf_begin2, svOutVec.begin(), mi);
    int expected_result = std::accumulate(svOutVec.begin(), svOutVec.end(), init, pls);

	EXPECT_EQ( sv_result, expected_result );
    EXPECT_EQ( dv_result, expected_result);
			       
}
/* /brief List of possible tests
 * Two input transform with first input a constant iterator
 * One input transform with a constant iterator
*/
int _tmain(int argc, _TCHAR* argv[])
{
    //  Register our minidump generating logic
    //bolt::miniDumpSingleton::enableMiniDumps( );

    //  Initialize googletest; this removes googletest specific flags from command line
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    bool print_clInfo = false;
    cl_uint userPlatform = 0;
    cl_uint userDevice = 0;
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;

    try
    {
        // Declare supported options below, describe what they do
        po::options_description desc( "Scan GoogleTest command line options" );
        desc.add_options()
            ( "help,h",         "produces this help message" )
            ( "queryOpenCL,q",  "Print queryable platform and device info and return" )
            ( "platform,p",     po::value< cl_uint >( &userPlatform )->default_value( 0 ),    
            "Specify the platform under test" )
            ( "device,d",       po::value< cl_uint >( &userDevice )->default_value( 0 ),    
            "Specify the device under test" )
            //( "gpu,g",         "Force instantiation of all OpenCL GPU device" )
            //( "cpu,c",         "Force instantiation of all OpenCL CPU device" )
            //( "all,a",         "Force instantiation of all OpenCL devices" )
            ;

        ////  All positional options (un-named) should be interpreted as kernelFiles
        //po::positional_options_description p;
        //p.add("kernelFiles", -1);

        //po::variables_map vm;
        //po::store( po::command_line_parser( argc, argv ).options( desc ).positional( p ).run( ), vm );
        //po::notify( vm );

        po::variables_map vm;
        po::store( po::parse_command_line( argc, argv, desc ), vm );
        po::notify( vm );

        if( vm.count( "help" ) )
        {
            //    This needs to be 'cout' as program-options does not support wcout yet
            std::cout << desc << std::endl;
            return 0;
        }

        if( vm.count( "queryOpenCL" ) )
        {
            print_clInfo = true;
        }

        //  The following 3 options are not implemented yet; they are meant to be used with ::clCreateContextFromType()
        if( vm.count( "gpu" ) )
        {
            deviceType    = CL_DEVICE_TYPE_GPU;
        }

        if( vm.count( "cpu" ) )
        {
            deviceType    = CL_DEVICE_TYPE_CPU;
        }

        if( vm.count( "all" ) )
        {
            deviceType    = CL_DEVICE_TYPE_ALL;
        }

    }
    catch( std::exception& e )
    {
        std::cout << _T( "Scan GoogleTest error condition reported:" ) << std::endl << e.what() << std::endl;
        return 1;
    }

    //  Query OpenCL for available platforms
    cl_int err = CL_SUCCESS;

    // Platform vector contains all available platforms on system
    std::vector< cl::Platform > platforms;
    //std::cout << "HelloCL!\nGetting Platform Information\n";
    bolt::cl::V_OPENCL( cl::Platform::get( &platforms ), "Platform::get() failed" );

    if( print_clInfo )
    {
        bolt::cl::control::printPlatforms( );
        return 0;
    }

    //  Do stuff with the platforms
    std::vector<cl::Platform>::iterator i;
    if(platforms.size() > 0)
    {
        for(i = platforms.begin(); i != platforms.end(); ++i)
        {
            if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str(), "Advanced Micro Devices, Inc."))
            {
                break;
            }
        }
    }
    bolt::cl::V_OPENCL( err, "Platform::getInfo() failed" );

    // Device info
    std::vector< cl::Device > devices;
    bolt::cl::V_OPENCL( platforms.front( ).getDevices( CL_DEVICE_TYPE_ALL, &devices ),"Platform::getDevices()failed" );

    cl::Context myContext( devices.at( userDevice ) );
    cl::CommandQueue myQueue( myContext, devices.at( userDevice ) );
    bolt::cl::control::getDefault( ).setCommandQueue( myQueue );

    std::string strDeviceName = bolt::cl::control::getDefault( ).getDevice( ).getInfo< CL_DEVICE_NAME >( &err );
    bolt::cl::V_OPENCL( err, "Device::getInfo< CL_DEVICE_NAME > failed" );

    std::cout << "Device under test : " << strDeviceName << std::endl;

    int retVal = RUN_ALL_TESTS( );

    bolt::cl::control::getDefault( ).setForceRunMode(bolt::cl::control::SerialCpu); 
    retVal = RUN_ALL_TESTS( );

    bolt::cl::control::getDefault( ).setForceRunMode(bolt::cl::control::MultiCoreCpu); 
	retVal = RUN_ALL_TESTS( );

    //  Reflection code to inspect how many tests failed in gTest
    ::testing::UnitTest& unitTest = *::testing::UnitTest::GetInstance( );

    unsigned int failedTests = 0;
    for( int i = 0; i < unitTest.total_test_case_count( ); ++i )
    {
        const ::testing::TestCase& testCase = *unitTest.GetTestCase( i );
        for( int j = 0; j < testCase.total_test_count( ); ++j )
        {
            const ::testing::TestInfo& testInfo = *testCase.GetTestInfo( j );
            if( testInfo.result( )->Failed( ) )
                ++failedTests;
        }
    }

    //  Print helpful message at termination if we detect errors, to help users figure out what to do next
    if( failedTests )
    {
        bolt::tout << _T( "\nFailed tests detected in test pass; please run test again with:" ) << std::endl;
        bolt::tout << _T( "\t--gtest_filter=<XXX> to select a specific failing test of interest" ) << std::endl;
        bolt::tout << _T( "\t--gtest_catch_exceptions=0 to generate minidump of failing test, or" ) << std::endl;
        bolt::tout << _T( "\t--gtest_break_on_failure to debug interactively with debugger" ) << std::endl;
        bolt::tout << _T( "\t    (only on googletest assertion failures, not SEH exceptions)" ) << std::endl;
    }

    return retVal;
}
