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

// PairTest.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include <bolt/amp/pair.h>
#include <bolt/amp/functional.h>
//#include <bolt/amp/control.h>
#include <gtest/gtest.h>
#include <cassert>

#include <iostream>
#include <algorithm>  // for testing against STL functions.
#include <numeric>
//#include "common/myocl.h"

///////////////////////////////////////////
/// Pair Manipulation Test
///////////////////////////////////////////

template <typename T>
class ConstructPair : public ::testing::Test {
    public:
         bolt::amp::pair<T,T>x,y;    

    ConstructPair()
    {
         x.first  = T(1.0);
         x.second = T(2.0);
         y.first  = T(1.0);
         y.second = T(2.0);
    }

    public:
    void DoPair(std::pair<T,T> stdx)
    {
        x = stdx;
        y = stdx;
    }

};


template <typename T>
class ValuePair : public ::testing::Test
{
    public:
         bolt::amp::pair<T,T>x,y;    
    ValuePair()
    {
            x.first  = T(1);
            y.second = T(2);
    }

    public:

    bolt::amp::pair<T,T> ValueMakePair(T value1, T value2)
    {
        x.first = value1;
        x.second = value2;
        return bolt::amp::make_pair<T,T>(value1, value2);
    }


};

template <typename T>
class PairOperators : public ::testing::Test 
{
    public:
         bolt::amp::pair<T,T>x,y;    
    PairOperators()
    {
        x.first = T(10);
        x.second = T(20);
        y.first = T(10);
        y.second = T(20);
                
    }
    public:
    void MakeXLess()
    {
        x.first = T(1);
        x.second = T(2);
        y.first = T(10);
        y.second = T(20);

    }

    void MakeYLess()
    {
        x.first = T(10);
        x.second = T(20);
        y.first = T(1);
        y.second = T(2);

    }

    void MakeXYEqual()
    {
        x.first = T(10);
        x.second = T(10);
        y.first = T(10);
        y.second = T(10);

    }



};


typedef ::testing::Types<int, float, double> AllTypes;
typedef ::testing::Types<int> Integer;
typedef ::testing::Types<double> DoubleType;
//typedef ::testing::Types <<int, float>, <float,int>, <int, double>> ComboTypes;
TYPED_TEST_CASE(ValuePair, AllTypes);
TYPED_TEST_CASE(ConstructPair, Integer);
TYPED_TEST_CASE(PairOperators, AllTypes);

TYPED_TEST(ValuePair, IntegerIntegerPair)
{
    
    
    //Value test
    EXPECT_EQ(1, this->x.first);
    EXPECT_EQ(2, this->y.second);

    // == op
    EXPECT_EQ(false, this->y.second == this->x.first);

    //!= op
    EXPECT_EQ(true, this->y.second != this->x.first);

    EXPECT_EQ(this->x, ValuePair< gtest_TypeParam_ >::ValueMakePair(10,20));


}

TYPED_TEST(ConstructPair, ConstructPair)
{
    // Inits
    EXPECT_EQ(this->x.first,  this->y.first);
    EXPECT_EQ(this->x.second, this->y.second);

    //Test with std constructors
    typedef bolt::amp::pair<int,int> P;

    P p1;
    P p2(p1);
    EXPECT_EQ(p1.first,  p2.first);
    EXPECT_EQ(p1.second, p2.second);

    std::pair<int,int> sp(p1.first, p1.second);
    EXPECT_EQ(p1.first,  sp.first);
    EXPECT_EQ(p1.second, sp.second);

    sp.first = 10;
    sp.second = 20;
    ConstructPair< gtest_TypeParam_ >::DoPair(sp);
    EXPECT_EQ(this->x.first,  sp.first );
    EXPECT_EQ(this->y.second, sp.second);
    
}


TYPED_TEST(PairOperators, OperatorTests)
{
    // < >
    EXPECT_EQ(false, this->x < this->y);
    EXPECT_EQ(false, this->y < this->x);

    
    EXPECT_EQ(false, this->x < this->y);
    EXPECT_EQ(false, this->y < this->x);
    
    
    PairOperators< gtest_TypeParam_ >::MakeXLess();
    //std::cout<<(this->x<this->y)<<" Here"<<std::endl;
    EXPECT_EQ(true,  this->x < this->y);
    EXPECT_EQ(false, this->y < this->x);
    
    PairOperators< gtest_TypeParam_ >::MakeYLess();
    //std::cout<<(this->x<this->y)<<" Here"<<std::endl;
    EXPECT_EQ(true,  this->x > this->y);
    EXPECT_EQ(false, this->y > this->x);
    
    
    EXPECT_EQ(true, this->x != this->y);
    EXPECT_EQ(true, this->y != this->x);

    // <=
    PairOperators< gtest_TypeParam_ >::MakeXYEqual();
    EXPECT_EQ(true, this->x <= this->y);
    EXPECT_EQ(true, this->y <= this->x);

    PairOperators< gtest_TypeParam_ >::MakeYLess();
    EXPECT_EQ(false, this->x <= PairOperators< gtest_TypeParam_ >::y);

    EXPECT_EQ(false, this-> x <= this->y);
    EXPECT_EQ(true,  this-> y <= this->x);

    // >=
   PairOperators< gtest_TypeParam_ >::MakeXYEqual();
    EXPECT_EQ(true, this-> x >=  this->y);
    EXPECT_EQ(true, this-> y >=  this->x);

    PairOperators< gtest_TypeParam_ >::MakeYLess();
    EXPECT_EQ(true,  PairOperators< gtest_TypeParam_ >::x >= PairOperators< gtest_TypeParam_ >::y);
    EXPECT_EQ(false, PairOperators< gtest_TypeParam_ >::y >= PairOperators< gtest_TypeParam_ >::x);

    PairOperators< gtest_TypeParam_ >::MakeXLess();
    EXPECT_EQ(false,  PairOperators< gtest_TypeParam_ >::x >= PairOperators< gtest_TypeParam_ >::y);
    EXPECT_EQ(true, PairOperators< gtest_TypeParam_ >::y >= PairOperators< gtest_TypeParam_ >::x);

}


int _tmain(int argc, _TCHAR* argv[])
{
    std::cout<<"All done"<<std::endl;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    return 0;
}

