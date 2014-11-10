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

#define TEST_DOUBLE 1
#define TEST_DEVICE_VECTOR 1
#define TEST_CPU_DEVICE 0
#define GOOGLE_TEST 1
#if (GOOGLE_TEST == 1)
#include "common/stdafx.h"
#include "common/myocl.h"
#include "bolt/cl/iterator/counting_iterator.h"

#include "bolt/cl/count.h"
#include "bolt/cl/functional.h"
#include "bolt/miniDump.h"
#include "bolt/unicode.h"
#include "bolt/cl/device_vector.h"

#include <gtest/gtest.h>
#include <boost/shared_array.hpp>
#include <array>
#include <algorithm>
class testCountIfFloatWithStdVector: public ::testing::TestWithParam<int>{
protected:
  int aSize;
public:
  testCountIfFloatWithStdVector():aSize(GetParam()){
  }

};


BOLT_TEMPLATE_FUNCTOR4(InRange,int,float,double,long long,
template<typename T>
// Functor for range checking.
struct InRange {
  InRange (T low, T high) {
    _low=low;
    _high=high;
  };

  bool operator() (const T& value) {
    return (value >= _low) && (value <= _high) ;
  };

  T _low;
  T _high;
};
);
//
//BOLT_CREATE_TYPENAME(InRange<int>);
//BOLT_CREATE_CLCODE(InRange<int>, InRange_CodeString);
//BOLT_CREATE_TYPENAME(InRange<float>);
//BOLT_CREATE_CLCODE(InRange<float>, InRange_CodeString);
//BOLT_CREATE_TYPENAME(InRange<double>);
//BOLT_CREATE_CLCODE(InRange<double>, InRange_CodeString);
//BOLT_CREATE_TYPENAME(InRange<__int64>);
//BOLT_CREATE_CLCODE(InRange<__int64>, InRange_CodeString);
//

TEST (testCountIf, OffsetintBtwRange)
{
    int aSize = 1024;
    std::vector<int> A(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = rand() % 10 + 1;
    }
  bolt::cl::device_vector< int > dA(A.begin(), aSize);
  int intVal = 1;

  int offset =  1+ rand()%(aSize-1);

   std::iterator_traits<std::vector<int>::iterator>::difference_type stdInRangeCount =
                                                                std::count( A.begin()+offset, A.end(), intVal ) ;
    bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type boltInRangeCount =
                                                        bolt::cl::count( dA.begin()+offset, dA.end(), intVal ) ;

    EXPECT_EQ(stdInRangeCount, boltInRangeCount);
}

TEST (testCountIf, SerialOffsetintBtwRange)
{
    int aSize = 1024;
    std::vector<int> A(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = rand() % 10 + 1;
    }
    bolt::cl::device_vector< int > dA(A.begin(), aSize);
    int intVal = 1;

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    int offset = 1+rand()%(aSize-1);

    std::iterator_traits<std::vector<int>::iterator>::difference_type stdInRangeCount =
                                                                std::count( A.begin()+offset, A.end(), intVal ) ;
    bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type boltInRangeCount =
                                                        bolt::cl::count( ctl, dA.begin()+offset, dA.end(), intVal ) ;

    EXPECT_EQ(stdInRangeCount, boltInRangeCount);
}

TEST (testCountIf, MultiCoreOffsetintBtwRange)
{
    int aSize = 1024;
    std::vector<int> A(aSize);

    for (int i=0; i < aSize; i++) {
        A[i] = rand() % 10 + 1;
    }
    bolt::cl::device_vector< int > dA(A.begin(), aSize);
    int intVal = 1;

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    int offset = 1+rand()%(aSize-1);

   std::iterator_traits<std::vector<int>::iterator>::difference_type stdInRangeCount =
                                                                std::count( A.begin()+offset, A.end(), intVal ) ;
    bolt::cl::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type boltInRangeCount =
                                                        bolt::cl::count( ctl, dA.begin()+offset, dA.end(), intVal ) ;

    EXPECT_EQ(stdInRangeCount, boltInRangeCount);
}

TEST_P (testCountIfFloatWithStdVector, countFloatValueInRange)
{
   std::vector<float> s(aSize);

  for (int i=0; i < aSize; i++) {
    s[i] = static_cast<float> (i+1);
  };

  bolt::cl::device_vector<float> A(s.begin(),s.end());
  bolt::cl::device_vector<float> B(s.begin(),s.end());


  size_t stdCount = std::count_if (A.begin(), A.end(), InRange<float>(6,10)) ;
  size_t boltCount = bolt::cl::count_if (B.begin(), B.end(), InRange<float>(6,10)) ;

  EXPECT_EQ (stdCount, boltCount)<<"Failed as: STD Count = "<<stdCount<<"\nBolt Count = "<<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"bolt Count = "<<boltCount<<std::endl;
}

TEST_P (testCountIfFloatWithStdVector, countFloatValueInRange2)
{
   std::vector<float> s(aSize);

  for (int i=0; i < aSize; i++) {
    s[i] = static_cast<float> (i+1);
  };

  bolt::cl::device_vector<float> A(s.begin(),s.end());
  bolt::cl::device_vector<float> B(s.begin(),s.end());

  size_t stdCount = std::count_if (A.begin(), A.end(), InRange<float>(1,10)) ;
  size_t boltCount = bolt::cl::count_if (B.begin(), B.end(), InRange<float>(1,10)) ;

  EXPECT_EQ (stdCount, boltCount)<<"Failed as: STD Count = "<<stdCount<<"\nBolt Count = "<<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"bolt Count = "<<boltCount<<std::endl;
}

//
//Test case id: 6 (FAILED)

class countFloatValueOccuranceStdVect :public ::testing::TestWithParam<int>{
protected:
  int stdVectSize;
public:
  countFloatValueOccuranceStdVect():stdVectSize(GetParam()){}
};

class StdVectCountingIterator :public ::testing::TestWithParam<int>{
protected:
    int mySize;
public:
    StdVectCountingIterator():mySize(GetParam()){
    }
};

TEST_P( StdVectCountingIterator, withCountingIterator)
{
    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    int myValue = 3;

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    size_t stdCount = std::count(a.begin(), a.end(), myValue);
    size_t boltCount = bolt::cl::count(first, last, myValue);

    EXPECT_EQ(stdCount, boltCount);
}


TEST_P( StdVectCountingIterator, SerialwithCountingIterator)
{
    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    int myValue = 3;

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);

    size_t stdCount = std::count(a.begin(), a.end(), myValue);
    size_t boltCount = bolt::cl::count(ctl, first, last, myValue);

    EXPECT_EQ(stdCount, boltCount);
}

TEST_P( StdVectCountingIterator, MultiCorewithCountingIterator)
{
    bolt::cl::counting_iterator<int> first(0);
    bolt::cl::counting_iterator<int> last = first +  mySize;

    std::vector<int> a(mySize);

    int myValue = 3;

    for (int i=0; i < mySize; i++) {
        a[i] = i;
    };

    bolt::cl::control ctl = bolt::cl::control::getDefault( );
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

    size_t stdCount = std::count(a.begin(), a.end(), myValue);
    size_t boltCount = bolt::cl::count(ctl, first, last, myValue);

    EXPECT_EQ(stdCount, boltCount);
}

TEST_P(countFloatValueOccuranceStdVect, floatVectSearchWithSameValue){
 
    float myFloatValue = 1.23f;
    std::vector<float> stdVect(stdVectSize,myFloatValue);
    bolt::cl::device_vector<float> boltVect(stdVect.begin(), stdVect.end());

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myFloatValue);
  size_t boltCount = bolt::cl::count(boltVect.begin(), boltVect.end(), myFloatValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<
      "Bolt Count = "<<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST_P(countFloatValueOccuranceStdVect, Serial_floatVectSearchWithSameValue){

    float myFloatValue = 1.23f;
    std::vector<float> stdVect(stdVectSize,myFloatValue);
    bolt::cl::device_vector<float> boltVect(stdVect.begin(), stdVect.end());


  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::SerialCpu);

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myFloatValue);
  size_t boltCount = bolt::cl::count(ctl, boltVect.begin(), boltVect.end(), myFloatValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST_P(countFloatValueOccuranceStdVect, MultiCore_floatVectSearchWithSameValue){
    float myFloatValue = 1.23f;
    std::vector<float> stdVect(stdVectSize,myFloatValue);
    bolt::cl::device_vector<float> boltVect(stdVect.begin(), stdVect.end());


  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);



  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myFloatValue);
  size_t boltCount = bolt::cl::count(ctl, boltVect.begin(), boltVect.end(), myFloatValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST_P(countFloatValueOccuranceStdVect, floatVectSearchWithSameValue2){
    float myFloatValue2 = 9.87f;
    std::vector<float> stdVect(stdVectSize,myFloatValue2);
    bolt::cl::device_vector<float> boltVect(stdVect.begin(), stdVect.end());

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myFloatValue2);
  size_t boltCount = bolt::cl::count(boltVect.begin(), boltVect.end(), myFloatValue2);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST_P(countFloatValueOccuranceStdVect, Serial_floatVectSearchWithSameValue2){
    float myFloatValue2 = 9.87f;
    std::vector<float> stdVect(stdVectSize,myFloatValue2);
    bolt::cl::device_vector<float> boltVect(stdVect.begin(), stdVect.end());

  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::SerialCpu);

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myFloatValue2);
  size_t boltCount = bolt::cl::count(ctl, boltVect.begin(), boltVect.end(), myFloatValue2);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST_P(countFloatValueOccuranceStdVect, MultiCore_floatVectSearchWithSameValue2){
    float myFloatValue2 = 9.87f;
    std::vector<float> stdVect(stdVectSize,myFloatValue2);
    bolt::cl::device_vector<float> boltVect(stdVect.begin(), stdVect.end());

  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myFloatValue2);
  size_t boltCount = bolt::cl::count(ctl, boltVect.begin(), boltVect.end(), myFloatValue2);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}


INSTANTIATE_TEST_CASE_P (useStdVectWithFloatValues, countFloatValueOccuranceStdVect,
                         ::testing::Values(1, 100, 1000, 10000, 100000));
INSTANTIATE_TEST_CASE_P (useStdVectWithIntValues, StdVectCountingIterator,
                         ::testing::Values(1, 100, 1000, 10000, 100000));

//Test case id: 7 (Failed)
class countDoubleValueUsedASKeyInStdVect :public ::testing::TestWithParam<int>{
protected:
  int stdVectSize;
public:
  countDoubleValueUsedASKeyInStdVect():stdVectSize(GetParam()){}
};


TEST_P(countDoubleValueUsedASKeyInStdVect, doubleVectSearchWithSameValue){
  double myDoubleValueAsKeyValue = 1.23456789l;
   std::vector<double> stdVect(stdVectSize,myDoubleValueAsKeyValue);
  bolt::cl::device_vector<double> boltVect(stdVect.begin(), stdVect.end());


  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myDoubleValueAsKeyValue);
  size_t boltCount = bolt::cl::count(boltVect.begin(), boltVect.end(), myDoubleValueAsKeyValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST_P(countDoubleValueUsedASKeyInStdVect, Serial_doubleVectSearchWithSameValue){
  double myDoubleValueAsKeyValue = 1.23456789l;
   std::vector<double> stdVect(stdVectSize,myDoubleValueAsKeyValue);
  bolt::cl::device_vector<double> boltVect(stdVect.begin(), stdVect.end());


  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::SerialCpu);

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myDoubleValueAsKeyValue);
  size_t boltCount = bolt::cl::count(ctl, boltVect.begin(), boltVect.end(), myDoubleValueAsKeyValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}


TEST_P(countDoubleValueUsedASKeyInStdVect, MultiCore_doubleVectSearchWithSameValue){

  double myDoubleValueAsKeyValue = 1.23456789l;
   std::vector<double> stdVect(stdVectSize,myDoubleValueAsKeyValue);
  bolt::cl::device_vector<double> boltVect(stdVect.begin(), stdVect.end());


  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myDoubleValueAsKeyValue);
  size_t boltCount = bolt::cl::count(ctl, boltVect.begin(), boltVect.end(), myDoubleValueAsKeyValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST_P(countDoubleValueUsedASKeyInStdVect, doubleVectSearchWithSameValue2){
  double myDoubleValueAsKeyValue2 = 9.876543210123456789l;
   std::vector<double> stdVect(stdVectSize,myDoubleValueAsKeyValue2);
  bolt::cl::device_vector<double> boltVect(stdVect.begin(), stdVect.end());

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myDoubleValueAsKeyValue2);
  size_t boltCount = bolt::cl::count(boltVect.begin(), boltVect.end(), myDoubleValueAsKeyValue2);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST_P(countDoubleValueUsedASKeyInStdVect, Serial_doubleVectSearchWithSameValue2){
  double myDoubleValueAsKeyValue2 = 9.876543210123456789l;
   std::vector<double> stdVect(stdVectSize,myDoubleValueAsKeyValue2);
  bolt::cl::device_vector<double> boltVect(stdVect.begin(), stdVect.end());


  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::SerialCpu);

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myDoubleValueAsKeyValue2);
  size_t boltCount = bolt::cl::count(ctl, boltVect.begin(), boltVect.end(), myDoubleValueAsKeyValue2);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST_P(countDoubleValueUsedASKeyInStdVect, MultiCore_doubleVectSearchWithSameValue2){
  double myDoubleValueAsKeyValue2 = 9.876543210123456789l;
   std::vector<double> stdVect(stdVectSize,myDoubleValueAsKeyValue2);
  bolt::cl::device_vector<double> boltVect(stdVect.begin(), stdVect.end());


  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

  size_t stdCount = std::count(stdVect.begin(), stdVect.end(), myDoubleValueAsKeyValue2);
  size_t boltCount = bolt::cl::count(ctl, boltVect.begin(), boltVect.end(), myDoubleValueAsKeyValue2);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

//test case: 1, test: 1
TEST (testCountIf, intBtwRange)
{
  int aSize = 1024;
  std::vector<int> s(aSize);
  for (int i=0; i < aSize; i++) {
    s[i] = rand() % 10 + 1;
  }

  bolt::cl::device_vector<int> A(s.begin(),s.end());


  std::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type stdInRangeCount = std::count_if
      (A.begin(), A.end(), InRange<int>(1,10)) ;
  std::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type boltInRangeCount = bolt::cl::count_if
      (A.begin(), A.end(), InRange<int>(1, 10)) ;

  EXPECT_EQ(stdInRangeCount, boltInRangeCount);
}

TEST (testCountIf, Serial_intBtwRange)
{
  int aSize = 1024;
  std::vector<int> s(aSize);
  for (int i=0; i < aSize; i++) {
    s[i] = rand() % 10 + 1;
  }

  bolt::cl::device_vector<int> A(s.begin(),s.end());

  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::SerialCpu);

  std::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type stdInRangeCount = std::count_if
      (A.begin(), A.end(), InRange<int>(1,10)) ;
  std::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type boltInRangeCount = bolt::cl::count_if
      (ctl, A.begin(), A.end(), InRange<int>(1, 10)) ;

  EXPECT_EQ(stdInRangeCount, boltInRangeCount);
}

TEST (testCountIf, MultiCore_intBtwRange)
{
  int aSize = 1024;
  std::vector<int> s(aSize);
  for (int i=0; i < aSize; i++) {
    s[i] = rand() % 10 + 1;
  }

  bolt::cl::device_vector<int> A(s.begin(),s.end());


  ::cl::Context myContext = bolt::cl::control::getDefault( ).getContext( );
  bolt::cl::control ctl = bolt::cl::control::getDefault( );
  ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);

  std::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type stdInRangeCount = std::count_if
      (A.begin(), A.end(), InRange<int>(1,10)) ;
  std::iterator_traits<bolt::cl::device_vector<int>::iterator>::difference_type boltInRangeCount = bolt::cl::count_if
      (ctl, A.begin(), A.end(), InRange<int>(1, 10)) ;

  EXPECT_EQ(stdInRangeCount, boltInRangeCount);
}


BOLT_FUNCTOR(UDD,
struct UDD {
    int a;
    int b;

    bool operator() (const UDD& lhs, const UDD& rhs) {
        return ((lhs.a+lhs.b) > (rhs.a+rhs.b));
    }
    bool operator < (const UDD& other) const {
        return ((a+b) < (other.a+other.b));
    }
    bool operator > (const UDD& other) const {
        return ((a+b) > (other.a+other.b));
    }
    bool operator == (const UDD& other) const {
        return ((a+b) == (other.a+other.b));
    }


    bool operator() (const int &x) {
        return (x == a || x == b);
    }

    UDD operator + (const UDD &rhs) const {
                UDD tmp = *this;
                tmp.a = tmp.a + rhs.a;
                tmp.b = tmp.b + rhs.b;
                return tmp;
    }

    UDD()
        : a(0),b(0) { }
    UDD(int _in)
        : a(_in), b(_in +1)  { }

};
);

BOLT_TEMPLATE_REGISTER_NEW_TYPE( bolt::cl::detail::CountIfEqual, int, UDD );
BOLT_TEMPLATE_REGISTER_NEW_ITERATOR( bolt::cl::device_vector, int, UDD );

TEST(countFloatValueOccuranceStdVect, CountInt){
    const int aSize = 1<<24;
    std::vector<int> stdInput(aSize);
    std::vector<int> tbbInput(aSize);

    int myintValue = 2;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = rand() % 10 + 1;
    tbbInput[i] = stdInput[i];
    }

    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myintValue);
    size_t boltCount = bolt::cl::count(tbbInput.begin(), tbbInput.end(), myintValue);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;

}

TEST(countFloatValueOccuranceStdVect, SerialCountInt){
    const int aSize = 1<<24;
    std::vector<int> stdInput(aSize);
    std::vector<int> tbbInput(aSize);

    int myintValue = 2;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = rand() % 10 + 1;
    tbbInput[i] = stdInput[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myintValue);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myintValue);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;

}


TEST(countFloatValueOccuranceStdVect, MulticoreCountInt){

    const int aSize = 1<<24;
    std::vector<int> stdInput(aSize);
    std::vector<int> tbbInput(aSize);

    //bolt::cl::device_vector<int> stdInput(aSize);
    //bolt::cl::device_vector<int> tbbInput(aSize);


    int myintValue = 2;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = rand() % 10 + 1;
    tbbInput[i] = stdInput[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myintValue);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myintValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
 // std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect, CountFloat){
    const int aSize = 1<<24;
    std::vector<float> stdInput(aSize);
    std::vector<float> tbbInput(aSize);

    //bolt::cl::device_vector<int> stdInput(aSize);
    //bolt::cl::device_vector<int> tbbInput(aSize);


    float myfloatValue = 9.5f;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = 9.5f;//rand() % 10 + 1;
    tbbInput[i] = stdInput[i];
    }

    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myfloatValue);
    size_t boltCount = bolt::cl::count(tbbInput.begin(), tbbInput.end(), myfloatValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect, SerialCountFloat){
    const int aSize = 1<<24;
    std::vector<float> stdInput(aSize);
    std::vector<float> tbbInput(aSize);

    //bolt::cl::device_vector<int> stdInput(aSize);
    //bolt::cl::device_vector<int> tbbInput(aSize);


    float myfloatValue = 9.5f;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = 9.5f;//rand() % 10 + 1;
    tbbInput[i] = stdInput[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myfloatValue);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myfloatValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}


TEST(countFloatValueOccuranceStdVect, MulticoreCountFloatTBB){
    const int aSize = 1<<24;
    std::vector<float> stdInput(aSize);
    std::vector<float> tbbInput(aSize);

    //bolt::cl::device_vector<int> stdInput(aSize);
    //bolt::cl::device_vector<int> tbbInput(aSize);


    float myfloatValue = 9.5f;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = 9.5f;//rand() % 10 + 1;
    tbbInput[i] = stdInput[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myfloatValue);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myfloatValue);

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}


TEST(countFloatValueOccuranceStdVect, CountUDD){
    const int aSize = 1<<21;
    std::vector<UDD> stdInput(aSize);
    std::vector<UDD> tbbInput(aSize);


    UDD myUDD;
    myUDD.a = 3;
    myUDD.b = 5;


    for (int i=0; i < aSize; i++) {
       stdInput[i].a = rand()%10;
       stdInput[i].b = rand()%10;
       tbbInput[i].a = stdInput[i].a;
       tbbInput[i].b = stdInput[i].b;
    }

    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myUDD);
    size_t boltCount = bolt::cl::count(tbbInput.begin(), tbbInput.end(), myUDD);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;
    //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect, SerialCountUDD){
    const int aSize = 1<<21;
    std::vector<UDD> stdInput(aSize);
    std::vector<UDD> tbbInput(aSize);


    UDD myUDD;
    myUDD.a = 3;
    myUDD.b = 5;


    for (int i=0; i < aSize; i++) {
       stdInput[i].a = rand()%10;
       stdInput[i].b = rand()%10;
       tbbInput[i].a = stdInput[i].a;
       tbbInput[i].b = stdInput[i].b;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myUDD);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myUDD);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;
    //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}


TEST(countFloatValueOccuranceStdVect, MulticoreCountUDDTBB){
    const int aSize = 1<<21;
    std::vector<UDD> stdInput(aSize);
    std::vector<UDD> tbbInput(aSize);

    //bolt::cl::device_vector<int> stdInput(aSize);
    //bolt::cl::device_vector<int> tbbInput(aSize);


    UDD myUDD;
    myUDD.a = 3;
    myUDD.b = 5;


    for (int i=0; i < aSize; i++) {
       stdInput[i].a = rand()%10;
       stdInput[i].b = rand()%10;
       tbbInput[i].a = stdInput[i].a;
       tbbInput[i].b = stdInput[i].b;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myUDD);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myUDD);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
    //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect,DeviceCountUDDTBB){
    const int aSize = 1<<21;
    std::vector<UDD> stdInput(aSize);

    UDD myUDD;
    myUDD.a = 3;
    myUDD.b = 5;

    for (int i=0; i < aSize; i++) {
       stdInput[i].a = rand()%10;
       stdInput[i].b = rand()%10;
    }

    bolt::cl::device_vector<UDD> tbbInput(stdInput.begin(),stdInput.end());

    bolt::cl::control ctl = bolt::cl::control::getDefault();

    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myUDD);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myUDD);


    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;

    //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect,Serial_DeviceCountUDDTBB){
    const int aSize = 1<<21;
    std::vector<UDD> stdInput(aSize);

    UDD myUDD;
    myUDD.a = 3;
    myUDD.b = 5;

    for (int i=0; i < aSize; i++) {
       stdInput[i].a = rand()%10;
       stdInput[i].b = rand()%10;
    }

    bolt::cl::device_vector<UDD> tbbInput(stdInput.begin(),stdInput.end());

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myUDD);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myUDD);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;
    //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect,MultiCore_DeviceCountUDDTBB){
    const int aSize = 1<<21;
    std::vector<UDD> stdInput(aSize);

    UDD myUDD;
    myUDD.a = 3;
    myUDD.b = 5;

    for (int i=0; i < aSize; i++) {
       stdInput[i].a = rand()%10;
       stdInput[i].b = rand()%10;
    }

    bolt::cl::device_vector<UDD> tbbInput(stdInput.begin(),stdInput.end());

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myUDD);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myUDD);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;
    //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect, STDCountUDD){
    const int aSize = 1<<21;
    std::vector<UDD> stdInput(aSize);
    std::vector<UDD> tbbInput(aSize);

    UDD myUDD;
    myUDD.a = 3;
    myUDD.b = 5;


    for (int i=0; i < aSize; i++) {
       stdInput[i].a = rand()%10;
       stdInput[i].b = rand()%10;
       tbbInput[i].a = stdInput[i].a;
       tbbInput[i].b = stdInput[i].b;
    }

    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myUDD);
    size_t boltCount = bolt::cl::count(tbbInput.begin(), tbbInput.end(), myUDD);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;
    //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}


TEST(countFloatValueOccuranceStdVect, Serial_STDCountUDD){
    const int aSize = 1<<21;
    std::vector<UDD> stdInput(aSize);
    std::vector<UDD> tbbInput(aSize);

    UDD myUDD;
    myUDD.a = 3;
    myUDD.b = 5;


    for (int i=0; i < aSize; i++) {
       stdInput[i].a = rand()%10;
       stdInput[i].b = rand()%10;
       tbbInput[i].a = stdInput[i].a;
       tbbInput[i].b = stdInput[i].b;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myUDD);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myUDD);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;
    //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect, MultiCore_STDCountUDDTBB){
    const int aSize = 1<<21;
    std::vector<UDD> stdInput(aSize);
    std::vector<UDD> tbbInput(aSize);


    UDD myUDD;
    myUDD.a = 3;
    myUDD.b = 5;


    for (int i=0; i < aSize; i++) {
       stdInput[i].a = rand()%10;
       stdInput[i].b = rand()%10;
       tbbInput[i].a = stdInput[i].a;
       tbbInput[i].b = stdInput[i].b;
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    size_t stdCount = std::count(stdInput.begin(), stdInput.end(), myUDD);
    size_t boltCount = bolt::cl::count(ctl, tbbInput.begin(), tbbInput.end(), myUDD);

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
        <<boltCount<<std::endl;
    //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect, CountifInt){
    const int aSize = 1<<24;
    std::vector<int> stdInput(aSize);

    int myintValue = 2;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = rand() % 10 + 1;
    }
    bolt::cl::device_vector<int> tbbInput(stdInput.begin(),stdInput.end());


    bolt::cl::control ctl = bolt::cl::control::getDefault();
    size_t stdCount = std::count_if(stdInput.begin(), stdInput.end(), InRange<int>(2,10000));
    size_t boltCount = bolt::cl::count_if(ctl, tbbInput.begin(), tbbInput.end(), InRange<int>(2,10000));

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;

}

TEST(countFloatValueOccuranceStdVect, Serial_CountifInt){
    const int aSize = 1<<24;
    std::vector<int> stdInput(aSize);

    int myintValue = 2;

    for (int i=0; i < aSize; i++) {
        stdInput[i] = rand() % 10 + 1;
    }
    bolt::cl::device_vector<int> tbbInput(stdInput.begin(),stdInput.end());


    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    size_t stdCount = std::count_if(stdInput.begin(), stdInput.end(), InRange<int>(2,10000));
    size_t boltCount = bolt::cl::count_if(ctl, tbbInput.begin(), tbbInput.end(), InRange<int>(2,10000));

    EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;

}

TEST(countFloatValueOccuranceStdVect, MulticoreCountifIntTBB){
    const int aSize = 1<<24;
    std::vector<int> stdInput(aSize);

    int myintValue = 2;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = rand() % 10 + 1;

    //tbbInput[i] = stdInput[i];
    }
    bolt::cl::device_vector<int> tbbInput(stdInput.begin(),stdInput.end());


    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    size_t stdCount = std::count_if(stdInput.begin(), stdInput.end(), InRange<int>(2,10000));
    size_t boltCount = bolt::cl::count_if(ctl, tbbInput.begin(), tbbInput.end(), InRange<int>(2,10000));

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect, CountifFloat){
    const int aSize = 1<<24;
    std::vector<float> stdInput(aSize);
    std::vector<float> tbbInput(aSize);


    float myfloatValue = 9.5f;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = (rand() % 10 + 1) * 45.f;
    tbbInput[i] = stdInput[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    size_t stdCount = std::count_if(stdInput.begin(), stdInput.end(), InRange<float>(5.2f,57.2f));
    size_t boltCount = bolt::cl::count_if(ctl, tbbInput.begin(), tbbInput.end(), InRange<float>(5.2f,57.2f));

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect, SerialCountifFloat){
    const int aSize = 1<<24;
    std::vector<float> stdInput(aSize);
    std::vector<float> tbbInput(aSize);


    float myfloatValue = 9.5f;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = (rand() % 10 + 1) * 45.f;
    tbbInput[i] = stdInput[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::SerialCpu);
    size_t stdCount = std::count_if(stdInput.begin(), stdInput.end(), InRange<float>(5.2f,57.2f));
    size_t boltCount = bolt::cl::count_if(ctl, tbbInput.begin(), tbbInput.end(), InRange<float>(5.2f,57.2f));

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

TEST(countFloatValueOccuranceStdVect, MulticoreCountifFloatTBB){
    const int aSize = 1<<24;
    std::vector<float> stdInput(aSize);
    std::vector<float> tbbInput(aSize);

    //bolt::cl::device_vector<int> stdInput(aSize);
    //bolt::cl::device_vector<int> tbbInput(aSize);


    float myfloatValue = 9.5f;

    for (int i=0; i < aSize; i++) {
    stdInput[i] = (rand() % 10 + 1) * 45.f;
    tbbInput[i] = stdInput[i];
    }

    bolt::cl::control ctl = bolt::cl::control::getDefault();
    ctl.setForceRunMode(bolt::cl::control::MultiCoreCpu);
    size_t stdCount = std::count_if(stdInput.begin(), stdInput.end(), InRange<float>(5.2f,57.2f));
    size_t boltCount = bolt::cl::count_if(ctl, tbbInput.begin(), tbbInput.end(), InRange<float>(5.2f,57.2f));

  EXPECT_EQ(stdCount, boltCount)<<"Failed as: \nSTD Count = "<<stdCount<<std::endl<<"Bolt Count = "
      <<boltCount<<std::endl;
  //std::cout<<"STD Count = "<<stdCount<<std::endl<<"Bolt Count = "<<boltCount<<std::endl;
}

#if(TEST_DOUBLE == 1)
INSTANTIATE_TEST_CASE_P (useStdVectWithDoubleValues, countDoubleValueUsedASKeyInStdVect,
                         ::testing::Values(1, 100, 1000, 10000, 100000));
INSTANTIATE_TEST_CASE_P (serialFloatValueWithStdVector, testCountIfFloatWithStdVector,
                         ::testing::Values(10, 100, 1000, 10000, 100000));
#endif

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest( &argc, &argv[ 0 ] );

    //  Register our minidump generating logic
//    bolt::miniDumpSingleton::enableMiniDumps( );

    int retVal = RUN_ALL_TESTS( );

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


#else



#include "stdafx.h"
#include <vector>
#include <algorithm>

#include <bolt/cl/count.h>

//Count with a vector input
void testCount1(int aSize)
{
  bolt::cl::device_vector<int> A(aSize);
  for (int i=0; i < aSize; i++) {
    A[i] = i+1;
  };

  bolt::cl::count (A.begin(), A.end(), 37);
};


// Count with an array input:
void testCount2()
{
  const int aSize = 13;
  int A[aSize] = {0, 10, 42, 55, 13, 13, 42, 19, 42, 11, 42, 99, 13};

  size_t count42 = bolt::cl::count (A, A+aSize, 42);
  size_t count13 = bolt::cl::count (A, A+aSize, 13);

  bolt::cl::control::getDefault().debug(bolt::cl::control::debug::Compile);

  std::cout << "Count42=" << count42 << std::endl;
  std::cout << "Count13=" << count13 << std::endl;
  std::cout << "Count7=" << bolt::cl::count (A, A+aSize, 7) << std::endl;
  std::cout << "Count10=" << bolt::cl::count (A, A+aSize, 10) << std::endl;
};



// This breaks the BOLT_CODE_STRING macro - need to move to a #include file or replicate the code.
std::string InRange_CodeString =
BOLT_CODE_STRING(
template<typename T>
// Functor for range checking.
struct InRange {
  InRange (T low, T high) {
    _low=low;
    _high=high;
  };

  bool operator() (const T& value) {
    //printf("Val=%4.1f, Range:%4.1f ... %4.1f\n", value, _low, _high);
    return (value >= _low) && (value <= _high) ;
  };

  T _low;
  T _high;
};
);

BOLT_CREATE_TYPENAME(InRange<int>);
BOLT_CREATE_CLCODE(InRange<int>, InRange_CodeString);
BOLT_CREATE_TYPENAME(InRange<float>);
BOLT_CREATE_CLCODE(InRange<float>, InRange_CodeString);







void testCountIf(int aSize)
{
  bolt::cl::device_vector<float> A(aSize);
  bolt::cl::device_vector<float> B(aSize);
  for (int i=0; i < aSize; i++) {
    A[i] = static_cast<float> (i+1);
    B[i] = A[i];
  };

  std::cout << "STD Count7..15=" << std::count_if (A.begin(), A.end(), InRange<float>(7,15)) << std::endl;
  std::cout << "BOLT Count7..15=" << bolt::cl::count_if (B.begin(), B.end(), InRange<float>(7,15)) << std::endl;
}

void test_bug(int aSize)
{
  //int aSize = 1024;
  bolt::cl::device_vector<int> A(aSize);
  bolt::cl::device_vector<int> B(aSize);
  for (int i=0; i < aSize; i++) {
    A[i] = rand() % 10 + 1;
    B[i] = A[i];
  }

  int stdInRangeCount = std::count_if (A.begin(), A.end(), InRange<int>(1,10)) ;
  int boltInRangeCount = bolt::cl::count_if (B.begin(), B.end(), InRange<int>(1, 10)) ;
  std:: cout << stdInRangeCount << "   "   << boltInRangeCount << "\n";
  //std::cout << "STD Count7..15=" << std::count_if (A.begin(), A.end(), InRange<float>(7,15)) << std::endl;
  //std::cout << "BOLT Count7..15=" << bolt::cl::count_if (A.begin(), A.end(),
  // InRange<float>(7,15), InRange_CodeString) << std::endl;
}

int _tmain(int argc, _TCHAR* argv[])
{
  testCount1(100);

  testCount2();

  testCountIf(1024);
  test_bug(1024);
  return 0;
}

#endif
