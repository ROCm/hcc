// RUN: %clamp-preprocess %s %t && cat %t | %FileCheck %s
#include <iostream>
#include <vector>
#include <algorithm>


template<typename Container>
static void printA2(const char * msg, const Container &a, const Container &b, int x_size)
{
  std::wcout << msg << std::endl;
  for (int i = 0; i < x_size; i++)
    std::wcout << a[i] << "\t" << b[i] << std::endl;
};


// From Bolt AMP test case
void TransformSimpleTest ()
{
    const int aSize = 16;
    int a[aSize] = {4,0,5,5,0,5,5,1,3,1,0,3,1,1,3,5};
    int out[aSize] = {0x0};
    std::transform(a,a+aSize, out, std::negate<int>());
// CHECK: bolt::amp::transform(a,a+aSize, out, std::negate<int>());
    
    printA2("Transform Neg - From Pointer", a, out, aSize);
}



int main ()
{  
    TransformSimpleTest();

    // generate random data (on host)
    size_t length = 4;
    std::vector<int> vec(length);
    vec[0] = 100;
    vec[1] = -122;
    vec[2] = 900;
    vec[3] = 1000;
    #define A vec.begin(),vec.end()
    std::vector<int>::iterator stlMaxE = (std::vector<int>::iterator) std::max_element(A);
// CHECK: std::vector<int>::iterator stlMaxE = (std::vector<int>::iterator) bolt::amp::max_element(A);

    std::cout << "Get max_element = " << *stlMaxE << "\n";
    std::cout << vec[0]<<"  <=  "<<vec[1]<<"  <=  "<<vec[2]<<"  <=  "<<vec[3]<<"\n";

    return 0;
}





