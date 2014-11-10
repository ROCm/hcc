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

#include <bolt/unicode.h>
#include <bolt/cl/sort.h>

#include <vector>
#include <numeric>
BOLT_FUNCTOR(MyType<int>,
template <typename T>
struct MyType {
    T a;

    bool operator() (const MyType& lhs, const MyType& rhs) const {
        return (lhs.a > rhs.a);
    }
    bool operator < (const MyType& other) const {
        return (a < other.a);
    }
    bool operator > (const MyType& other) const {
        return (a > other.a);
    }
    bool operator >= (const MyType& other) const {
        return (a >= other.a);
    }
    MyType()
        : a(0) { }
};
);

//  Create a new bolt template specialization of the bolt::cl::greater template,
//  using the same definition already registered with the built in 'int' type,
//  to handle the new user defined MyType<int>
BOLT_TEMPLATE_REGISTER_NEW_TYPE(bolt::cl::greater, int, MyType<int>);

//  Create a new bolt template specialization of the bolt::cl::device_vector template,
//  using the same definition already registered with the built in 'int' type,
//  to contain the new user defined MyType<int>
BOLT_TEMPLATE_REGISTER_NEW_ITERATOR(bolt::cl::device_vector, int, MyType<int>);

template <typename T>
void CheckDescendingPtr(T *input, size_t length)
{
    size_t index;
    for( index = 0; index < length -1; ++index )
    {
        if(input[index] >= input[index+1])
            continue;
        else
            break;
    }
    if(index == (length-1))
    {
        std::cout << "PASSED....\n";
    }
    else
    {
        std::cout << "FAILED....\n";
    }
}
template <typename T>
void CheckDescending(T &input, size_t length)
{
    size_t index;
    for( index = 0; index < input.size( ) -1; ++index )
    {
        if(input[index] >= input[index+1])
            continue;
        else
            break;
    }
    if(index == (length-1))
    {
        std::cout << "PASSED....\n";
    }
    else
    {
        std::cout << "FAILED....\n";
    }
}
template <typename T>
void CheckAscending(T &input, size_t length)
{
    size_t index;
    for( index = 0; index < input.size( ) -1; ++index )
    {
        if(input[index] <= input[index+1])
            continue;
        else
            break;
    }
    if(index == (length-1))
    {
        std::cout << "PASSED....\n";
    }
    else
    {
        std::cout << "FAILED....\n";
    }
}

int main()
{
	//Usage with basic vector implementation.
	int length = 1024;
	std::vector<int> input(length);
    int a[8] = {2, 9, 3, 7, 5, 6, 3, 8};
	std::generate(input.begin(), input.end(), rand);
    std::cout << "\nSort EXAMPLE \n";
    //Usage with std::vector types.
    std::cout << "\nSorting std::vector of size " << length << " elements. ...\n";
	bolt::cl::sort( input.begin(), input.end(), bolt::cl::greater<int>());
    CheckDescending (input, length);

    //Usage with Array types.
    std::cout << "\nSorting Array of integers of size 8 elements ...\n";
	bolt::cl::sort( a, a+8, bolt::cl::greater<int>());
    CheckDescendingPtr (a, 8);

    //The below sample demonstration will not work for non AMD GPUs.
    //If you have an AMD GPU you can enable the code and build it.

    //Sort using Device Vector
	//std::vector<int> boltInput(length);
	//std::generate(boltInput.begin(), boltInput.end(), rand);
    /*Sort using the bolt::cl::control and device_vector*/
	//MyOclContext ocl = initOcl(CL_DEVICE_TYPE_GPU, 0);
	//bolt::cl::control c(ocl._queue);  // construct control structure from the queue.
    //device_vector created here
    //bolt::cl::device_vector<int> dvInput( boltInput.begin(), boltInput.end(), CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, c);

    //std::cout << "\nSorting device_vector of integers of size " << length << " elements. ...\n";
	//bolt::cl::sort(c, dvInput.begin(), dvInput.end());
    //CheckAscending (dvInput, length);

    //Usage with user defined data types.
	//typedef MyType<int> mytype;
	//std::vector<mytype> myTypeBoltInput1(length);
	//std::vector<mytype> myTypeBoltInput2(length);

    //for (int i=0;i<length;i++)
    //{
    //    myTypeBoltInput1[i].a= (int)(i +2);
    //    myTypeBoltInput2[i].a= (int)(i +2);
    //}
    //std::cout << "\nSorting user-defined data type of size " << length << " elements with bolt Functor. ...\n";
    //bolt::cl::sort(c, myTypeBoltInput1.begin(), myTypeBoltInput1.end(),bolt::cl::greater<mytype>());
    //CheckDescending (myTypeBoltInput1, length);

    //OR
    //std::cout << "\nSorting user-defined data type of size " << length << " elements with User-defined functor. ...\n";
    //bolt::cl::sort(c, myTypeBoltInput2.begin(), myTypeBoltInput2.end(),mytype());
    //We use descending here because the FUNCTOR does a greater than comparison.
    //CheckDescending (myTypeBoltInput1, length);
    std::cout << "COMPLETED. ...\n";
	return 0;
}
