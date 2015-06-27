// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>test the use of Enum with supported datatypes</summary>

#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

enum Suit : long {
    Diamonds = 100L,
    Hearts,
    Clubs,
    Spades,
	Invalid
};

bool foo(Suit suit) __GPU
{
    if (suit == Diamonds)
        return true;
    else
        return false;
}

Suit typeCast( long value ) __GPU
{
	if( value < (long)Diamonds  || value > (long) Spades )
		return Invalid;
	return (Suit)value;
}

Suit& testEnumRef( long value ) __GPU
{
	Suit *ptr = (Suit *) &value;
	Suit &result =  *ptr;
	*ptr = (Suit) ( *ptr + 1 );
	return result;
}

int testEnum()
{
	bool passed = true ;
	passed &= ( foo(Hearts) == false );
	passed &= ( foo(Diamonds) == true );
	
	passed &= (typeCast(99L) == Invalid);
	passed &= (typeCast(100L) == Diamonds);
	passed &= (typeCast(101L) == Hearts);
	passed &= (typeCast(102L) == Clubs);
	passed &= (typeCast(103L) == Spades);
	passed &= (typeCast(104L) == Invalid);
	passed &= (typeCast(105L) == Invalid);
	passed &= (testEnumRef(99L) == Diamonds);
	printf("Test : %s \n",passed?"passed" : "failed");
	return ( passed ? 0 : 1 );	
}

int main(int argc, char **argv)
{
	accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }
    return testEnum();
}

