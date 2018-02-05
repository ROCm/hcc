// Copyright (c) Microsoft
// All rights reserved
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache Version 2.0 License for specific language governing permissions and limitations under the License.
/// <tags>P0</tags>
/// <summary>Test the use of Union definitions in GPU function and array declaration </summary>


#include <amptest.h>

using namespace Concurrency;
using namespace Concurrency::Test;

enum TypeOfData
{
	Int,
	Long,
	Struct,
	EnumType
};

typedef struct
{
	int lbyte;
}LSB;

typedef struct
{
	int mbyte;
}MSB;

typedef struct
{
	LSB lsb;
	MSB msb;
}Word;

typedef union
{
	int A;
	long B;
	Word word;
	TypeOfData type;
}SampleUnion;

SampleUnion testUnion(TypeOfData type) __GPU
{
	SampleUnion a;
	if( Int == type  )
	{
		a.A = 10;
	}
	else if ( Long == type )
	{
		a.B = 12345L;
	}
	else if ( Struct == type )
	{
		a.word.lsb.lbyte = 0x0;
		a.word.msb.mbyte = 0xff;
	}
	else if ( EnumType == type )
	{
		a.type = EnumType;
	}
	return a;
}

int test(accelerator_view &rv)
{
	
	bool pass = true;
	TypeOfData type = Int;
	SampleUnion obj = testUnion(type);
	pass &=  ( obj.A == 10 );
	
	type = Long;
	obj = testUnion(type);
	pass &= ( obj.B == 12345L );
	
	type = Struct;
	obj = testUnion(type);
	pass &= ( obj.word.lsb.lbyte == 0x0 && obj.word.msb.mbyte == 0xff );
	
	type = EnumType;
	obj = testUnion(type);
	pass &= ( obj.type == EnumType );
	
	// Verifying Array Declaration of Union does not result in Compilation Error
	extent<1> vector(128);
    array<SampleUnion, 1> aS(vector, rv);
	
	printf("Test : %s \n",pass?"passed" : "failed");
	return ( pass ? 0 : 1 );
}

int main()
{
	accelerator device;
    if (!get_device(Device::ALL_DEVICES, device))
    {
        printf("Unable to get requested compute device\n");
        return 2;
    }
    accelerator_view rv = device.get_default_view();

    return test(rv);
}

