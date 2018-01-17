
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <amp.h>

class MyBaseClass
{
	//Make class non-empty
    int i;
};

class MyDerivedClass : virtual public MyBaseClass {};

void VirtualBaseClassNotAllowed(int x) restrict(amp)
{
	MyDerivedClass obj;
}

int main()
{
	//Execution should never reach here
	//return 1 to indicate failure
        //success under -fhsa-ext
	return 0;
}

