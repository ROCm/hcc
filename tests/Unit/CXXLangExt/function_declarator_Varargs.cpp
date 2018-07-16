
// RUN: %hc %s -o %t.out && %t.out

#include <hc/hc.hpp>

void NoEllipsisAllowed(int x, ...) [[hc]] {}

int main()
{
	//Execution should never reach here
	// return 1 to indicate failure.
        //success under -fhsa-ext
        return 0;
}

