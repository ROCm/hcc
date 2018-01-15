
// RUN: %hc %s -o %t.out && %t.out

#include <amp.h>

void NoEllipsisAllowed(int x, ...) restrict(amp) {}

int main()
{
	//Execution should never reach here
	// return 1 to indicate failure.
        //success under -fhsa-ext
        return 0;
}

