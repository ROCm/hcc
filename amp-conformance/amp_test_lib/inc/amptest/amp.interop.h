#pragma once
/**********************************************************************************
* amptest\amp.interop.h
*
*
**********************************************************************************/

// Attach the dpctest.lib
#include <amp.h>
#include <amptest.h>

namespace Concurrency
{
    namespace Test
    {
		// For properly wrapping the macros
		#define AMP_MAC_S                   do {
		#define AMP_MAC_E                   } while (0)

		#define AMP_RELEASE(_p)             AMP_MAC_S if ((_p) != NULL) { ULONG ref = (_p)->Release(); if (0 == ref) (_p) = nullptr; } AMP_MAC_E
		#define AMP_RELEASE_VERIFY(_p)      AMP_MAC_S if ((_p) != NULL) { if (0 != (_p)->Release()) return runall_fail; (_p) = nullptr; } AMP_MAC_E
		
		bool objects_same(IUnknown *pObject1, IUnknown *pObject2)
		{
			IUnknown *pBase1 = NULL, *pBase2 = NULL;
			bool bRes = true;
			if (SUCCEEDED(pObject1->QueryInterface( __uuidof(IUnknown), (void**)&pBase1)) && (pBase1 != NULL))
			{
				if (SUCCEEDED(pObject2->QueryInterface( __uuidof(IUnknown), (void**)&pBase2)) && (pBase2 != NULL))
				{
					bRes = (pBase1 == pBase2);
				}

				// Clean up this function side effects
				AMP_RELEASE(pBase1);
				AMP_RELEASE(pBase2);
			}

			return bRes;
		}
	}
}
