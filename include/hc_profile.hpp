#pragma once

#include <string>

#ifdef USE_CODEXL_ACTIVITY_LOGGER
#include "CXLActivityLogger.h"
#endif

class cxlProfiler {
public:
        cxlProfiler() {
#ifdef USE_CODEXL_ACTIVITY_LOGGER
                amdtInitializeActivityLogger();
#endif
	}

        ~cxlProfiler() {
#ifdef USE_CODEXL_ACTIVITY_LOGGER
               	amdtFinalizeActivityLogger();
#endif
        }
};

class cxlMarker {
public:
	cxlMarker(const char* szMarkerName, const char* szGroupName, const char* szUserString) {
#ifdef USE_CODEXL_ACTIVITY_LOGGER
		int err = amdtBeginMarker(szMarkerName, szGroupName, szUserString);
#endif
	}

	~cxlMarker() {
#ifdef USE_CODEXL_ACTIVITY_LOGGER
                int err = amdtEndMarker();
#endif
	}
};

#define __HC_XSTR(S) __HC_STR(S)
#define __HC_STR(S)  #S
#define CXL_PROFILER       cxlProfiler();
#define CXL_MARKER         cxlMarker( (std::string(__FUNCTION__) + std::string(" ") \
				+ std::string(__HC_XSTR([__FILE__:__LINE__]))).c_str() \
				,nullptr,nullptr );
#define CXL_MARKER_CLASS   cxlMarker( (std::string(typeid(*this).name()) + std::string(" ") \
                                + std::string(__HC_XSTR([__FILE__:__LINE__]))).c_str() \
                                ,nullptr,nullptr );
