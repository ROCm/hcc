#include "hc_profile.hpp"

#ifdef CODEXL_PROFILING_ENABLED
#include "CXLActivityLogger.h"
#endif

cxlProfiler::cxlProfiler() {
#ifdef CODEXL_PROFILING_ENABLED
    if (getenv("HCC_CODEXL_PROFILING")) {
        enabled = true;
        cxlMarker::enabled = true;
        amdtInitializeActivityLogger();
    }
#endif
}
cxlProfiler::~cxlProfiler() {
#ifdef CODEXL_PROFILING_ENABLED
    if (enabled) {
        amdtFinalizeActivityLogger();
    }
#endif
}

bool cxlMarker::enabled = false;

cxlMarker::cxlMarker(const char* szMarkerName, const char* szGroupName, const char* szUserString) {
#ifdef CODEXL_PROFILING_ENABLED
    if (enabled) {
        int err = amdtBeginMarker(szMarkerName, szGroupName, szUserString);
    }
#endif
}
cxlMarker::~cxlMarker() {
#ifdef CODEXL_PROFILING_ENABLED
    if (enabled) {
       int err = amdtEndMarker();
    }
#endif
}

