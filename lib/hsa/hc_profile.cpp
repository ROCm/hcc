#include "hc_profile.hpp"
#include "CXLActivityLogger.h"

cxlProfiler::cxlProfiler() {
    if (getenv("HCC_CODEXL_PROFILING")) {
        enabled = 1;
        cxlMarker::enabled = 1;
        amdtInitializeActivityLogger();
    }
}
cxlProfiler::~cxlProfiler() {
    if (enabled) {
        amdtFinalizeActivityLogger();
    }
}

int cxlMarker::enabled = 0;

cxlMarker::cxlMarker(const char* szMarkerName, const char* szGroupName, const char* szUserString) {
    if (enabled) {
        int err = amdtBeginMarker(szMarkerName, szGroupName, szUserString);
    }
}
cxlMarker::~cxlMarker() {
    if (enabled) {
       int err = amdtEndMarker();
    }
}

