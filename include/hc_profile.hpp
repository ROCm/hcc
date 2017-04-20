#pragma once

#include <string>
#include <cxxabi.h>

class cxlProfiler {
private:
    int enabled;
public:
    cxlProfiler();
    ~cxlProfiler();
};

class cxlMarker {
public:
    static int enabled;
    cxlMarker(const char* szMarkerName, const char* szGroupName, const char* szUserString);
    ~cxlMarker();
};

int status;

#define __HC_XSTR(S) __HC_STR(S)
#define __HC_STR(S)  #S
#define CXL_PROFILER       cxlProfiler();
#define CXL_MARKER         cxlMarker( (std::string(__FUNCTION__) + " " \
                                + std::string(__HC_XSTR([__FILE__:__LINE__]))).c_str() \
                                ,nullptr,nullptr );
#define CXL_MARKER_CLASS   cxlMarker( (std::string(abi::__cxa_demangle(typeid(*this).name(), 0, 0, &status)) + " " \
                                + std::string(__HC_XSTR([__FILE__:__LINE__]))).c_str() \
                                ,nullptr,nullptr );
