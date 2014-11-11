/***************************************************************************
*   Copyright 2012 - 2013 Advanced Micro Devices, Inc.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/

#if defined(BOLT_DEBUG_LOG)
#if !defined( BOLT_LOG )
#define BOLT_LOG
#include <vector>
#include <string>
#pragma once

namespace BOLTLOG{

    typedef enum CodePaths
    {
        BOLT_MULTICORE_CPU,
        BOLT_OPENCL_GPU,
        BOLT_OPENCL_CPU,
        BOLT_SERIAL_CPU,
    };

    typedef enum FUNCTION_EXE
    {
        BOLT_BINARYSEARCH,
        BOLT_COPY,
        BOLT_COUNT,
        BOLT_FILL,
		BOLT_GATHER,
        BOLT_GENERATE,
        BOLT_INNERPRODUCT,
		BOLT_MERGE,
        BOLT_MAXELEMENT,
        BOLT_MINELEMENT,
        BOLT_REDUCE,
        BOLT_REDUCEBYKEY,
        BOLT_SCAN,
        BOLT_SCANBYKEY,
		BOLT_SCATTER,
        BOLT_SORT,
        BOLT_SORTBYKEY,
        BOLT_STABLESORT,
        BOLT_STABLESORTBYKEY,
        BOLT_TRANSFORMREDUCE,
        BOLT_TRANSFORMSCAN,
        BOLT_TRANSFORM
    };

    class FunPaths
    {
        public:
        FunPaths(FUNCTION_EXE f,CodePaths p,std::string m=""):fun(f),path(p),msg(m){}
        CodePaths path;
        FUNCTION_EXE fun;
        std::string msg;
    };



    class CaptureLog
    {
        private:
        CaptureLog(){}
        CaptureLog(CaptureLog &p);
        CaptureLog & operator = (const CaptureLog&);
        static CaptureLog *instance;
        static bool instanceFlag;

        protected:
        static std::vector<FunPaths> takePaths;

        public:
        static  CaptureLog* getInstance();
        static void Initialize();
        static void CodePathTaken(FUNCTION_EXE fun,CodePaths path,std::string m="");
        static void WhatPathTaken(std::vector<FunPaths> &paths);

        ~CaptureLog()
        {
            instanceFlag = false;
        }

    };

        bool CaptureLog::instanceFlag = false;
        CaptureLog *CaptureLog::instance = NULL;
        std::vector<FunPaths> CaptureLog::takePaths;


    CaptureLog* CaptureLog::getInstance()
    {
        if(! instanceFlag)
        {
            instance = new CaptureLog();
            instanceFlag = true;
            return instance;
        }
        else
        {
            return instance;
        }
    }


    void  CaptureLog::CodePathTaken(FUNCTION_EXE fun,CodePaths path,std::string m)
    {
        takePaths.push_back(FunPaths(fun,path,m));
    }

    void  CaptureLog::Initialize()
    {
        takePaths.clear();
    }

    void CaptureLog::WhatPathTaken(std::vector<FunPaths> &paths)
    {
        for(std::vector<FunPaths>::iterator parse=takePaths.begin(); parse!=takePaths.end(); parse++)
        {
            paths.push_back(*parse);
        }
    }


} // End of namespace BOLTLOG
#endif
#endif
