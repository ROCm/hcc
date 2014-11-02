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


#include <iostream>
#include <fstream>
#include <streambuf>
#if defined( _WIN32 )
#include <direct.h>  //windows CWD for error message
#endif
#include <bolt/unicode.h>
#include <algorithm>
#include <vector>
#include <set>

#include "bolt/cl/bolt.h"
#include "bolt/unicode.h"

//  Include all kernel string objects

#include "bolt/binary_search_kernels.hpp"
#include "bolt/copy_kernels.hpp"
#include "bolt/count_kernels.hpp"
#include "bolt/fill_kernels.hpp"
#include "bolt/gather_kernels.hpp"
#include "bolt/generate_kernels.hpp"
#include "bolt/merge_kernels.hpp"
#include "bolt/min_element_kernels.hpp"
#include "bolt/reduce_kernels.hpp"
#include "bolt/reduce_by_key_kernels.hpp"
#include "bolt/scan_kernels.hpp"
#include "bolt/scan_by_key_kernels.hpp"
#include "bolt/scatter_kernels.hpp"
#include "bolt/sort_kernels.hpp"
#include "bolt/sort_uint_kernels.hpp"
#include "bolt/sort_int_kernels.hpp"
#include "bolt/sort_common_kernels.hpp"
#include "bolt/sort_by_key_kernels.hpp"
#include "bolt/sort_by_key_int_kernels.hpp"
#include "bolt/sort_by_key_uint_kernels.hpp"
#include "bolt/stablesort_kernels.hpp"
#include "bolt/stablesort_by_key_kernels.hpp"
#include "bolt/transform_kernels.hpp"
#include "bolt/transform_reduce_kernels.hpp"
#include "bolt/transform_scan_kernels.hpp"

namespace bolt {
    namespace cl {

    void getVersion( cl_uint& major, cl_uint& minor, cl_uint& patch )
    {
        major	= BoltVersionMajor;
        minor	= BoltVersionMinor;
        patch	= BoltVersionPatch;
    }

    // for printing errors
    const std::string hr = "###############################################################################";
    const std::string es = "ERROR: ";

    std::string clErrorStringA( const cl_int& status )
    {
        switch( status )
        {
            case CL_INVALID_DEVICE_PARTITION_COUNT:
                return "CL_INVALID_DEVICE_PARTITION_COUNT";
            case CL_INVALID_LINKER_OPTIONS:
                return "CL_INVALID_LINKER_OPTIONS";
            case CL_INVALID_COMPILER_OPTIONS:
                return "CL_INVALID_COMPILER_OPTIONS";
            case CL_INVALID_IMAGE_DESCRIPTOR:
                return "CL_INVALID_IMAGE_DESCRIPTOR";
            case CL_INVALID_PROPERTY:
                return "CL_INVALID_PROPERTY";
            case CL_INVALID_GLOBAL_WORK_SIZE:
                return "CL_INVALID_GLOBAL_WORK_SIZE";
            case CL_INVALID_MIP_LEVEL:
                return "CL_INVALID_MIP_LEVEL";
            case CL_INVALID_BUFFER_SIZE:
                return "CL_INVALID_BUFFER_SIZE";
            case CL_INVALID_GL_OBJECT:
                return "CL_INVALID_GL_OBJECT";
            case CL_INVALID_OPERATION:
                return "CL_INVALID_OPERATION";
            case CL_INVALID_EVENT:
                return "CL_INVALID_EVENT";
            case CL_INVALID_EVENT_WAIT_LIST:
                return "CL_INVALID_EVENT_WAIT_LIST";
            case CL_INVALID_GLOBAL_OFFSET:
                return "CL_INVALID_GLOBAL_OFFSET";
            case CL_INVALID_WORK_ITEM_SIZE:
                return "CL_INVALID_WORK_ITEM_SIZE";
            case CL_INVALID_WORK_GROUP_SIZE:
                return "CL_INVALID_WORK_GROUP_SIZE";
            case CL_INVALID_WORK_DIMENSION:
                return "CL_INVALID_WORK_DIMENSION";
            case CL_INVALID_KERNEL_ARGS:
                return "CL_INVALID_KERNEL_ARGS";
            case CL_INVALID_ARG_SIZE:
                return "CL_INVALID_ARG_SIZE";
            case CL_INVALID_ARG_VALUE:
                return "CL_INVALID_ARG_VALUE";
            case CL_INVALID_ARG_INDEX:
                return "CL_INVALID_ARG_INDEX";
            case CL_INVALID_KERNEL:
                return "CL_INVALID_KERNEL";
            case CL_INVALID_KERNEL_DEFINITION:
                return "CL_INVALID_KERNEL_DEFINITION";
            case CL_INVALID_KERNEL_NAME:
                return "CL_INVALID_KERNEL_NAME";
            case CL_INVALID_PROGRAM_EXECUTABLE:
                return "CL_INVALID_PROGRAM_EXECUTABLE";
            case CL_INVALID_PROGRAM:
                return "CL_INVALID_PROGRAM";
            case CL_INVALID_BUILD_OPTIONS:
                return "CL_INVALID_BUILD_OPTIONS";
            case CL_INVALID_BINARY:
                return "CL_INVALID_BINARY";
            case CL_INVALID_SAMPLER:
                return "CL_INVALID_SAMPLER";
            case CL_INVALID_IMAGE_SIZE:
                return "CL_INVALID_IMAGE_SIZE";
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
                return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case CL_INVALID_MEM_OBJECT:
                return "CL_INVALID_MEM_OBJECT";
            case CL_INVALID_HOST_PTR:
                return "CL_INVALID_HOST_PTR";
            case CL_INVALID_COMMAND_QUEUE:
                return "CL_INVALID_COMMAND_QUEUE";
            case CL_INVALID_QUEUE_PROPERTIES:
                return "CL_INVALID_QUEUE_PROPERTIES";
            case CL_INVALID_CONTEXT:
                return "CL_INVALID_CONTEXT";
            case CL_INVALID_DEVICE:
                return "CL_INVALID_DEVICE";
            case CL_INVALID_PLATFORM:
                return "CL_INVALID_PLATFORM";
            case CL_INVALID_DEVICE_TYPE:
                return "CL_INVALID_DEVICE_TYPE";
            case CL_INVALID_VALUE:
                return "CL_INVALID_VALUE";
            case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
                return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            case CL_DEVICE_PARTITION_FAILED:
                return "CL_DEVICE_PARTITION_FAILED";
            case CL_LINK_PROGRAM_FAILURE:
                return "CL_LINK_PROGRAM_FAILURE";
            case CL_LINKER_NOT_AVAILABLE:
                return "CL_LINKER_NOT_AVAILABLE";
            case CL_COMPILE_PROGRAM_FAILURE:
                return "CL_COMPILE_PROGRAM_FAILURE";
            case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
                return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:
                return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case CL_MAP_FAILURE:
                return "CL_MAP_FAILURE";
            case CL_BUILD_PROGRAM_FAILURE:
                return "CL_BUILD_PROGRAM_FAILURE";
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:
                return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case CL_IMAGE_FORMAT_MISMATCH:
                return "CL_IMAGE_FORMAT_MISMATCH";
            case CL_MEM_COPY_OVERLAP:
                return "CL_MEM_COPY_OVERLAP";
            case CL_PROFILING_INFO_NOT_AVAILABLE:
                return "CL_PROFILING_INFO_NOT_AVAILABLE";
            case CL_OUT_OF_HOST_MEMORY:
                return "CL_OUT_OF_HOST_MEMORY";
            case CL_OUT_OF_RESOURCES:
                return "CL_OUT_OF_RESOURCES";
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case CL_COMPILER_NOT_AVAILABLE:
                return "CL_COMPILER_NOT_AVAILABLE";
            case CL_DEVICE_NOT_AVAILABLE:
                return "CL_DEVICE_NOT_AVAILABLE";
            case CL_DEVICE_NOT_FOUND:
                return "CL_DEVICE_NOT_FOUND";
            case CL_SUCCESS:
                return "CL_SUCCESS";
            default:
                return "Error code not defined";
            break;
        }
    }

    std::wstring clErrorStringW( const cl_int& status )
    {
        switch( status )
        {
            case CL_INVALID_DEVICE_PARTITION_COUNT:
                return L"CL_INVALID_DEVICE_PARTITION_COUNT";
            case CL_INVALID_LINKER_OPTIONS:
                return L"CL_INVALID_LINKER_OPTIONS";
            case CL_INVALID_COMPILER_OPTIONS:
                return L"CL_INVALID_COMPILER_OPTIONS";
            case CL_INVALID_IMAGE_DESCRIPTOR:
                return L"CL_INVALID_IMAGE_DESCRIPTOR";
            case CL_INVALID_PROPERTY:
                return L"CL_INVALID_PROPERTY";
            case CL_INVALID_GLOBAL_WORK_SIZE:
                return L"CL_INVALID_GLOBAL_WORK_SIZE";
            case CL_INVALID_MIP_LEVEL:
                return L"CL_INVALID_MIP_LEVEL";
            case CL_INVALID_BUFFER_SIZE:
                return L"CL_INVALID_BUFFER_SIZE";
            case CL_INVALID_GL_OBJECT:
                return L"CL_INVALID_GL_OBJECT";
            case CL_INVALID_OPERATION:
                return L"CL_INVALID_OPERATION";
            case CL_INVALID_EVENT:
                return L"CL_INVALID_EVENT";
            case CL_INVALID_EVENT_WAIT_LIST:
                return L"CL_INVALID_EVENT_WAIT_LIST";
            case CL_INVALID_GLOBAL_OFFSET:
                return L"CL_INVALID_GLOBAL_OFFSET";
            case CL_INVALID_WORK_ITEM_SIZE:
                return L"CL_INVALID_WORK_ITEM_SIZE";
            case CL_INVALID_WORK_GROUP_SIZE:
                return L"CL_INVALID_WORK_GROUP_SIZE";
            case CL_INVALID_WORK_DIMENSION:
                return L"CL_INVALID_WORK_DIMENSION";
            case CL_INVALID_KERNEL_ARGS:
                return L"CL_INVALID_KERNEL_ARGS";
            case CL_INVALID_ARG_SIZE:
                return L"CL_INVALID_ARG_SIZE";
            case CL_INVALID_ARG_VALUE:
                return L"CL_INVALID_ARG_VALUE";
            case CL_INVALID_ARG_INDEX:
                return L"CL_INVALID_ARG_INDEX";
            case CL_INVALID_KERNEL:
                return L"CL_INVALID_KERNEL";
            case CL_INVALID_KERNEL_DEFINITION:
                return L"CL_INVALID_KERNEL_DEFINITION";
            case CL_INVALID_KERNEL_NAME:
                return L"CL_INVALID_KERNEL_NAME";
            case CL_INVALID_PROGRAM_EXECUTABLE:
                return L"CL_INVALID_PROGRAM_EXECUTABLE";
            case CL_INVALID_PROGRAM:
                return L"CL_INVALID_PROGRAM";
            case CL_INVALID_BUILD_OPTIONS:
                return L"CL_INVALID_BUILD_OPTIONS";
            case CL_INVALID_BINARY:
                return L"CL_INVALID_BINARY";
            case CL_INVALID_SAMPLER:
                return L"CL_INVALID_SAMPLER";
            case CL_INVALID_IMAGE_SIZE:
                return L"CL_INVALID_IMAGE_SIZE";
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
                return L"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
            case CL_INVALID_MEM_OBJECT:
                return L"CL_INVALID_MEM_OBJECT";
            case CL_INVALID_HOST_PTR:
                return L"CL_INVALID_HOST_PTR";
            case CL_INVALID_COMMAND_QUEUE:
                return L"CL_INVALID_COMMAND_QUEUE";
            case CL_INVALID_QUEUE_PROPERTIES:
                return L"CL_INVALID_QUEUE_PROPERTIES";
            case CL_INVALID_CONTEXT:
                return L"CL_INVALID_CONTEXT";
            case CL_INVALID_DEVICE:
                return L"CL_INVALID_DEVICE";
            case CL_INVALID_PLATFORM:
                return L"CL_INVALID_PLATFORM";
            case CL_INVALID_DEVICE_TYPE:
                return L"CL_INVALID_DEVICE_TYPE";
            case CL_INVALID_VALUE:
                return L"CL_INVALID_VALUE";
            case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
                return L"CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            case CL_DEVICE_PARTITION_FAILED:
                return L"CL_DEVICE_PARTITION_FAILED";
            case CL_LINK_PROGRAM_FAILURE:
                return L"CL_LINK_PROGRAM_FAILURE";
            case CL_LINKER_NOT_AVAILABLE:
                return L"CL_LINKER_NOT_AVAILABLE";
            case CL_COMPILE_PROGRAM_FAILURE:
                return L"CL_COMPILE_PROGRAM_FAILURE";
            case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
                return L"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:
                return L"CL_MISALIGNED_SUB_BUFFER_OFFSET";
            case CL_MAP_FAILURE:
                return L"CL_MAP_FAILURE";
            case CL_BUILD_PROGRAM_FAILURE:
                return L"CL_BUILD_PROGRAM_FAILURE";
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:
                return L"CL_IMAGE_FORMAT_NOT_SUPPORTED";
            case CL_IMAGE_FORMAT_MISMATCH:
                return L"CL_IMAGE_FORMAT_MISMATCH";
            case CL_MEM_COPY_OVERLAP:
                return L"CL_MEM_COPY_OVERLAP";
            case CL_PROFILING_INFO_NOT_AVAILABLE:
                return L"CL_PROFILING_INFO_NOT_AVAILABLE";
            case CL_OUT_OF_HOST_MEMORY:
                return L"CL_OUT_OF_HOST_MEMORY";
            case CL_OUT_OF_RESOURCES:
                return L"CL_OUT_OF_RESOURCES";
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                return L"CL_MEM_OBJECT_ALLOCATION_FAILURE";
            case CL_COMPILER_NOT_AVAILABLE:
                return L"CL_COMPILER_NOT_AVAILABLE";
            case CL_DEVICE_NOT_AVAILABLE:
                return L"CL_DEVICE_NOT_AVAILABLE";
            case CL_DEVICE_NOT_FOUND:
                return L"CL_DEVICE_NOT_FOUND";
            case CL_SUCCESS:
                return L"CL_SUCCESS";
            default:
                return L"Error code not defined";
            break;
        }
    }

    std::string fileToString(const std::string &fileName)
    {
        std::ifstream infile (fileName.c_str());
        if (infile.fail() ) {
#if defined( _WIN32 )
            TCHAR osPath[ MAX_PATH ];

            //	If loading the .cl file fails from the specified path, then make a last ditch attempt (purely for convenience) to find the .cl file right to the executable,
            //	regardless of what the CWD is
            //	::GetModuleFileName( ) returns TCHAR's (and we define _UNICODE for windows); but the fileName string is char's,
            //	so we needed to create an abstraction for string/wstring
            if( ::GetModuleFileName( NULL, osPath, MAX_PATH ) )
            {
                bolt::tstring thisPath( osPath );
                bolt::tstring::size_type pos = thisPath.find_last_of( _T( "\\" ) );

                bolt::tstring newPath;
                if( pos != bolt::tstring::npos )
                {
                    bolt::tstring exePath	= thisPath.substr( 0, pos + 1 );	// include the \ character

                    //	Narrow to wide conversion should always work, but beware of wide to narrow!
                    bolt::tstring convName( fileName.begin( ), fileName.end( ) );
                    newPath = exePath + convName;
                }

                infile.open( newPath.c_str( ) );
            }
#endif
            if (infile.fail() ) {
                //  Note:  Commented out because this widestr not initialized yet if called from global scope
                //TCHAR cCurrentPath[FILENAME_MAX];
                //if (_tgetcwd(cCurrentPath, sizeof(cCurrentPath) / sizeof(TCHAR))) {
                //    bolt::tout <<  _T( "CWD=" ) << cCurrentPath << std::endl;
                //};
                std::cout << "error: failed to open file: " << fileName << std::endl;
                throw;
            }
        }

        std::string str((std::istreambuf_iterator<char>(infile)),
            std::istreambuf_iterator<char>());
        return str;
    };


    /**********************************************************************
        * acquireProgram
        * returns cl::Program object by constructing
        * and compiling the program, or by returning the Program if
        * previously compiled.
        * Called from getKernels.
        * see bolt/cl/detail/scan.inl for example usage
        **********************************************************************/
    ::cl::Program acquireProgram(
        const ::cl::Context& context,
        const ::cl::Device&  device,
        const ::std::string& compileOptions,
        const ::std::string& completeKernelSource
        );

    /**********************************************************************
        * compileProgram
        * returns cl::Program object by constructing
        * and compiling the program.
        * Called from acquireProgram.
        * see bolt/cl/detail/scan.inl for example usage
        **********************************************************************/
    ::cl::Program compileProgram(
        const ::cl::Context& context,
        const ::cl::Device&  device,
        const ::std::string& compileOptions,
        const ::std::string& completeKernelSource,
        cl_int * err = NULL);


    void wait(const bolt::cl::control &ctl, ::cl::Event &e)
    {
        const bolt::cl::control::e_WaitMode waitMode = ctl.getWaitMode();
        if (waitMode == bolt::cl::control::BusyWait) {
            const ::cl::CommandQueue& q = ctl.getCommandQueue();
            q.flush();
            while (e.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() != CL_COMPLETE) {
                // spin here for fast completion detection...
            };
        } else if ((waitMode == bolt::cl::control::NiceWait) || (waitMode == bolt::cl::control::BalancedWait)) {
            cl_int l_Error = e.wait();
            V_OPENCL( l_Error, "wait call failed" );
        } else if (waitMode == bolt::cl::control::ClFinish) {
            const ::cl::CommandQueue& q = ctl.getCommandQueue();
            cl_int l_Error = q.finish();
            V_OPENCL( l_Error, "clFinish call failed" );
        }
    };

    /**************************************************************************
     * Compile Kernel from primitive information
     *************************************************************************/
    void printKernels(
        const ::std::vector< ::std::string >& kernelNames,
        const ::std::string& completeKernelString,
        const ::std::string& compileOptions)
    {
        std::cout << hr << std::endl;
        std::cout << "Kernel Names:" << std::endl;
        std::cout << hr << std::endl;
        std::for_each(kernelNames.begin(), kernelNames.end(), [&](const std::string& name) {
            std::cout << name << std::endl;
        });
        std::cout << hr << std::endl;
        std::cout << "Kernel String:" << std::endl;
        std::cout << hr << std::endl;
        std::cout << completeKernelString << std::endl;
        std::cout << hr << std::endl;
        std::cout << "Kernel Compile Options:" << std::endl;
        std::cout << hr << std::endl;
        std::cout << compileOptions << std::endl;
        std::cout << hr << std::endl;
    }

    /**************************************************************************
    * getKernels
    * - concatenates input strings into complete kernel string to be compiled
    * - takes into account control
    * - requests program/kernel from ProgramMap
    **************************************************************************/
    ::std::vector< ::cl::Kernel > getKernels(
        const control&      ctl,
        const std::vector<std::string>& typeNames,
        const KernelTemplateSpecializer * const kts,
        const std::vector<std::string>& typeDefs,
        const std::string&  kernelString,
        const std::string&  options )
    {
        std::string completeKernelString;
        /* In device vector.h functional.h and bolt.h the defintions of cl_* are given. These cl_* are typedef'd
         * to there corresponding types in cl_platforms.h. To the kernel Actually the cl_* are passed, But the OpenCL
           kernel does not understand cl_* So we need the below typdefinitions. */
        const std::string PreprocessorDefinitions =
        "#define cl_int    int\n"
        "#define cl_uint   unsigned int\n"
        "#define cl_short  short\n"
        "#define cl_ushort unsigned short\n"
        "#define cl_long   long\n"
        "#define cl_ulong  unsigned long\n"
        "#define cl_float  float\n"
        "#define cl_double double\n"
        "#define cl_char   char\n"
        "#define cl_uchar  unsigned char\n" ;

        completeKernelString = PreprocessorDefinitions;

        // (1) type definitions
        completeKernelString += "\n// Type Definitions\n";
        for (size_t i = 0; i < typeDefs.size(); i++)
        {
            completeKernelString += "\n" + typeDefs[i] + "\n";
        }

        // (2) raw kernel
        completeKernelString += "\n// Raw Kernel\n\n" + kernelString;

        //for ( std::set<std::string>::iterator iter = typeDefs.begin(); iter != typeDefs.end(); iter++ )
        //{
        //    std::cout << "concat " << *iter << std::endl;
        //    completeKernelString += "\n" + *iter + "\n";
        //}

        //std::for_each( typeDefs.begin( ), typeDefs.end( ), [&]( const std::string &typeDef )
        //{
        //    completeKernelString += "\n" + typeDef+ "\n";
        //});

        // (3) template specialization
        std::string templateSpecialization = (*kts)(typeNames);
        completeKernelString += "\n// Kernel Template Specialization\n" + templateSpecialization;

        // compile options
        std::string compileOptions = options;
        compileOptions += ctl.getCompileOptions( );
        compileOptions += " -x clc++ ";
        if (ctl.getDebugMode() & control::debug::SaveCompilerTemps) {
            compileOptions += " -save-temps=BOLT ";
        }
        if (ctl.getDebugMode() & control::debug::Compile) {
            printKernels(kts->getKernelNames(), completeKernelString, compileOptions);
        }

        // request program from program cache (ProgramMap)
        ::cl::Program program = acquireProgram(
            ctl.getContext(),
            ctl.getDevice(),
            compileOptions,
            completeKernelString);

        // retrieve kernels from program
        //std::cout << "Getting " << kts->numKernels() << " from program." << std::endl;
        ::std::vector< ::cl::Kernel > kernels;
        for (unsigned int i = 0; i < kts->numKernels() ; i++)
        {
            ::std::string name = kts->name(i);
            name += "Instantiated";
            try
            {
                cl_int l_err;
                ::cl::Kernel kernel(
                    program,
                    name.c_str(),
                    &l_err);
                V_OPENCL( l_err, "Kernel::constructor() failed" );
                kernels.push_back(kernel);
            }
            catch( const ::cl::Error& e)
            {
                std::cerr << hr << std::endl;
                std::cerr << es << "::cl::Kernel() in bolt::cl::acquireKernels()" << std::endl;
                std::cerr << es << "Error Code:   " << clErrorStringA(e.err()) << " (" << e.err() << ")" << std::endl;
                std::cerr << es << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
                std::cerr << es << "Error String: " << e.what() << std::endl;
                std::cerr << hr << std::endl;
            }
        }

        return kernels;
    }

    /**************************************************************************
     * aquireKernels
     * - returns kernels from ProgramMap if exist
     * - otherwise compiles program/kernels, adds to map, then returns
     *************************************************************************/
    ::cl::Program acquireProgram(
        const ::cl::Context& context,
        const ::cl::Device&  device,
        const ::std::string& options,
        const ::std::string& source)
    {
        // only one threads get to seach and retrieve-or-compile at a time
        boost::lock_guard< boost::mutex > lock( ::bolt::cl::programMapMutex ); // unlocks upon return
        cl_int l_err;

        // Does Program already exist?
        std::string deviceStr = device.getInfo< CL_DEVICE_NAME >( );
        deviceStr += "; " + device.getInfo< CL_DEVICE_VERSION >( );
        deviceStr += "; " + device.getInfo< CL_DEVICE_VENDOR >( );
        ProgramMapKey key = {context, deviceStr, options, source};
        ProgramMap::iterator iter = programMap.find( key );
        ::cl::Program program;

        // map does not yet contain desired program
        if( iter == programMap.end( ) )
        {
            program = ::bolt::cl::compileProgram(context, device, options, source, &l_err);
            V_OPENCL( l_err, "bolt::cl::compileProgram() failed" );
            ProgramMapValue value = { program };
            programMap.insert( std::make_pair( key, value ) );
        }
        else // map already contains desired kernel
        {
            program = iter->second.program;
        }
        return program;
    } // aquireProgram

    /**************************************************************************
    * compileProgram
    * - compiles OpenCL kernel string and returns Program object
    **************************************************************************/
    ::cl::Program compileProgram(
        const ::cl::Context& context,
        const ::cl::Device&  device,
        const ::std::string& options,
        const ::std::string& source,
        cl_int * err )
    {
        cl_int l_err;
        ::cl::Program program(context, source, false, &l_err);
        V_OPENCL( l_err, "Program::constructor() failed" );
        if (err != NULL) *err = l_err;
        try
        {
            std::vector< ::cl::Device > devices;
            devices.push_back(device);
            l_err = program.build(devices, options.c_str());
            V_OPENCL( l_err, "Program::build() failed" );
            if (err != NULL) *err = l_err;
        } catch(::cl::Error e) {
            std::cerr << hr << std::endl;
            std::cerr << es << "::cl::Program::build() in bolt::cl::compileProgram() failed." << std::endl;
            std::cerr << es << "Error Code:   " << clErrorStringA(e.err()) << "(" << e.err() << ")" << std::endl;
            std::cerr << es << "File:         " << __FILE__ << ", line " << __LINE__ << std::endl;
            std::cerr << es << "Error String: " << e.what() << std::endl;
            std::cerr << es << "Device:       " << device.getInfo<CL_DEVICE_NAME>() << "_"
                << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "cu_"
                << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz" << std::endl;
            std::cerr << es << "Status:       " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << std::endl;
            std::cerr << es << "Options:      " << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device) << std::endl;
            std::cerr << es << "Compile Log:"   << std::endl;
            std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            std::cerr << "[END COMPILE LOG]" << std::endl;
            std::cerr << hr << std::endl;
            std::cerr << es << "Kernel String:" << std::endl;
            std::cerr << source << std::endl;
            std::cerr << "[END KERNEL STRING]" << std::endl;
            std::cerr << hr << std::endl;
            if (err != NULL) *err = l_err;
            //throw;
        } // catch
        return program;
    } // compileProgram


        // externed in bolt.h
        boost::mutex programMapMutex;
        ProgramMap programMap;


    }; //namespace bolt::cl
}; // namespace bolt
