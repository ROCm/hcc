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


/*! \file bolt/cl/bolt.h
    \brief Define global functions for Bolt CL.
*/

#pragma once
#if !defined( BOLT_CL_BOLT_H )
#define BOLT_CL_BOLT_H
#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <CL/cl.h>
/*For enabling only the OpenCL 1.1 specification uncomment the following line*/
//#undef CL_VERSION_1_2
#include <CL/cl.hpp>


#include <string>
#include <map>
#include <boost/thread/mutex.hpp>
#include "bolt/BoltVersion.h"
#include "bolt/cl/control.h"
#include "bolt/cl/clcode.h"

#define PUSH_BACK_UNIQUE(CONTAINER, ELEMENT) \
    if (std::find(CONTAINER.begin(), CONTAINER.end(), ELEMENT) == CONTAINER.end()) \
        CONTAINER.push_back(ELEMENT);

/*! \file bolt.h
 *  \brief Main public header file defining global functions for Bolt
 *  \todo Develop googletest framework for count
 *  \todo Develop googletest framework for reduce
 *  \todo Follow the coding guideline for expanding tabs to spaces, max line char width of 120 chars
 *  \todo Add support for vs2008
 *  \todo Add support for linux/mingw
 *  \todo Review the the use of parameters to the Bolt API; should parameters for chained functions past
 *  the public API be references?  Iterators and everything.
 *  \todo Add CPU implementations, i.e. link an external library or define our own CPU implementation
 *  \todo Statically link the Boost libraries into the Bolt library
 *  \todo Explain the cl_code parameter better, with possible use cases
 *  \todo Fix FAQ entry for device_vector; better explain when to use DV as opposed to host vectors
 *  \todo More examples of using Bolt with regular pointers, for instance float*'s
 *  \todo Develop auto-tuning framework for deciding to run on CPU vs GPU.
 *  \todo Specify limits on LDS usage?  How will a user know how much LDS space the functor code can allocate?  Discourage use.
 *  \todo Test Bolt functions for 'unusual' limits, for instance very large types which exhausts LDS.
 *  \todo Explore how asynchronous API's are implemented.  Load-balancing may drive desire to have async APIs
 *  \todo Move the *.cl files to the <bolt_root>/bolt subdirectory
 *  \todo When moving our source to github, remove our build depenencies on internal servers
*/

namespace bolt {
    namespace cl {

        extern const std::string binary_search_kernels;
        extern const std::string copy_kernels;
        extern const std::string count_kernels;
        extern const std::string fill_kernels;
        extern const std::string gather_kernels;
        extern const std::string generate_kernels;
        extern const std::string merge_kernels;
        extern const std::string min_element_kernels;
        extern const std::string reduce_kernels;
        extern const std::string reduce_by_key_kernels;
        extern const std::string scan_kernels;
        extern const std::string scan_by_key_kernels;
        extern const std::string scatter_kernels;
        extern const std::string sort_kernels;
        extern const std::string stablesort_kernels;
        extern const std::string stablesort_by_key_kernels;
        extern const std::string sort_uint_kernels;
        extern const std::string sort_int_kernels;
        extern const std::string sort_common_kernels;
        extern const std::string sort_by_key_kernels;
        extern const std::string sort_by_key_int_kernels;
        extern const std::string sort_by_key_uint_kernels;
        extern const std::string transform_kernels;
        extern const std::string transform_reduce_kernels;
        extern const std::string transform_scan_kernels;

        // transform_scan kernel names
        //static std::string transform_scan_kernel_names_array[] = { "perBlockTransformScan", "intraBlockInclusiveScan", "perBlockAddition" };
        //const std::vector<std::string> transformScanKernelNames(transform_scan_kernel_names_array, transform_scan_kernel_names_array+3);

        /******************************************************************
         * Kernel Template Specialization
         *****************************************************************/
        class KernelTemplateSpecializer
        {
            public:
                // kernel template specializer functor
                virtual const ::std::string operator() (const ::std::vector< ::std::string >& typeNames) const
                { return "Error; virtual function not overloaded"; }

                // add a kernel name
                void addKernelName( const std::string& kernelName) { kernelNames.push_back(kernelName); }

                // get the name of a particular kernel
                const ::std::string name( int kernelIndex ) const { return kernelNames[ kernelIndex ]; }

                // return number of kernels
                size_t numKernels() const { return kernelNames.size(); }

                // kernel vector
                const ::std::vector< ::std::string > getKernelNames() const { return kernelNames; }

            public:
                ::std::vector< ::std::string > kernelNames;
        };

        class control;
        //class KernelTemplateSpecializer;

        extern std::string fileToString(const std::string &fileName);

        /**********************************************************************
         * getKernels
         * returns vector of cl::Kernel objects either by constructing
         * and compiling the kernels, or by returning the kernels if
         * previously compiled.
         * see bolt/cl/detail/scan.inl for example usage
         **********************************************************************/
        ::std::vector< ::cl::Kernel > getKernels(
            const control&      ctl,
            const ::std::vector< ::std::string >& typeNames,
            const KernelTemplateSpecializer * const kts,
            const ::std::vector< ::std::string >& typeDefinitions,
            const std::string&  baseKernelString,
            const std::string&  compileOptions = ""
                 );

        /*! \brief Query the Bolt library for version information
            *  \details Return the major, minor and patch version numbers associated with the Bolt library
            *  \param[out] major Major functionality change
            *  \param[out] minor Minor functionality change
            *  \param[out] patch Bug fixes, documentation changes, no new features introduced
            */
        void getVersion( cl_uint& major, cl_uint& minor, cl_uint& patch );

        /*! \brief Translates an integer OpenCL error code to a std::string at runtime
        *  \param status The OpenCL error code
         * \return The error code stringized
        */
        std::string clErrorStringA( const cl_int& status );

        /*! \brief Translates an integer OpenCL error code to a std::wstring at runtime
        *  \param status The OpenCL error code
         * \return The error code stringized
        */
        std::wstring clErrorStringW( const cl_int& status );

        /*! \brief Helper print function to stringify OpenCL error codes
        *  \param res The OpenCL error code
        *  \param msg A relevant message to be printed out supplied by user
        *  \param lineno Source line number; not currently used
        *  \note std::exception is built to only use narrow text
        */
        inline cl_int V_OpenCL( cl_int res, const std::string& msg, size_t lineno )
        {
            switch( res )
            {
                case    CL_SUCCESS:
                    break;
                default:
                {
                    std::string tmp;
                    tmp.append( "V_OpenCL< " );
                    tmp.append( clErrorStringA( res ) );
                    tmp.append( " >: " );
                    tmp.append( msg );
                    //std::cout << tmp << std::endl;
                    throw ::cl::Error( res, tmp.c_str( ) );
                }
            }

            return	res;
        }
        #define V_OPENCL( status, message ) V_OpenCL( status, message, __LINE__ )

        void wait( const bolt::cl::control &ctl, ::cl::Event &e );

        /******************************************************************
         * Program Map - so each kernel is only compiled once
         *****************************************************************/
        /*! \brief This structure ensures that a kernel is compiled only once for specified devices.
        */
        struct ProgramMapKey
        {
            ::cl::Context context;
            ::std::string device;
            ::std::string compileOptions;
            ::std::string kernelSource;
        };

        struct ProgramMapValue
        {
            ::cl::Program program;
        };

        struct ProgramMapKeyComp
        {
            bool operator( )( const ProgramMapKey& lhs, const ProgramMapKey& rhs ) const
            {
                int comparison;
                // context
                // Do I really need to compare the context? Yes, required by OpenCL. -DT
                if( lhs.context() < rhs.context() )
                    return true;
                else if( lhs.context() > rhs.context() )
                    return false;
                // else equal; compare using next element of key

                // device
                comparison = lhs.device.compare(rhs.device);
                //std::cout << "Compare Device: " << comparison << std::endl;
                if( comparison < 0 )
                {
                    return true;
                }
                else if( comparison > 0 )
                {
                    return false;
                }
                // else equal; compare using next element of key

                // compileOptions
                comparison = lhs.compileOptions.compare(rhs.compileOptions);
                //std::cout << "Compare Options: " << comparison << std::endl;
                if( comparison < 0 )
                {
                    return true;
                }
                else if( comparison > 0 )
                {
                    return false;
                }
                //else
                //    std::cout << "<" << lhs.compileOptions << "> == <" << rhs.compileOptions << ">" << std::endl;
                // else equal; compare using next element of key

                // kernelSource
                comparison = lhs.kernelSource.compare(rhs.kernelSource);
                //std::cout << "Compare Source: " << comparison << std::endl;
                if( comparison < 0 )
                    return true;
                else if( comparison > 0 )
                    return false;
                //else
                //    std::cout << "<lhs.kernelSource> == <rhs.kernelSource>" << std::endl;
                // else equal; compare using next element of key

                // all elements equal
                return false;
            }
        };

        typedef ::std::map< ProgramMapKey, ProgramMapValue, ProgramMapKeyComp > ProgramMap;
        //typedef ::std::map< ::std::string, ProgramMapValue> ProgramMap;

        // declared in bolt.cpp
        extern boost::mutex programMapMutex;
        extern ProgramMap programMap;

    };
};

#if defined( _WIN32 )
#define ALIGNED( bound ) __declspec( align( bound ) )
#else
#define ALIGNED( bound ) __attribute__ ( (aligned( bound ) ) )
#endif

//Visual Studio 2012 is not able to map char to cl_char. Hence this typename is added.
BOLT_CREATE_TYPENAME( char );

BOLT_CREATE_TYPENAME( cl_char );   
BOLT_CREATE_TYPENAME( cl_uchar );  
BOLT_CREATE_TYPENAME( cl_short ); 
BOLT_CREATE_TYPENAME( cl_ushort );
BOLT_CREATE_TYPENAME( cl_int );   
BOLT_CREATE_TYPENAME( cl_uint );  
BOLT_CREATE_TYPENAME( cl_long );  
BOLT_CREATE_TYPENAME( cl_ulong ); 
BOLT_CREATE_TYPENAME( cl_float ); 
BOLT_CREATE_TYPENAME( cl_double );

////  Pre-define standard primitives that are likely to be used in a variety of OpenCL kernels
//BOLT_CREATE_TYPENAME( cl_int );
//BOLT_CREATE_CLCODE( cl_int, "int" );
//
//BOLT_CREATE_TYPENAME( cl_int2 );
//BOLT_CREATE_CLCODE( cl_int2, "int2" );
//
//BOLT_CREATE_TYPENAME( cl_int4 );
//BOLT_CREATE_CLCODE( cl_int4, "int4" );
//
//BOLT_CREATE_TYPENAME( cl_uint );
//BOLT_CREATE_CLCODE( cl_uint, "uint" );
//
//BOLT_CREATE_TYPENAME( cl_uint2 );
//BOLT_CREATE_CLCODE( cl_uint2, "uint2" );
//
//BOLT_CREATE_TYPENAME( cl_uint4 );
//BOLT_CREATE_CLCODE( cl_uint4, "uint4" );
//
//BOLT_CREATE_TYPENAME( cl_float );
//BOLT_CREATE_CLCODE( cl_float, "float" );
//
//BOLT_CREATE_TYPENAME( cl_float2 );
//BOLT_CREATE_CLCODE( cl_float2, "float2" );
//
//BOLT_CREATE_TYPENAME( cl_float4 );
//BOLT_CREATE_CLCODE( cl_float4, "float4" );
//
//BOLT_CREATE_TYPENAME( cl_double );
//BOLT_CREATE_CLCODE( cl_double, "double" );
//
//BOLT_CREATE_TYPENAME( cl_double2 );
//BOLT_CREATE_CLCODE( cl_double2, "double2" );
//
//BOLT_CREATE_TYPENAME( cl_double4 );
//BOLT_CREATE_CLCODE( cl_double4, "double4" );

#endif
