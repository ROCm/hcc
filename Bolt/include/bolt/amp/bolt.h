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

/*! \file bolt/amp/bolt.h
    \brief Define global functions for Bolt AMP.
*/


#pragma once
#if !defined( BOLT_AMP_BOLT_H )
#define BOLT_AMP_BOLT_H

#if defined(__APPLE__) || defined(__MACOSX)
#else
    #include <amp.h>
#endif

#include <string>
#include <map>
#include "bolt/BoltVersion.h"
#include "bolt/amp/control.h"

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
 *  \todo Add buffer pool for temporary memory allocated by Bolt calls
 *  \todo Review documentation for typos, clarity, etc
 *  \todo Add CPU implementations, i.e. link an external library or define our own CPU implementation
 *  \todo Statically link the Boost libraries into the Bolt library
 *  \todo Explain the cl_code parameter better, with possible use cases
 *  \todo Fix FAQ entry for device_vector; better explain when to use DV as opposed to host vectors
 *  \todo More examples of using Bolt with regular pointers, for instance float*'s
 *  \todo Develop auto-tuning framework for deciding to run on CPU vs GPU.
 *  \todo Specify limits on LDS usage?  How will a user know how much LDS space the functor code can allocate?  Discourage use.
 *  \todo Test Bolt functions for 'unusual' limits, for instance very large types which exhausts LDS.
 *  \todo Explore how asynchronous API's are implemented.  Load-balancing may drive desire to have async APIs
 *  \todo When moving our source to github, remove our build depenencies on internal servers
*/

namespace bolt {
    namespace amp {

        class control;


        /*! \brief Query the Bolt library for version information
            *  \details Return the major, minor and patch version numbers associated with the Bolt library
            *  \param[out] major Major functionality change
            *  \param[out] minor Minor functionality change
            *  \param[out] patch Bug fixes, documentation changes, no new features introduced
            */
        void getVersion( unsigned int& major, unsigned int& minor, unsigned int& patch );

        /*! \brief Translates an integer OpenCL error code to a std::string at runtime
        *  \param status The OpenCL error code
         * \return The error code stringized
        */
        //std::string clErrorStringA( const cl_int& status );

        /*! \brief Translates an integer OpenCL error code to a std::wstring at runtime
        *  \param status The OpenCL error code
         * \return The error code stringized
        */
        //std::wstring clErrorStringW( const cl_int& status );

        /*! \brief Helper print function to stringify OpenCL error codes
        *  \param res The OpenCL error code
        *  \param msg A relevant message to be printed out supplied by user
        *  \param lineno Source line number; not currently used
        *  \note std::exception is built to only use narrow text
        */
        /*
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
        */

		//void wait(const bolt::amp::control &ctl, ::cl::Event &e) ;


	};
};

#if defined( _WIN32 )
#define ALIGNED( bound ) __declspec( align( bound ) )
#else
#define ALIGNED( bound ) __attribute__ ( (aligned( bound ) ) )
#endif

#endif
