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


/*! \file bolt/cl/control.h
    \brief Control the parameters of a specific Bolt algorithm call.
*/

#pragma once
#if !defined( BOLT_CL_CONTROL_H )
#define BOLT_CL_CONTROL_H


#include <bolt/cl/bolt.h>
#include <string>
#include <map>

#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/shared_ptr.hpp>

/*! \file control.h
*/


namespace bolt {
    namespace cl {

        /*! \addtogroup miscellaneous
        */

        /*! \addtogroup miscellaneous
        */

        /*! \addtogroup CL-control
        * \ingroup miscellaneous
        * \{
        */

        /*! The \p control class lets you control the parameters of a specific Bolt algorithm call,
         such as the command-queue where GPU kernels run, debug information, load-balancing with
         the host, and more.  Each Bolt Algorithm call accepts the
        \p control class as an optional first argument.  Additionally, Bolt contains a global default
        \p control structure that is used in cases where the \p control argument is not specified; also,
        developers can modify this structure.  Some examples:

        * \code
        * cl::CommandQueue myCommandQueue = ...
        * bolt::cl::control myControl;
        * myControl.commandQueue(myCommandQueue);
        * int sum = bolt::cl::reduce(myControl, ...);
        * \endcode


        * Developers also can inialize and save control structures to avoid the cost of copying the default on each function call.  An example:
        * \code
        * class MyClass {
        *   MyClass(...) {
        *      cl::CommandQueue myCommandQueue = ...
        *      _boltControl.commandQueue(myCommandQueue);
        *   };
        *
        *   void runIt()
        *   {
        *     int sum = bolt::cl::reduce(_boltControl, ...);
        *   };
        *
        *   private:
        *		bolt::cl::control _boltControl;
        *   }
        * \endcode


        * Sometimes, it can be useful to set the global default \p control structure used by Bolt algorithms
        * calls that do not explicitly specify
        * a control parameter as the first argument.  For example, the application initialization routine can examine
        * all the available GPU devices and select the one to be used for all subsequent Bolt calls.  This can be
        * done by writing the global default \p control structure, i.e.:
        * \code
        * cl::CommandQueue myCommandQueue = ...
        * bolt::cl::control::getDefault().commandQueue(myCommandQueue);
        * bolt::cl::control::getDefault().debug(bolt::cl::control::debug:SaveCompilerTemps);
        * ...
        * \endcode
        *
         * \{
         */
        class control {
        public:
            enum e_UseHostMode {NoUseHost, UseHost};
            enum e_RunMode     {Automatic,
                                SerialCpu,
                                MultiCoreCpu,
                                OpenCL };

            enum e_AutoTuneMode{NoAutoTune=0x0,
                                AutoTuneDevice=0x1,
                                AutoTuneWorkShape=0x2,
                                AutoTuneAll=0x3}; // FIXME, experimental
            struct debug {
                static const unsigned None=0;
                static const unsigned Compile = 0x1;
                static const unsigned ShowCode = 0x2;
                static const unsigned SaveCompilerTemps = 0x4;
                static const unsigned DebugKernelRun = 0x8;
                static const unsigned AutoTune = 0x10;
            };

            enum e_WaitMode {BalancedWait,	// Balance of Busy and Nice: tries to use Busy for short-running kernels.  \todo: Balanced currently maps to nice.
                             NiceWait,		// Use an OS semaphore to detect completion status.
                             BusyWait,		// Busy a CPU core continuously monitoring results.  Lowest-latency, but requires a dedicated core.
                             ClFinish,      // Call clFinish on the queue.
            };

        public:

            // Construct a new control structure, copying from default control for arguments that are not overridden.
            control(
                const ::cl::CommandQueue& commandQueue = getDefault().getCommandQueue(),
                e_UseHostMode useHost=getDefault().getUseHost(),
                unsigned debug=getDefault().getDebugMode()
                ) :
            m_commandQueue(commandQueue),
                m_useHost(useHost),
                m_forceRunMode(OpenCL),   //Replaced this with automatic because the default is not MultiCoreCPU if no GPU is found
                m_defaultRunMode(OpenCL),
                m_debug(debug),
                m_autoTune(getDefault().m_autoTune),
                m_wgPerComputeUnit(getDefault().m_wgPerComputeUnit),
                m_compileOptions(getDefault().m_compileOptions),
                m_compileForAllDevices(getDefault().m_compileForAllDevices),
                m_waitMode(getDefault().m_waitMode),
                m_unroll(getDefault().m_unroll)
            {};


            control( const control& ref) :
                m_commandQueue(ref.m_commandQueue),
                m_useHost(ref.m_useHost),
                m_forceRunMode(ref.m_forceRunMode),
                m_defaultRunMode(ref.m_defaultRunMode),
                m_debug(ref.m_debug),
                m_autoTune(ref.m_autoTune),
                m_wgPerComputeUnit(ref.m_wgPerComputeUnit),
                m_compileOptions(ref.m_compileOptions),
                m_compileForAllDevices(ref.m_compileForAllDevices),
                m_waitMode(ref.m_waitMode),
                m_unroll(ref.m_unroll)
            {
                //printf("control::copy construcor\n");
            };

            //setters:
            //! Set the OpenCL command queue (and associated device) for Bolt algorithms to use.
            //! Only one command-queue can be specified for each call; Bolt does not load-balance across
            //! multiple command queues.  Bolt also uses the specified command queue to determine the OpenCL context and
            //! device.
            void setCommandQueue(::cl::CommandQueue commandQueue) { m_commandQueue = commandQueue; };

            //! If enabled, Bolt can use the host CPU to run parts of the algorithm.  If false, Bolt runs the
            //! entire algorithm using the device specified by the command-queue. This can be appropriate
            //! on a discrete GPU, where the input data is located on the device memory.
            void setUseHost(e_UseHostMode useHost) { m_useHost = useHost; };


            //! Force the Bolt command to run on the specifed device.  Default is "Automatic," in which case the Bolt
            //! runtime selects the device.  Forcing the mode to SerialCpu can be useful for debugging the algorithm.
            //! Forcing the mode can also be useful for performance comparisons, or for direct
            //! control over the run location (perhaps due to knowledge that the algorithm is best-suited for GPU).
            //! Please note that forcing the run modes will not change the OpenCL device in the control object. This
            //! API is designed to simplify the process of choosing the appropriate path in the Bolt API.
            void setForceRunMode(e_RunMode forceRunMode) { m_forceRunMode = forceRunMode; };

            /*! Enable debug messages to be printed to stdout as the algorithm is compiled, run, and tuned.  See the #debug
            * namespace for a list of values.  Multiple debug options can be combined with the + sign, as in
            * following example.  Use this technique rather than separate calls to the debug() API;
            * each call resets the debug level, rather than merging with the existing debug() setting.
            * \code
            * bolt::cl::control myControl;
            * // Show example of combining two debug options with the '+' sign.
            * myControl.setDebug(bolt::cl::control::debug::Compile + bolt::cl::control:debug::SaveCompilerTemps);
            * \endcode
            */
            void setDebugMode(unsigned debug) { m_debug = debug; };

            /*! Set the work-groups-per-compute unit that will be used for reduction-style operations (reduce, transform_reduce).
                Higher numbers can hide latency by improving the occupancy but will increase the amoutn of data that
                has to be reduced in the final, less efficient step.  Experimentation may be required to find
                the optimal point for a given algorithm and device; typically 8-12 will deliver good results */
            void setWGPerComputeUnit(int wgPerComputeUnit) { m_wgPerComputeUnit = wgPerComputeUnit; };

            /*! Set the method used to detect completion at the end of a Bolt routine. */
            void setWaitMode(e_WaitMode waitMode) { m_waitMode = waitMode; };

            /*! unroll assignment */
            void setUnroll(int unroll) { m_unroll = unroll; };

            //!
            //! Specify the compile options passed to the OpenCL(TM) compiler.
            void setCompileOptions(std::string &compileOptions) { m_compileOptions = compileOptions; };

            // getters:
            ::cl::CommandQueue&         getCommandQueue( ) { return m_commandQueue; };
            const ::cl::CommandQueue&   getCommandQueue( ) const { return m_commandQueue; };
            ::cl::Context               getContext() const { return m_commandQueue.getInfo<CL_QUEUE_CONTEXT>();};
            ::cl::Device                getDevice() const { return m_commandQueue.getInfo<CL_QUEUE_DEVICE>();};
            e_UseHostMode               getUseHost() const { return m_useHost; };
            e_RunMode                   getForceRunMode() const { return m_forceRunMode; };
            e_RunMode                   getDefaultPathToRun() const { return m_defaultRunMode; };
            unsigned                    getDebugMode() const { return m_debug;};
            int const                   getWGPerComputeUnit() const { return m_wgPerComputeUnit; };
            const ::std::string         getCompileOptions() const { return m_compileOptions; };
            e_WaitMode                  getWaitMode() const { return m_waitMode; };
            int                         getUnroll() const { return m_unroll; };
            bool                        getCompileForAllDevices() const { return m_compileForAllDevices; };

            /*!
              * Return default default \p control structure.  This is used for Bolt API calls when the user
              * does not explicitly specify a \p control structure.  Also, newly created \p control structures copy
              * the default structure for their initial values.  Note that changes to the default \p control structure
              * are not automatically copied to already-created control structures.  Typically, the default \p control
              * structure is modified as part of the application initialiation; then, as other \p control structures
              * are created, they pick up the modified defaults.  Some examples:
              * \code
              * bolt::cl::control myControl = bolt::cl::getDefault();  // copy existing default control.
              * bolt::cl::control myControl;  // same as last line - the constructor also copies values from the default control
              *
              * // Modify a setting in the default \p control
              * bolt::cl::control::getDefault().compileOptions("-g");
              * \endcode
              */
            static control &getDefault()
            {
                // Default control structure; this can be accessed by the bolt::cl::control::getDefault()
                static control _defaultControl( true );
                return _defaultControl;
            };

            static void printPlatforms( bool printDevices = true, cl_device_type deviceType = CL_DEVICE_TYPE_ALL );
            static void printPlatformsRange( std::vector< ::cl::Platform >::iterator begin, std::vector< ::cl::Platform >::iterator end,
                                            bool printDevices = true, cl_device_type deviceType = CL_DEVICE_TYPE_ALL );

               /*! \brief Convenience method to help users create and initialize an OpenCL CommandQueue.
                * \todo The default commandqueue is created with a context that contains all GPU devices in platform.  Since kernels
                * are only compiled on first invocation, switching between GPU devices is OK, but switching to a CPU
                * device afterwards causes an exception because the kernel was not compiled for CPU.  Should we provide
                * more options and expose more intefaces to the user?
                */
            static ::cl::CommandQueue getDefaultCommandQueue( );

            /*! \brief Buffer pool support functions
             */
            typedef boost::shared_ptr< ::cl::Buffer > buffPointer;

            /*! Return device memory size */
            size_t totalBufferSize( );
            /*! Return a pointer to memory from per allocated memory pool */
            buffPointer acquireBuffer( size_t reqSize, cl_mem_flags flags = CL_MEM_READ_WRITE, const void* host_ptr = NULL );
            /*! Freeing memory*/
            void freeBuffers( );

        private:

            // This is the private constructor is only used to create the initial default control structure.
            control(bool createGlobal) :
                m_commandQueue( getDefaultCommandQueue( ) ),
                m_useHost(UseHost),
                m_debug(debug::None),
                m_autoTune(AutoTuneAll),
                m_wgPerComputeUnit(8),
                m_compileForAllDevices(true),
                m_waitMode(BusyWait),
                m_unroll(1)
            {
                ::cl_device_type dType = CL_DEVICE_TYPE_CPU;
                if(m_commandQueue() != NULL)
                {
                    ::cl::Device device = m_commandQueue.getInfo<CL_QUEUE_DEVICE>();
                    dType = device.getInfo<CL_DEVICE_TYPE>();
                }
                if(dType == CL_DEVICE_TYPE_CPU || m_commandQueue() == NULL)
                {
                    //m_commandQueue will be NULL if no platforms are found and
                    //if a non AMD paltform is found but cound not enumerate any CPU device
#ifdef ENABLE_TBB
                    m_forceRunMode = MultiCoreCpu;
                    m_defaultRunMode = MultiCoreCpu;
#else
                    m_forceRunMode = SerialCpu;
                    m_defaultRunMode = SerialCpu;
#endif
                }
                else
                {
                    //If dType = CL_DEVICE_TYPE_GPU
                    m_forceRunMode   = OpenCL;
                    m_defaultRunMode = OpenCL;
                }
            };

            ::cl::CommandQueue  m_commandQueue;
            e_UseHostMode       m_useHost;
            e_RunMode           m_forceRunMode;
            e_RunMode           m_defaultRunMode;
            e_AutoTuneMode      m_autoTune;  /* auto-tune the choice of device CPU/GPU and  workgroup shape */
            unsigned            m_debug;
            int                 m_wgPerComputeUnit;
            ::std::string       m_compileOptions;  // extra options to pass to OpenCL compiler.
            bool                m_compileForAllDevices;  // compile for all devices in the context.  False means to only compile for specified device.
            e_WaitMode          m_waitMode;
            int                 m_unroll;

            struct descBufferKey
            {
                ::cl::Context buffContext;
                cl_mem_flags memFlags;
                const void* host_ptr;
            };

            struct descBufferValue
            {
                size_t buffSize;
                bool inUse;
                ::cl::Buffer buffBuff;
            };

            struct descBufferComp
            {
                bool operator( )( const descBufferKey& lhs, const descBufferKey& rhs ) const
                {
                    if( lhs.memFlags < rhs.memFlags )
                    {
                        return true;
                    }
                    else if( lhs.memFlags == rhs.memFlags )
                    {
                        if( lhs.buffContext( ) < rhs.buffContext( ) )
                        {
                            return true;
                        }
                        else if( lhs.buffContext( ) == rhs.buffContext( ) )
                        {
                            if( lhs.host_ptr < rhs.host_ptr )
                            {
                                return true;
                            }
                            else
                            {
                                return false;
                            }
                        }
                        else
                        {
                            return false;
                        }
                    }
                    else
                    {
                        return false;
                    }
                }
            };

            typedef std::multimap< descBufferKey, descBufferValue, descBufferComp > mapBufferType;

            /*! \brief Class used with shared_ptr<> as a custom deleter, to signal to the context object when
             * a buffer is finished being used by a client.  We want to remove the ability to destroy the buffer
             * from the caller; only the context object shall control the lifetime of these scratch
             * buffers.  This most likely will happen in the default destructor
             * generated by the smart_ptr<> class, however, a client also has the option of calling reset()
             * directly on the smart_ptr<>.  In order for this class to work, the iterator that we store
             * MUST NOT BE INVALIDATED BY INSERTIONS OR DELETIONS INTO THE UNDERLYING CONTAINER
            */
            class UnlockBuffer
            {
                mapBufferType::iterator m_iter;
                control& m_control;

            public:
                //  Basic constructor requires a reference to the container and a positional element
                UnlockBuffer( control& p_control, mapBufferType::iterator it ): m_iter( it ), m_control( p_control )
                {}

                void operator( )( const void* pBuff )
                {
                    //  TODO: I think a general mutex is overkill here; we should try to use an interlocked instruction to modify the
                    //  inUse flag
                    boost::lock_guard< boost::mutex > lock( m_control.mapGuard );
                    m_iter->second.inUse = false;
                }
            };

            friend class UnlockBuffer;
            mapBufferType mapBuffer;
            boost::mutex mapGuard;

        }; // end class control

    };
};


// Implementor note:
// When adding a new field to this structure, don't forget to:
//   * Add the new field, ie "int _foo.
//   * Add setter function and getter function, ie "void foo(int fooValue)" and "int foo const { return _foo; }"
//   * Add the field to the private constructor.  This is used to set the global default "_defaultControl".
//   * Add the field to the public constructor, copying from the _defaultControl.

// Sample usage:
// bolt::control c(myCmdQueue);
// c.debug(bolt::control::ShowCompile);
// bolt::cl::reduce(c, a.begin(), a.end(), std::plus<int>);
//
//
// reduce (bolt::control(myCmdQueue),

#endif
