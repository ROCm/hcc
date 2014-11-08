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


/*! \file bolt/amp/control.h
    \brief Control the parameters of a specific Bolt algorithm call.
*/
#if !defined( BOLT_AMP_CONTROL_H )
#define BOLT_AMP_CONTROL_H

#pragma once

#include <amp.h>
#include <string>
#include <map>

namespace bolt
{
namespace amp
{

 /*! \addtogroup miscellaneous
        */

        /*! \addtogroup miscellaneous
        */

        /*! \addtogroup AMP-control
        * \ingroup miscellaneous
        * \{
        */

/*! The \p control class lets you control the parameters of a specific Bolt algorithm call,
 * such as the debug information, load-balancing with  the host, and more.  Each Bolt Algorithm call
 * accepts the
 * \p control class as an optional first argument.  Additionally, Bolt contains a global default
 * \p control structure that is used in cases where the \p control argument is not specified; also,
 * developers can modify this structure.  Some examples:

 * \code
 * ::Concurrency::accelerator accel(::Concurrency::accelerator::default_accelerator);
 * bolt::amp::control myControl(accel);
 * int sum = bolt::cl::reduce(myControl, ...);
 * \endcode

 * Developers also can inialize and save control structures to avoid the cost of copying the default on each function call.  An example:
 * \code
 * class MyClass {
 *   MyClass(...) {
 *      ::Concurrency::accelerator accel(::Concurrency::accelerator::default_accelerator);
 *      _boltControl.accelerator(accel);
 *   };
 *
 *   void runIt()
 *   {
 *     int sum = bolt::amp::reduce(_boltControl, ...);
 *   };
 *
 *   private:
 *		bolt::amp::control _boltControl;
 *   }
 * \endcode
 *
 * \{
 */
class control {
public:
    enum e_UseHostMode {
        NoUseHost,
        UseHost};
    enum e_RunMode {
        Automatic,
        SerialCpu,
        MultiCoreCpu,
        Gpu };

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

    enum e_WaitMode {
        BalancedWait,	// Balance of Busy and Nice: tries to use Busy for short-running kernels.  \todo: Balanced currently maps to nice.
        NiceWait,		// Use an OS semaphore to detect completion status.
        BusyWait,		// Busy a CPU core continuously monitoring results.  Lowest-latency, but requires a dedicated core.
        ClFinish,      // Call clFinish on the queue.
    };

public:

    // Construct a new control structure, copying from default control for arguments that are not overridden.
    control(
        Concurrency::accelerator accel=getDefault().getAccelerator(),
        e_UseHostMode useHost=getDefault().getUseHost(),
        unsigned debug=getDefault().getDebug()
        ):
        m_accelerator(accel),
        m_useHost(useHost),
        m_forceRunMode(getDefault().m_forceRunMode),
		m_defaultRunMode(getDefault().m_defaultRunMode),
        m_debug(debug),
        m_autoTune(getDefault().m_autoTune),
        m_wgPerComputeUnit(getDefault().m_wgPerComputeUnit),
        m_waitMode(getDefault().m_waitMode),
        m_unroll(getDefault().m_unroll)
    {};

    control( const control& ref) :
        m_accelerator(ref.m_accelerator),
        m_useHost(ref.m_useHost),
        m_forceRunMode(ref.m_forceRunMode),
		m_defaultRunMode(ref.m_defaultRunMode),
        m_debug(ref.m_debug),
        m_autoTune(ref.m_autoTune),
        m_wgPerComputeUnit(ref.m_wgPerComputeUnit),
        m_waitMode(ref.m_waitMode),
        m_unroll(ref.m_unroll)
    {
        //printf("control::copy construcor\n");
    };

    //setters:
    //! Set the AMP Accelerator for Bolt algorithms to use.
    void setAccelerator(::Concurrency::accelerator accel) { m_accelerator = accel; };

    //! If enabled, Bolt can use the host CPU to run parts of the algorithm.  If false, Bolt runs the
    //! entire algorithm using the device specified by the accelerator. This can be appropriate
    //! on a discrete GPU, where the input data is located on the device memory.
    void setUseHost(e_UseHostMode useHost) { m_useHost = useHost; };

    //! Force the Bolt command to run on the specifed device.  Default is "Automatic," in which case the Bolt
    //! runtime selects the device.  Forcing the mode to SerialCpu can be useful for debugging the algorithm.
    //! Forcing the mode can also be useful for performance comparisons, or for direct
    //! control over the run location (perhaps due to knowledge that the algorithm is best-suited for GPU).
    void setForceRunMode(e_RunMode forceRunMode) { m_forceRunMode = forceRunMode; };

    /*! Enable debug messages to be printed to stdout as the algorithm is compiled, run, and tuned.  See the #debug
     * namespace for a list of values.  Multiple debug options can be combined with the + sign, as in
     * following example.  Use this technique rather than separate calls to the debug() API;
     * each call resets the debug level, rather than merging with the existing debug() setting.
     * \code
     * bolt::amp::control myControl;
     * // Show example of combining two debug options with the '+' sign.
     * myControl.debug(bolt::amp::control::debug::Compile + bolt::amp::control:debug::SaveCompilerTemps);
     * \endcode
     */
    void setDebug(unsigned debug) { m_debug = debug; };

    /*! Set the work-groups-per-compute unit that will be used for reduction-style operations (reduce, transform_reduce).
        Higher numbers can hide latency by improving the occupancy but will increase the amount of data that
        has to be reduced in the final, less efficient step.  Experimentation may be required to find
        the optimal point for a given algorithm and device; typically 8-12 will deliver good results */
    void setWGPerComputeUnit(int wgPerComputeUnit) { m_wgPerComputeUnit = wgPerComputeUnit; };

    /*! Set the method used to detect completion at the end of a Bolt routine. */
    void setWaitMode(e_WaitMode waitMode) { m_waitMode = waitMode; };

    /*! unroll assignment */
    void setUnroll(int unroll) { m_unroll = unroll; };

    // getters:
    Concurrency::accelerator& getAccelerator( ) { return m_accelerator; };
    const Concurrency::accelerator& getAccelerator( ) const { return m_accelerator; };

    e_UseHostMode getUseHost() const { return m_useHost; };
    e_RunMode getForceRunMode() const { return m_forceRunMode; };
	e_RunMode getDefaultPathToRun() const { return m_defaultRunMode; };
    unsigned getDebug() const { return m_debug;};
    int const getWGPerComputeUnit() const { return m_wgPerComputeUnit; };
    e_WaitMode getWaitMode() const { return m_waitMode; };
    int getUnroll() const { return m_unroll; };

    /*!
     * Return default default \p control structure.  This is used for Bolt API calls when the user
     * does not explicitly specify a \p control structure.  Also, newly created \p control structures copy
     * the default structure for their initial values.  Note that changes to the default \p control structure
     * are not automatically copied to already-created control structures.  Typically, the default \p control
     * structure is modified as part of the application initialiation; then, as other \p control structures
     * are created, they pick up the modified defaults.  Some examples:
     * \code
     * bolt::amp::control myControl = bolt::cl::getDefault();  // copy existing default control.
     * bolt::amp::control myControl;  // same as last line - the constructor also copies values from the default control
     *
     * // Modify a setting in the default \p control
     * bolt::amp::control::getDefault().compileOptions("-g");
     * \endcode
     */
    static control &getDefault()
    {
        // Default control structure; this can be accessed by the bolt::cl::control::getDefault()
        static control _defaultControl( true );
        return _defaultControl;
    };

    //TODO - implement the below function in control.cpp
    /*static void printPlatforms( bool printDevices = true, cl_device_type deviceType = CL_DEVICE_TYPE_ALL );
    static void printPlatformsRange( std::vector< ::cl::Platform >::iterator begin, std::vector< ::cl::Platform >::iterator end,
                                    bool printDevices = true, cl_device_type deviceType = CL_DEVICE_TYPE_ALL );*/

private:

    // This is the private constructor is only used to create the initial default control structure.
    control(bool createGlobal) :
        m_accelerator( Concurrency::accelerator::default_accelerator ),
        m_useHost(UseHost),
        m_forceRunMode(Automatic),
        m_debug(debug::None),
        m_autoTune(AutoTuneAll),
        m_wgPerComputeUnit(8),
        m_waitMode(BusyWait),
        m_unroll(1)
    {

		if(m_accelerator.default_accelerator == NULL)
        {
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
            m_forceRunMode   = Gpu;
			m_defaultRunMode = Gpu;
        }
	
	
	};

    //::cl::CommandQueue  m_commandQueue;
    ::Concurrency::accelerator m_accelerator;
    e_UseHostMode       m_useHost;
    e_RunMode           m_forceRunMode;
	e_RunMode           m_defaultRunMode;
    e_AutoTuneMode      m_autoTune;  /* auto-tune the choice of device CPU/GPU and  workgroup shape */
    unsigned            m_debug;
    int                 m_wgPerComputeUnit;
    e_WaitMode          m_waitMode;
    int                 m_unroll;
};
};
};

// Implementor note:
// When adding a new field to this structure, don't forget to:
//   * Add the new field, ie "int _foo.
//   * Add setter function and getter function, ie "void foo(int fooValue)" and "int foo const { return _foo; }"
//   * Add the field to the private constructor.  This is used to set the global default "_defaultControl".
//   * Add the field to the public constructor, copying from the _defaultControl.

// Sample usage:
// Concurrency::accelerator::default_accelerator
// bolt::amp::control ctl(Concurrency::accelerator::default_accelerator);
// c.debug(bolt::amp::control::ShowCompile);
// bolt::amp::reduce(ctl, a.begin(), a.end(), std::plus<int>);


#endif
