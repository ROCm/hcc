############################################################################                                                                                     
#   Copyright 2012 - 2013 Advanced Micro Devices, Inc.                                     
#                                                                                    
#   Licensed under the Apache License, Version 2.0 (the "License");   
#   you may not use this file except in compliance with the License.                 
#   You may obtain a copy of the License at                                          
#                                                                                    
#       http://www.apache.org/licenses/LICENSE-2.0                      
#                                                                                    
#   Unless required by applicable law or agreed to in writing, software              
#   distributed under the License is distributed on an "AS IS" BASIS,              
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         
#   See the License for the specific language governing permissions and              
#   limitations under the License.                                                   

############################################################################                                                                                     

# We require 2.8.3 for windows because of a bug in cmake that prevented vs2010 from generating
# executables properly with multiple periods
cmake_minimum_required( VERSION 2.8.3 )

# uncomment these to debug nmake and borland makefiles
#SET(CMAKE_START_TEMP_FILE "")
#SET(CMAKE_END_TEMP_FILE "")
#SET(CMAKE_VERBOSE_MAKEFILE 1)

# This becomes the name of the solution file
project( Benchmark )


list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
# Need to check out CMAKE_RUNTIME_OUTPUT_DIRECTORY variable, and see if that eliminates the need to modify install path
#if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
#    set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
#endif( )

# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )

if( MSVC_IDE )
    set( Bolt_BUILD64 ${CMAKE_CL_64} )
#    set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
#    option( Bolt_BUILD64 "Build a 64-bit product" ON )
 set( CMAKE_BUILD_TYPE ${BOLT_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE )   
endif()
 message( STATUS "64bit build :" ${Bolt_BUILD64} )
 message( STATUS "Build Type R/D:" ${CMAKE_BUILD_TYPE} )

# Modify the global find property to help us find libraries like Boost in the correct paths for 64-bit
# Essentially, find_library calls will look for /lib64 instead of /lib; works for windows and linux
if( Bolt_BUILD64 )
    set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
    message( STATUS "64bit build - FIND_LIBRARY_USE_LIB64_PATHS TRUE" )
else()
    set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE )
    message( STATUS "32bit build - FIND_LIBRARY_USE_LIB64_PATHS FALSE" )
endif()

# Various options below on configuring the build, and how to generate the solution files

if ( MSVC )
    set(Bolt_ampDefault OFF)
else ()
    set(Bolt_ampDefault OFF)
endif() 
option( BUILD_ampBolt "Create a solution that compiles Bolt for AMP"  ${Bolt_ampDefault})
option( BUILD_clBolt "Create a solution that compiles Bolt for OpenCL" ON )
# Currently, linux has a problem outputing both narrow and wide characters,
# which happens in our client because openCL only supports narrow characters
if( WIN32 )
    option( BUILD_UNICODE "Create a solution that compiles Bolt with Unicode Support" ON )
    if( BUILD_UNICODE )
        message( STATUS "UNICODE build" )
    endif( )
else()
    set( BUILD_UNICODE OFF )
    message( STATUS "UNICODE disabled on linux" )
endif()

# FFLAGS depend on the compiler, grab the compiler name from the path
get_filename_component( C_COMPILER_NAME ${CMAKE_C_COMPILER} NAME_WE )
# message( "C_COMPILER_NAME: " ${C_COMPILER_NAME} )
# message( "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER} )

# Set common compile and link options
if( C_COMPILER_NAME STREQUAL "cl" )
    # Following options for nMake
    message( STATUS "Detected MSVS Ver: " ${MSVC_VERSION} )
    
    # CMake uses huge stack frames for windows, for some reason.  We remove.
    string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" )
    string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
    string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}" ) 
elseif( (C_COMPILER_NAME STREQUAL "gcc") OR (CMAKE_COMPILER_IS_GNUCXX GREATER 0))
    message( STATUS "Detected GNU fortran compiler." )
    set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
    if( Bolt_BUILD64 )
        set( CMAKE_CXX_FLAGS "-m64 ${CMAKE_CXX_FLAGS}" )
        set( CMAKE_C_FLAGS "-m64 ${CMAKE_C_FLAGS}" )
        set( CMAKE_EXE_LINKER_FLAGS " -m64 ${CMAKE_EXE_LINKER_FLAGS}" )
        set( CMAKE_SHARED_LINKER_FLAGS "-lpthread -m64 ${CMAKE_SHARED_LINKER_FLAGS}" )
    else( )
        set( CMAKE_CXX_FLAGS "-m32 -msse2  ${CMAKE_CXX_FLAGS}" )
        set( CMAKE_C_FLAGS "-m32 -msse2 ${CMAKE_C_FLAGS}" )
	set( CMAKE_SHARED_LINKER_FLAGS "-lpthread -m32 ${CMAKE_SHARED_LINKER_FLAGS}" )
	set( CMAKE_EXE_LINKER_FLAGS " -m32 ${CMAKE_EXE_LINKER_FLAGS}" )
    endif( )
else( )
    message( AUTHOR_WARNING "Compiler not recognized.  Using default flags." )
endif( )

# If UNICODE is defined, pass extra definitions into 
if( BUILD_UNICODE )
    add_definitions( "/DUNICODE /D_UNICODE" )
endif( )

# Bolt statically links in the boost libraries, disable the example apps from trying to link in boost themselves
# add_definitions( "/DBOOST_ALL_NO_LIB" )

# Print out compiler flags for viewing/debug
message( STATUS "CMAKE_CXX_COMPILER flags: " ${CMAKE_CXX_FLAGS} )
message( STATUS "CMAKE_CXX_COMPILER debug flags: " ${CMAKE_CXX_FLAGS_DEBUG} )
message( STATUS "CMAKE_CXX_COMPILER release flags: " ${CMAKE_CXX_FLAGS_RELEASE} )
message( STATUS "CMAKE_CXX_COMPILER relwithdebinfo flags: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
message( STATUS "CMAKE_EXE_LINKER link flags: " ${CMAKE_EXE_LINKER_FLAGS} )

# Depending on whether we are building for 64 or 32 bit, construct common paths and names that subdirectories can reference for their use
if( Bolt_BUILD64 )
    set( CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${Bolt_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-x64" )
    set( INCLUDE_DIR include )
    set( BIN_DIR bin64 )
    set( LIB_DIR lib64 )
else( )
    set( CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${Bolt_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-x32" )
    set( INCLUDE_DIR include )
    set( BIN_DIR bin )
    set( LIB_DIR lib )
endif( )

# Build AMP library (only header files for now)
if( BUILD_ampBolt )
    if( MSVC_VERSION VERSION_LESS 1700 )
        message( FATAL_ERROR "Visual Studio 11 or higher is required because Bolt uses AMP C++ features" )
    endif( )
endif( )
    

option( BUILD_TBB "Create a solution that will work with TBB" OFF )

if( BUILD_TBB )
    # This will define TBB_FOUND
    message ( STATUS "Setting up TBB paths")
    find_package( TBB REQUIRED )
    if ( TBB_FOUND )
        message( STATUS "TBB is found installed in the path:" ${TBB_ROOT} )
        #list( APPEND Bolt.Dependencies TBB )
        include_directories( ${TBB_INCLUDE_DIRS} )
        add_definitions( "-DENABLE_TBB=true" )
        set(TBB_LIBRARIES debug;${TBB_LIBRARY_DEBUG};${TBB_LIBRARY_MALLOC_DEBUG};optimized;${TBB_LIBRARY};${TBB_LIBRARY_MALLOC})
    else ( ) 
        message( FATAL_ERROR "TBB not found. Install TBB and set TBB_ROOT env variable" )
        #message( STATUS "Setting up TBB external..." )
        #include( ExternalTBB )
        #message( STATUS "TBB_ROOT configured as: " ${TBB_ROOT} )
        #list( APPEND Bolt.Dependencies TBB )
        #list( APPEND Bolt.Cmake.Args -DBUILD_TBB=TRUE )
    endif( )
endif( )

# This will define OPENCL_FOUND
find_package( OpenCL REQUIRED )

# Use our packaged version of Boost if not specified by the user

if( NOT BOOST_ROOT )
    set( BOOST_ROOT "$ENV{BOLT_ROOT}" )
endif( )

set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_LIBS   ON )
set( Boost_DETAILED_FAILURE_MSG   ON )
# set( Boost_DEBUG ON )

# The root tree of BOOST was specified on the command line; use it to to find the specific Boost the user points too
# This will define Boost_FOUND
find_package( Boost 1.55 COMPONENTS thread system date_time chrono REQUIRED )

if( NOT BOLT_ROOT )
    set( BOLT_ROOT "$ENV{BOLT_ROOT}" )
endif( )

# This will define BOLT_FOUND
find_package( Bolt REQUIRED )

add_subdirectory( Benchmark )

