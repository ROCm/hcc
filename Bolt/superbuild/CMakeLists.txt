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

# We require 2.8.10 because of the added support to download from https URL's
cmake_minimum_required( VERSION 2.8.10 )

# uncomment these to debug nmake and borland makefiles
#SET(CMAKE_START_TEMP_FILE "")
#SET(CMAKE_END_TEMP_FILE "")
#SET(CMAKE_VERBOSE_MAKEFILE 1)

# This becomes the name of the solution file
project( Bolt.SuperBuild )

# Define a version for the code
if( NOT DEFINED Bolt.SuperBuild_VERSION_MAJOR )
    set( Bolt.SuperBuild_VERSION_MAJOR 1 )
endif( )

if( NOT DEFINED Bolt.SuperBuild_VERSION_MINOR )
    set( Bolt.SuperBuild_VERSION_MINOR 2 )
endif( )

if( NOT DEFINED Bolt.SuperBuild_VERSION_PATCH )
    set( Bolt.SuperBuild_VERSION_PATCH 0 )
endif( )

set( Bolt.SuperBuild_VERSION "${Bolt.SuperBuild_VERSION_MAJOR}.${Bolt.SuperBuild_VERSION_MINOR}.${Bolt.SuperBuild_VERSION_PATCH}")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
# Need to check out CMAKE_RUNTIME_OUTPUT_DIRECTORY variable, and see if that eliminates the need to modify install path
if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
    set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
#if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
#  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE )
#endif()

if( MSVC_IDE )
    set( Bolt_BUILD64 ${CMAKE_CL_64} )
    set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
    option( Bolt_BUILD64 "Build a 64-bit product" ON )
    set( BOLT_BUILD_TYPE "RELEASE" CACHE STRING "Build a RELESAE/DEBUG  product" )

endif()

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
# Only Visual Studio may build Bolt with C++AMP as of now 
if( MSVC )
  set(Bolt_AMP_default ON)
else()
  set(Bolt_AMP_default ON)
endif() 


option( BUILD_Boost "Install and Compile Boost as an external dependency" ON )
option( BUILD_Doxygen "Install Doxygen as an external dependency" OFF )
option( BUILD_Bolt "Setup Bolt to use all the external dependencies" ON )
option( BUILD_gTest "Install googleTest as an external dependency" ON )
option( BUILD_TBB "Install TBB as an external dependency" OFF )
option( BUILD_AMP "Generate projects that use the AMP backend" ${Bolt_AMP_default})
option( BUILD_OpenCL "Generate projects that use the OpenCL backend" OFF )
option( BUILD_StripSymbols "When making debug builds, remove symbols and program database files" OFF )
option( BUILD_benchmarks "Build Bench Mark" OFF )

######## Perform silent build with clamp ##########
set (BUILD_Boost ON)
set (BUILD_StripSymbols OFF)
set (BUILD_AMP ON)
set (BUILD_OpenCL OFF)
set (BUILD_TBB OFF)
set (BUILD_Bolt ON)

# DO NOT BOTHER TO CONFIGURE THE FOLLOWING UNLESS YOU WANT
set (BUILD_Doxygen OFF)
set (BUILD_gTest ON)
set (BUILD_benchmarks OFF)
set (BUILD_tests ON)

# Comment this out because this does not work yet
# option( BUILD_Examples "Generate example projects to demonstrate Bolt usage" OFF )

set( Bolt.Dependencies )
set( Bolt.Cmake.Args )
set( BUILD_Examples OFF )

# If the user selects, download, uncompress, and setup Boost
if( BUILD_Boost )
    message( STATUS "Setting up Boost external..." )
    include( ExternalBoost )
    set( BOOST_ROOT ${binary_dir} )
    message( STATUS "BOOST_ROOT configured as: " ${BOOST_ROOT} )
    list( APPEND Bolt.Dependencies Boost )
    list( APPEND Bolt.Cmake.Args -D BUILD_benchmarks=${BUILD_benchmarks} -D Boost.VERSION=${Boost_VERSION} -D BOOST_ROOT=${BOOST_ROOT} )
endif( )

# If the user selects, download, uncompress, and setup Doxygen
if( BUILD_Doxygen )
    message( STATUS "Setting up Doxygen external..." )
    include( ExternalDoxygen )  
    set( DOXYGEN_ROOT ${binary_dir} )
    message( STATUS "DOXYGEN_ROOT configured as: " ${DOXYGEN_ROOT} )
    list( APPEND Bolt.Dependencies Doxygen )
    list( APPEND Bolt.Cmake.Args -D BUILD_documentation=TRUE -D DOXYGEN_ROOT=${DOXYGEN_ROOT} )
endif( )

# If the user selects, download, uncompress, and setup googleTest
if( BUILD_gTest )
    message( STATUS "Setting up googleTest external..." )
    include( ExternalGtest )
#   set( GTEST_ROOT ${binary_dir} )
    message( STATUS "GTEST_ROOT configured as: " ${GTEST_ROOT} )
    list( APPEND Bolt.Dependencies gTestDebug gTestRelease )
    list( APPEND Bolt.Cmake.Args -DBUILD_tests=TRUE -DGTEST_ROOT=${GTEST_ROOT} )
    if( MSVC )
        list( APPEND Bolt.Cmake.Args -DGTEST_MSVC_SEARCH=MT )
    endif( )
endif( )

if(UNIX)
    list( APPEND Bolt.Cmake.Args -DBolt_BUILD64=${Bolt_BUILD64} -DBOLT_BUILD_TYPE=${BOLT_BUILD_TYPE} )
endif()
# Build TBB library
if( BUILD_TBB )
    #find_package( TBB REQUIRED )
    #if ( TBB_FOUND )
    #    message( STATUS "TBB is found installed in the path:" ${TBB_ROOT} )
    #    list( APPEND Bolt.Dependencies TBB )
        list( APPEND Bolt.Cmake.Args -DBUILD_TBB=TRUE )
    #else ( ) 
    #    message( STATUS "Setting up TBB external..." )
    #    include( ExternalTBB )
    #    message( STATUS "TBB_ROOT configured as: " ${TBB_ROOT} )
    #    list( APPEND Bolt.Dependencies TBB )
    #    list( APPEND Bolt.Cmake.Args -DBUILD_TBB=TRUE )
    #endif( )
endif( )

if( BUILD_AMP )
    # Pass from clamp project
    if( UNIX )
        # Pass to Bolt project since it is ExternalProject_Add not add_subdirectory
        list (APPEND Bolt.Cmake.Args -DCMAKE_C_COMPILER=${CLAMP_C_COMPILER} -DCMAKE_CXX_COMPILER=${CLAMP_CXX_COMPILER} )
        list (APPEND Bolt.Cmake.Args -DCMAKE_COMPILER_IS_GNUCXX=0 )
	
	# Ugly codes to make cmake argument separated by space
	string( REPLACE ";" " " STRING_CLAMP_INC_DIRS "${CLAMP_INC_DIRS}" )
        list (APPEND Bolt.Cmake.Args -DSTRING_CLAMP_INC_DIRS=${STRING_CLAMP_INC_DIRS} )
	
	string( REPLACE ";" " " STRING_CLAMP_LIB_DIRS "${CLAMP_LIB_DIRS}" )
        list (APPEND Bolt.Cmake.Args -DSTRING_CLAMP_LIB_DIRS=${STRING_CLAMP_LIB_DIRS} )
	
	string( REPLACE ";" " " STRING_CLAMP_LIBRARIES "${CLAMP_LIBRARIES}" )
        list (APPEND Bolt.Cmake.Args -DSTRING_CLAMP_LIBRARIES=${STRING_CLAMP_LIBRARIES} )

    endif()
endif()

# If the user selects, download, uncompress, and setup Doxygen
if( BUILD_Bolt )
    message( STATUS "Setting up Bolt external..." )
    
    # Below is a fancy CMake command to download, build and install Boost on the users computer
    ExternalProject_Add(
        Bolt
        DEPENDS ${Bolt.Dependencies}
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/..
        BINARY_DIR Bolt-build
        CMAKE_ARGS ${Bolt.Cmake.Args} -DBUILD_tests=${BUILD_tests} -DBUILD_ampBolt=${BUILD_AMP} -DBUILD_clBolt=${BUILD_OpenCL} -DBUILD_examples=${BUILD_Examples} -DBUILD_StripSymbols=${BUILD_StripSymbols} -DBolt_VERSION_MAJOR=${Bolt.SuperBuild_VERSION_MAJOR} -DBolt_VERSION_MINOR=${Bolt.SuperBuild_VERSION_MINOR} -DBolt_VERSION_PATCH=${Bolt.SuperBuild_VERSION_PATCH} 
        INSTALL_COMMAND ""
    )
endif( )
