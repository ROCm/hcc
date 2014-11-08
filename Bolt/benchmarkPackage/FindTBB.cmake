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

# Locate a TBB implementation.
#
# Defines the following variables:
#
#   TBB_FOUND - Found the OPENCL framework
#   TBB_INCLUDE_DIRS - Include directories
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
#   TBB_LIBRARY 
#   TBB_MALLOC_LIBRARY 
#   TBB_LIBRARY_DEBUG 
#   TBB_MALLOC_LIBRARY_DEBUG 
#   TBB_INCLUDE_DIRS
# Accepts the following variables as input:
#
#   TBB_ROOT - (as a CMake or environment variable)
#                The root directory of the TBB implementation found
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether findTBB should search for 
#                              64bit or 32bit libs
#-----------------------
# Example Usage:
#
#    find_package(TBB REQUIRED)
#    include_directories(${TBB_INCLUDE_DIRS})
#
#    add_executable(foo foo.cc)
#    target_link_libraries(foo ${TBB_LIBRARIES})
#
#-----------------------

set(_TBB_LIB_NAME "tbb")
set(_TBB_LIB_MALLOC_NAME "${_TBB_LIB_NAME}malloc")
set(_TBB_LIB_DEBUG_NAME "${_TBB_LIB_NAME}_debug")
set(_TBB_LIB_MALLOC_DEBUG_NAME "${_TBB_LIB_MALLOC_NAME}_debug")


if( MSVC )
    if( MSVC_VERSION VERSION_LESS 1700 )
        set(TBB_COMPILER "vc10")
    else( )
        set(TBB_COMPILER "vc11")
    endif( )
endif( )

if ( NOT TBB_ROOT )
    set(TBB_ROOT $ENV{TBB_ROOT})
endif( )
message ("TBB_ROOT:" ${TBB_ROOT} )
if ( NOT TBB_ROOT )
    message( "TBB install not found in the system.")
else ( ) 
    # Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
    get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )

    if( LIB64 )
        set(TBB_ARCH_PLATFORM intel64)
    else( )
        set(TBB_ARCH_PLATFORM ia32)
    endif( )    
    

    #Find TBB header files
    find_path( TBB_INCLUDE_DIRS 
        NAMES tbb/tbb.h
        HINTS ${TBB_ROOT}/include
        DOC "TBB header file path"
    )
    mark_as_advanced( TBB_INCLUDE_DIRS )
    message ("TBB_INCLUDE_DIRS: " ${TBB_INCLUDE_DIRS} )
    
    #Find TBB Libraries
    set (_TBB_LIBRARY_DIR ${TBB_ROOT}/lib/${TBB_ARCH_PLATFORM} )
    find_library(TBB_LIBRARY ${_TBB_LIB_NAME} HINTS ${_TBB_LIBRARY_DIR}/${TBB_COMPILER}
                PATHS ENV LIBRARY_PATH ENV LD_LIBRARY_PATH)
    find_library(TBB_MALLOC_LIBRARY ${_TBB_LIB_MALLOC_NAME} HINTS ${_TBB_LIBRARY_DIR}/${TBB_COMPILER}
                PATHS ENV LIBRARY_PATH ENV LD_LIBRARY_PATH)
    find_library(TBB_LIBRARY_DEBUG ${_TBB_LIB_DEBUG_NAME} HINTS ${_TBB_LIBRARY_DIR}/${TBB_COMPILER}
                PATHS ENV LIBRARY_PATH ENV LD_LIBRARY_PATH)
    find_library(TBB_MALLOC_LIBRARY_DEBUG ${_TBB_LIB_MALLOC_DEBUG_NAME} HINTS ${_TBB_LIBRARY_DIR}/${TBB_COMPILER}
                PATHS ENV LIBRARY_PATH ENV LD_LIBRARY_PATH)        

    message ("TBB_LIBRARY:" ${TBB_LIBRARY})
    message ("TBB_MALLOC_LIBRARY:" ${TBB_MALLOC_LIBRARY})
    message ("TBB_LIBRARY_DEBUG:" ${TBB_LIBRARY_DEBUG})
    message ("TBB_MALLOC_LIBRARY_DEBUG:" ${TBB_MALLOC_LIBRARY_DEBUG})

    mark_as_advanced( TBB_LIBRARY )
    mark_as_advanced( TBB_MALLOC_LIBRARY )
    mark_as_advanced( TBB_LIBRARY_DEBUG )
    mark_as_advanced( TBB_MALLOC_LIBRARY_DEBUG )

    mark_as_advanced( TBB_ROOT )
    message ( "TBB_ROOT: "${TBB_ROOT} )

    include( FindPackageHandleStandardArgs )
    FIND_PACKAGE_HANDLE_STANDARD_ARGS( TBB DEFAULT_MSG TBB_LIBRARY TBB_MALLOC_LIBRARY TBB_LIBRARY_DEBUG TBB_MALLOC_LIBRARY_DEBUG TBB_INCLUDE_DIRS TBB_ROOT)

    if( NOT TBB_FOUND )
        message( STATUS "FindTBB looked for libraries named tbb but could not find" )
    endif()
endif ( ) 