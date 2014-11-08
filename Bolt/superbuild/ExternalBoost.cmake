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

message( STATUS "Configuring Boost SuperBuild..." )
include( ExternalProject )

set( ext.Boost_VERSION "1.55.0" CACHE STRING "Boost version to download/use" )
mark_as_advanced( ext.Boost_VERSION )
string( REPLACE "." "_" ext.Boost_Version_Underscore ${ext.Boost_VERSION} )

message( STATUS "ext.Boost_VERSION: " ${ext.Boost_VERSION} )

# Purely for debugging the file downloading URLs
# file( DOWNLOAD "http://downloads.sourceforge.net/project/boost/boost/1.49.0/boost_1_49_0.7z" 
        # "${CMAKE_CURRENT_BINARY_DIR}/download/boost-${ext.Boost_VERSION}/boost_1_49_0.7z" SHOW_PROGRESS STATUS fileStatus LOG fileLog )
# message( STATUS "status: " ${fileStatus} )
# message( STATUS "log: " ${fileLog} )

# Initialize various command names based on platform
if ( UNIX )
        set(Boost.B2 "./b2")
  set(Boost.Bootstrap "./bootstrap.sh")
else( )
        set(Boost.B2 "b2")
  set(Boost.Bootstrap "bootstrap.bat")
endif( )

set( Boost.Command ${Boost.B2} -j 4 --with-program_options --with-thread --with-system --with-date_time --with-chrono )
if ( BUILD_SHARED_LIBS ) 
    list ( APPEND Boost.Command --cxxflags=-fPIC )
endif()

if ( UNIX AND BUILD_AMP ) 

  # Create user-config.jam in $HOME to specify clang as the compiler to build Boost. 
  # /b2 will search it first for configurations  
  execute_process( COMMAND echo " # ---------------------\n"
"# CLAMP compiler configuration.\n"
"# ---------------------\n"
"using clang : :  ${CLAMP_CXX_COMPILER} ;"
OUTPUT_FILE $ENV{HOME}/user-config.jam )

  list ( APPEND Boost.Command --debug-configuration toolset=clang )
  list ( APPEND Boost.Command linkflags="-L${CLAMP_LIBCXX_LIB_DIR}" linkflags="-L${CLAMP_LIBCXXRT_LIB_DIR}")
  list ( APPEND Boost.Command include="${CLAMP_LIBCXX_INC_DIR}" )
endif() 

if( Bolt_BUILD64 )
    list( APPEND Boost.Command address-model=64 )
else( )
    list( APPEND Boost.Command address-model=32 )
endif( )

if( MSVC )
    if( MSVC_VERSION VERSION_LESS 1700 )
        list( APPEND Boost.Command toolset=msvc-10.0 )
    elseif( MSVC_VERSION VERSION_LESS 1800 )
        list( APPEND Boost.Command toolset=msvc-11.0 )
    else()    
        list( APPEND Boost.Command toolset=msvc-12.0 )    
    endif( )
endif( )

if ( BUILD_SHARED_LIBS ) 
    list( APPEND Boost.Command link=shared stage )
else()
    list( APPEND Boost.Command link=static stage )
endif()

# If the user has cached the Boost download to a local location, they may prefer to download the package from there instead of from the internet, as the
# internet could be unpredictable or slow.
# If the user has a local copy stored somewhere, they can define the full path to the package in a BOOST_URL environment variable
if( DEFINED ENV{BOOST_URL} )
    set( ext.Boost_URL "$ENV{BOOST_URL}" CACHE STRING "URL to download Boost from" )
else( )
    set( ext.Boost_URL "http://sourceforge.net/projects/boost/files/boost/${ext.Boost_VERSION}/boost_${ext.Boost_Version_Underscore}.zip/download" CACHE STRING "URL to download Boost from" )
endif( )
mark_as_advanced( ext.Boost_URL )

# Below is a fancy CMake command to download, build and install Boost on the users computer

ExternalProject_Add(
    Boost
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/boost
    URL ${ext.Boost_URL}
#    URL_MD5 f310a8198318c10e5e4932a07c755a6a
    URL_MD5 8aca361a4713a1f491b0a5e33fee0f1f
    UPDATE_COMMAND ${Boost.Bootstrap}
#    PATCH_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${Boost.Command}
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)

set_property( TARGET Boost PROPERTY FOLDER "Externals")

ExternalProject_Get_Property( Boost source_dir )
ExternalProject_Get_Property( Boost binary_dir )
set( Boost_INCLUDE_DIRS ${source_dir} )

if( MSVC )
#    if( MSVC_VERSION VERSION_LESS 1700 )
#        set( Boost_LIBRARIES debug;${binary_dir}/stage/lib/libboost_program_options-vc110-mt-gd-1_50.lib;optimized;${binary_dir}/stage/lib/libboost_program_options-vc110-mt-1_50.lib )
#    else()
#        set( Boost_LIBRARIES debug;${binary_dir}/stage/lib/libboost_program_options-vc120-mt-gd-1_50.lib;optimized;${binary_dir}/stage/lib/libboost_program_options-vc120-mt-1_50.lib )
#    endif()    
else()
set( Boost_LIBRARIES debug;${binary_dir}/stage/lib/libboost_program_options.a;optimized;${binary_dir}/stage/lib/libboost_program_options.a )
endif()

set( Boost_FOUND TRUE )

# Can't get packages to download from github because the cmake file( download ... ) does not understand https protocal
# Gitorious is problematic because it apparently only offers .tar.gz files to download, which windows doesn't support by default
# Also, both repo's do not like to append a filetype to the URL, instead they use some forwarding script.  ExternalProject_Add wants a filetype.
# set( BOOST_BUILD_PROJECTS program_options CACHE STRING "* seperated Boost modules to be built")
# ExternalProject_Add(
   # Boost
   # URL http://gitorious.org/boost/cmake/archive-tarball/cmake-${ext.Boost_VERSION}.tar.gz
   # LIST_SEPARATOR *
   # CMAKE_ARGS     -DENABLE_STATIC=ON 
                # -DENABLE_SHARED=OFF 
                # -DENABLE_DEBUG=OFF 
                # -DENABLE_RELEASE=ON 
                # -DENABLE_SINGLE_THREADED=OFF 
                # -DENABLE_MULTI_THREADED=ON 
                # -DENABLE_STATIC_RUNTIME:BOOL=OFF 
                # -DENABLE_DYNAMIC_RUNTIME=ON 
                # -DWITH_PYTHON:BOOL=OFF 
                # -DBUILD_PROJECTS=${BOOST_BUILD_PROJECTS} ${CMAKE_ARGS}
# )
