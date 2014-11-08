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

message( STATUS "Configuring gTest SuperBuild..." )
include( ExternalProject )

set( ext.gTest_Version "1.6.0" CACHE STRING "gTest version to download/use" )
mark_as_advanced( ext.gTest_Version )

message( STATUS "ext.gTest_Version: " ${ext.gTest_Version} )

# Purely for debugging the file downloading URLs
# file( DOWNLOAD "http://code.google.com/p/googletest/downloads/detail?name=gtest-1.6.0.zip" 
		# "${CMAKE_CURRENT_BINARY_DIR}/download/boost-${ext.gTest_Version}/boost_1_49_0.7z" SHOW_PROGRESS STATUS fileStatus LOG fileLog )
# message( STATUS "status: " ${fileStatus} )
# message( STATUS "log: " ${fileLog} )
if( Bolt_BUILD64 )
    set( LIB_DIR lib64 )
else( )
    set( LIB_DIR lib )
endif( )

# We specify Nmake for windows, so that we can pass CMAKE_BUILD_TYPE to the generator.  Visual Studio generators ignore CMAKE_BUILD_TYPE
if( WIN32 )
	set( gTest.Generator "NMake Makefiles" )
else( )
	set( gTest.Generator "Unix Makefiles" )
	if(Bolt_BUILD64)
		set( BUILD_BITNESS "-m64")
	else()
		set( BUILD_BITNESS "-m32")
	endif()

endif( )

set( gTest.Cmake.Args 	-Dgtest_force_shared_crt=ON
						-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${LIB_DIR}	# present to support minsizerel and relwithdebinfo
						)

if( DEFINED ENV{GTEST_URL} )
    set( ext.gTest_URL "$ENV{GTEST_URL}" CACHE STRING "URL to download gTest from" )
else( )
    set( ext.gTest_URL "https://googletest.googlecode.com/files/gtest-${ext.gTest_Version}.zip" CACHE STRING "URL to download gTest from" )
endif( )
mark_as_advanced( ext.gTest_URL )

if ( UNIX AND BUILD_AMP )
   list (APPEND gTest.Cmake.Args -DCMAKE_C_COMPILER=${CLAMP_C_COMPILER} -DCMAKE_CXX_COMPILER=${CLAMP_CXX_COMPILER} )
   list (APPEND gTest.Cmake.Args -v -DGTEST_LINKED_AS_SHARED_LIBRARY=1 -DGTEST_HAS_TR1_TUPLE=0 -U__STRICT_ANSI__)
   set (BUILD_BITNESS "${BUILD_BITNESS} -DCMAKE_CXX_FLAGS=-stdlib=libc++ -I${CLAMP_LIBCXX_INC_DIR} ")
endif()

# FindGTest.cmake assumes that debug gtest libraries end with a 'd' postfix.  The official gtest cmakelist files do not add this postfix, 
# but luckily cmake allows us to specify a postfix through the CMAKE_DEBUG_POSTFIX variable.  However, the debug build of gtest is only 
# optional, and the release bits of gtest need to be present, even for a debug configuration.  So we always build both

# Command to download, build and install a debug build gTest
ExternalProject_Add(
    gTestDebug
	PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/gtest
    URL ${ext.gTest_URL}
	URL_MD5 4577b49f2973c90bf9ba69aa8166b786
	CMAKE_GENERATOR ${gTest.Generator}
	PATCH_COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/internal_utils.gTest.vs11.patch.cmake" <SOURCE_DIR>/cmake/internal_utils.cmake
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gtest-port.h" <SOURCE_DIR>/include/gtest/internal
	CMAKE_ARGS -DCMAKE_CXX_FLAGS=${BUILD_BITNESS} ${gTest.Cmake.Args} -DCMAKE_BUILD_TYPE=Debug -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=${LIB_DIR} -DCMAKE_DEBUG_POSTFIX=d
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/${LIB_DIR} <BINARY_DIR>/../staging/${LIB_DIR}
)

# Need to copy the header files to the staging directory too
ExternalProject_Add_Step( gTestDebug stage
   COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include <BINARY_DIR>/../staging/include
   DEPENDEES install
)

# Command to download, build and install a release build gTest
# Depends on gTestDebug so that it doesn't download at the same time, download should already exist from 
ExternalProject_Add(
    gTestRelease
	DEPENDS gTestDebug
	PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/gtest
    URL ${ext.gTest_URL}
	URL_MD5 4577b49f2973c90bf9ba69aa8166b786
	CMAKE_GENERATOR ${gTest.Generator}
	PATCH_COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/internal_utils.gTest.vs11.patch.cmake" <SOURCE_DIR>/cmake/internal_utils.cmake
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gtest-port.h" <SOURCE_DIR>/include/gtest/internal
	CMAKE_ARGS  -DCMAKE_CXX_FLAGS=${BUILD_BITNESS}  ${gTest.Cmake.Args} -DCMAKE_BUILD_TYPE=Release -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=${LIB_DIR}
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/${LIB_DIR} <BINARY_DIR>/../staging/${LIB_DIR}
)



set_property( TARGET gTestDebug PROPERTY FOLDER "Externals")
set_property( TARGET gTestRelease PROPERTY FOLDER "Externals")

ExternalProject_Get_Property( gTestRelease source_dir )
ExternalProject_Get_Property( gTestRelease binary_dir )

# This emulates the behavior of FindGtest.cmake, but you probaby don't need to use any of these variables in superbuilds
set( GTEST_INCLUDE_DIRS ${source_dir}/../staging/include )
set( GTEST_LIBRARIES debug;${binary_dir}/../staging/${LIB_DIR}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES};optimized;${binary_dir}/../staging/${LIB_DIR}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES} )
set( GTEST_MAIN_LIBRARIES debug;${binary_dir}/../staging/${LIB_DIR}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES};optimized;${binary_dir}/../staging/${LIB_DIR}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES} )
set( GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES};${GTEST_MAIN_LIBRARIES} )
set( GTEST_FOUND TRUE )

# For use by the user of ExternalGtest.cmake
set( GTEST_ROOT ${binary_dir}/../staging )
