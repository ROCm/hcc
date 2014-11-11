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

# List the names of common files to compile across all platforms
message( STATUS "Setting up testing projects..." )

if( GTEST_ROOT )
    # The root tree of BOOST was specified on the command line; use it to to find the specific gTest the user wants
    find_package( GTest REQUIRED )
    # This will define GTEST_FOUND
else( )
    message( "Configure Bolt in <BOLT_ROOT>/bin to build the SuperBuild which will download and build GoogleTest automatically" )   
    message( SEND_ERROR "The testing projects for Bolt requires GTEST_ROOT to be defined" ) 
endif( )

# Include Bolt headers for all testing projects
include_directories( ${BOLT_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/include ${GTEST_INCLUDE_DIRS})

# Googletest requires this to be defined to 10 in order to compile on windows, because windows does not support variable length templates yet
add_definitions( "-D_VARIADIC_MAX=10" )
# treat warning as errors
if(MSVC_IDE)
add_definitions( "/WX" )
endif()


if(BUILD_TBB)
    include_directories( ${TBB_INCLUDE_DIRS} )
    add_definitions( "-DENABLE_TBB" )
    #add_library( ${TBB_LIBRARY} STATIC IMPORTED )
    set(TBB_LIBRARIES debug;${TBB_LIBRARY_DEBUG};${TBB_LIBRARY_MALLOC_DEBUG};optimized;${TBB_LIBRARY};${TBB_LIBRARY_MALLOC})
endif()

if( BUILD_ampBolt AND GTEST_FOUND )
    add_subdirectory( amp )
endif( )

if( BUILD_clBolt AND GTEST_FOUND )
    add_subdirectory( cl )
endif( )
