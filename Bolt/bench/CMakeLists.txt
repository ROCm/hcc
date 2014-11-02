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
message( STATUS "Setting up benchmarking projects..." )

set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_LIBS   ON )
set( Boost_DETAILED_FAILURE_MSG   ON )
# set( Boost_DEBUG ON )

if( BOOST_ROOT )
    # The root tree of BOOST was specified on the command line; use it to to find the specific Boost the user points too
    find_package( Boost ${Boost.VERSION} COMPONENTS program_options date_time chrono REQUIRED )
    # This will define Boost_FOUND
    message( STATUS "Boost_PROGRAM_OPTIONS_LIBRARY: ${Boost_PROGRAM_OPTIONS_LIBRARY}" )
else( )
    message( "Configure Bolt in <BOLT_ROOT>/bin to build the SuperBuild which will download and build Boost automatically" )    
    message( SEND_ERROR "The Benchmarking projects for Bolt requires BOOST_ROOT to be defined" )
endif( )

include_directories( ${BOLT_INCLUDE_DIR} ${PROJECT_BINARY_DIR}/include ${Boost_INCLUDE_DIRS} )

if( BUILD_TBB )
    include_directories( ${TBB_INCLUDE_DIRS} )
    add_definitions( "-DENABLE_TBB" )
    set( TBB_LIBRARIES debug;${TBB_LIBRARY_DEBUG};${TBB_LIBRARY_MALLOC_DEBUG};optimized;${TBB_LIBRARY};${TBB_LIBRARY_MALLOC} )
endif()

#if( BUILD_ampBolt AND Boost_FOUND )
#    add_subdirectory( AMP )
#endif( )

#if( BUILD_clBolt AND Boost_FOUND )
#    include_directories( ${OPENCL_INCLUDE_DIRS} )
#    add_subdirectory( cl )
#endif( )

if( (BUILD_clBolt OR BUILD_ampBolt) AND Boost_FOUND )
    include_directories( ${OPENCL_INCLUDE_DIRS} )
    add_subdirectory( Benchmark )
endif( )


install( FILES  
            errorHandler.py
            measurePerformance.py
            performanceTesting.py
            plotPerformance.py
        DESTINATION
            ${BIN_DIR} )

install( FILES  
            errorHandler.py
            measurePerformance.py
            performanceTesting.py
            plotPerformance.py
        DESTINATION
            ${PROJECT_BINARY_DIR}/staging/\${CMAKE_INSTALL_CONFIG_NAME} )
