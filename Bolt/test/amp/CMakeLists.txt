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
message( STATUS "Setting up AMP specific test projects..." )

set( Boost_USE_MULTITHREADED ON )
if ( BUILD_SHARED_LIBS ) 
    set( Boost_USE_STATIC_LIBS   OFF )
else()
    set( Boost_USE_STATIC_LIBS   ON )
endif()
set( Boost_DETAILED_FAILURE_MSG   ON )

# Set the test folder directory. This is common to cl and amp. 
set( BOLT_AMP_TEST_DIR ${BOLT_TEST_DIR}/amp )
set( BOLT_CL_TEST_DIR ${BOLT_TEST_DIR}/cl )

# set( Boost_DEBUG ON )

if( BOOST_ROOT )
    # The root tree of BOOST was specified on the command line; use it to to find the specific Boost the user points too
    find_package( Boost ${Boost.VERSION} COMPONENTS thread date_time chrono program_options REQUIRED )
    # This will define Boost_FOUND
else( )
    message( "Configure Bolt in <BOLT_ROOT>/bin to build the SuperBuild which will download and build Boost automatically" )    
    message( SEND_ERROR "The AMP Bolt library requires BOOST_ROOT to be defined" )
endif( )

# Include standard Boost headers
include_directories( ${Boost_INCLUDE_DIRS} )

# Include standard OpenCL headers
include_directories(
    ${BOLT_AMP_TEST_DIR}
    ${BOLT_AMP_TEST_DIR}/common
    ${OPENCL_INCLUDE_DIRS}) 
# include_directories( ${BOLT_CL_TEST_DIR} )
# Build tests as clamp applications
if ( UNIX )    
    link_directories( ${CLAMP_LIB_DIRS} )
    include_directories( ${CLAMP_INC_DIRS} )
    # Invoke directly what are in clamp's config: cxxflags/ldflags
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++amp -stdlib=libc++ " )
endif()
add_definitions( -DAMP_TESTS )

#add_subdirectory( BinarySearchTest )
add_subdirectory( PairTest )
#add_subdirectory( GenerateTest )
add_subdirectory( FillTest )
#add_subdirectory( InnerProductTest )
add_subdirectory( CopyTest )
# Deadlock
#add_subdirectory( MinElementTest )
add_subdirectory( MergeTest )
# Deadlock
#add_subdirectory( MaxElementTest )
#add_subdirectory( CountTest )
#add_subdirectory( PermutationIteratorTest )
#add_subdirectory( ReduceTest )
#add_subdirectory( ReduceByKeyTest )
#add_subdirectory( ScanTest )
# Deadlock
#add_subdirectory( ScanByKeyTest )
#add_subdirectory( SortTest )
#add_subdirectory( SortByKeyTest )
#add_subdirectory( StableSortTest )
#add_subdirectory( StableSortByKeyTest )
add_subdirectory( TransformTest )   
#add_subdirectory( TransformReduceTest )
# Deadlock
#add_subdirectory( TransformScanTest )
add_subdirectory( ScatterTest )
add_subdirectory( GatherTest )
add_subdirectory( device_vector )


