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

# failed some tests on SPIR(has dependency on bolt::amp::sort availability), failed some tests on HSA
add_subdirectory( BinarySearchTest )

# passed
add_subdirectory( PairTest )

# compile OK, failed some tests on HSA
add_subdirectory( GenerateTest )

# passed
add_subdirectory( FillTest )

# passed on SPIR path. 'double type' related cases fail if no support in GPU
add_subdirectory( InnerProductTest )

# passed
add_subdirectory( CopyTest )

# compile OK, failed some tests on SPIR and HSA
add_subdirectory( MinElementTest )

# passed
add_subdirectory( MergeTest )

# compile OK, failed some tests on SPIR and HSA
add_subdirectory( MaxElementTest )

# passed
add_subdirectory( CountTest )

# compile OK, crashed on HSA after passing some tests
add_subdirectory( PermutationIteratorTest )

# passed on SPIR path. failed some tests on SPIR and HSA
add_subdirectory( ReduceTest )

# compile OK, crashed on HSA after passing some tests
add_subdirectory( ReduceByKeyTest )

# compile OK, failed some tests on SPIR and HSA
add_subdirectory( ScanTest )

# compile OK, failed some tests on SPIR and HSA
add_subdirectory( ScanByKeyTest )

# compile OK, crashed on HSA after passing some tests
add_subdirectory( SortTest )

# compile OK, failed some tests on HSA
add_subdirectory( SortByKeyTest )

# compile OK, crashed on HSA after passing some tests
add_subdirectory( StableSortTest )

# compile OK, failed some tests on HSA
add_subdirectory( StableSortByKeyTest )

# passed on SPIR path. crashed on HSA after passing some tests
add_subdirectory( TransformTest )   

# passed on SPIR path. failed some tests on HSA
add_subdirectory( TransformReduceTest )

# compile OK, failed some tests on SPIR and HSA
add_subdirectory( TransformScanTest )

# compile OK, crashed on HSA after passing some tests
add_subdirectory( ScatterTest )

# passed
add_subdirectory( GatherTest )

# compile OK, crashed on HSA after passing some tests
add_subdirectory( device_vector )

