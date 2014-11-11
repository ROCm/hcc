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
message( STATUS "Setting up CL specific benchmarking projects" )

# Include standard OpenCL headers
include_directories( ${OPENCL_INCLUDE_DIRS} )

if(BUILD_TBB)
    include_directories( ${TBB_INCLUDE_DIRS} )
    add_definitions( "-DENABLE_TBB=true" )
    #add_library( ${TBB_LIBRARY} STATIC IMPORTED )
    set(TBB_LIBRARIES debug;${TBB_LIBRARY_DEBUG};${TBB_LIBRARY_MALLOC_DEBUG};optimized;${TBB_LIBRARY};${TBB_LIBRARY_MALLOC})
endif()

if( BUILD_clBolt )

if( MSVC_IDE )
    # Include standard OpenCL headers
    #add_subdirectory( Benchmark )
    # add_subdirectory( CopyBench )
    # add_subdirectory( CopyBuffer )
    # add_subdirectory( Fill ) 
    # add_subdirectory( Generate )
    # add_subdirectory( InnerProduct )
    # add_subdirectory( Reduce )
    # add_subdirectory( Scan )
    # add_subdirectory( ScanByKeyBench )
    # add_subdirectory( Sort )
    # add_subdirectory( StableSort )
    # add_subdirectory( StableSortByKey )
    # add_subdirectory( Transform )
    # add_subdirectory( TransformScanBench )
    # add_subdirectory( Gather )
    # add_subdirectory( Scatter )
else()
    # Include standard OpenCL headers
    #add_subdirectory( Benchmark )
endif()

endif( )
