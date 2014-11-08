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
set( clBolt.Bench.StableSort.Source 
        StableSortBench.cpp )

set( clBolt.Bench.StableSort.Headers stdafx.h 
        ${BOLT_INCLUDE_DIR}/bolt/cl/stablesort.h 
        ${BOLT_INCLUDE_DIR}/bolt/cl/detail/stablesort.inl)

set( clBolt.Bench.StableSort.Files 
        ${clBolt.Bench.StableSort.Source} 
        ${clBolt.Bench.StableSort.Headers} )

# Include standard OpenCL headers
include_directories( ${OPENCL_INCLUDE_DIRS} )

if( BUILD_TBB )
    add_definitions( "-DENABLE_TBB" )
endif( )

add_executable( clBolt.Bench.StableSort ${clBolt.Bench.StableSort.Files} )

target_link_libraries( clBolt.Bench.StableSort ${Boost_LIBRARIES} ${TBB_LIBRARIES} clBolt.Runtime )

set_target_properties( clBolt.Bench.StableSort PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( clBolt.Bench.StableSort PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET clBolt.Bench.StableSort PROPERTY FOLDER "Benchmark/OpenCL")

# CPack configuration; include the executable into the package
install( TARGETS clBolt.Bench.StableSort
	RUNTIME DESTINATION ${BIN_DIR}
	LIBRARY DESTINATION ${LIB_DIR}
	ARCHIVE DESTINATION ${LIB_DIR}
	)

