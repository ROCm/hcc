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
set( clBolt.Bench.ScanByKey.Source BenchScanByKey.cpp )
set( clBolt.Bench.ScanByKey.Headers ${BOLT_INCLUDE_DIR}/bolt/cl/scan_by_key.h )

set( clBolt.Bench.ScanByKey.Files ${clBolt.Bench.ScanByKey.Source} ${clBolt.Bench.ScanByKey.Headers} )

add_executable( clBolt.Bench.ScanByKey ${clBolt.Bench.ScanByKey.Files} )

if (BUILD_TBB)
    target_link_libraries( clBolt.Bench.ScanByKey ${Boost_LIBRARIES} clBolt.Runtime ${TBB_LIBRARIES} )
else (BUILD_TBB)
    target_link_libraries( clBolt.Bench.ScanByKey ${Boost_LIBRARIES} clBolt.Runtime )
endif()

set_target_properties( clBolt.Bench.ScanByKey PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( clBolt.Bench.ScanByKey PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET clBolt.Bench.ScanByKey PROPERTY FOLDER "Benchmark/OpenCL")

# CPack configuration; include the executable into the package
install( TARGETS clBolt.Bench.ScanByKey
	RUNTIME DESTINATION ${BIN_DIR}
	LIBRARY DESTINATION ${LIB_DIR}
	ARCHIVE DESTINATION ${LIB_DIR}
	)
