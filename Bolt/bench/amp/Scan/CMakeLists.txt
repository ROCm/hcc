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
set( ampBolt.Bench.Scan.Source stdafx.cpp Scan.cpp )
set( ampBolt.Bench.Scan.Headers stdafx.h targetver.h ${BOLT_INCLUDE_DIR}/bolt/amp/Scan.h )

set( ampBolt.Bench.Scan.Files ${ampBolt.Bench.Scan.Source} ${ampBolt.Bench.Scan.Headers} )

# Include standard OpenCL headers
include_directories( ${Boost_INCLUDE_DIRS} )

add_executable( ampBolt.Bench.Scan ${ampBolt.Bench.Scan.Files} )

target_link_libraries( ampBolt.Bench.Scan ${Boost_LIBRARIES} ampBolt.Runtime )

add_dependencies( ampBolt.Bench.Scan Boost )

set_target_properties( ampBolt.Bench.Scan PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( ampBolt.Bench.Scan PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET ampBolt.Bench.Scan PROPERTY FOLDER "Benchmark/AMP")

# CPack configuration; include the executable into the package
install( TARGETS ampBolt.Bench.Scan
	RUNTIME DESTINATION ${BIN_DIR}
	LIBRARY DESTINATION ${LIB_DIR}
	ARCHIVE DESTINATION ${LIB_DIR}/import
	)
