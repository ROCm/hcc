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
set( clBolt.StringifyKernels.Source StringifyKernels.cpp )
set( clBolt.StringifyKernels.Headers "" )

set( clBolt.StringifyKernels.Files ${clBolt.StringifyKernels.Source} ${clBolt.StringifyKernels.Headers} )

add_executable( clBolt.StringifyKernels ${clBolt.StringifyKernels.Files} )
target_link_libraries( clBolt.StringifyKernels ${Boost_LIBRARIES} )

set_target_properties( clBolt.StringifyKernels PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( clBolt.StringifyKernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET clBolt.StringifyKernels PROPERTY FOLDER "Tools")
		
# CPack configuration; include the executable into the package
install( TARGETS clBolt.StringifyKernels
	RUNTIME DESTINATION ${BIN_DIR}
	LIBRARY DESTINATION ${LIB_DIR}
	ARCHIVE DESTINATION ${LIB_DIR}/import
	)
