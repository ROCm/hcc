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
set( clBolt.Bench.InnerProduct.Source stdafx.cpp InnerProduct.cpp )
set( clBolt.Bench.InnerProduct.Headers stdafx.h targetver.h ${BOLT_INCLUDE_DIR}/bolt/cl/inner_product.h )

set( clBolt.Bench.InnerProduct.Files ${clBolt.Bench.InnerProduct.Source} ${clBolt.Bench.InnerProduct.Headers} )

add_executable( clBolt.Bench.InnerProduct ${clBolt.Bench.InnerProduct.Files} )

if (BUILD_TBB)
    target_link_libraries( clBolt.Bench.InnerProduct ${Boost_LIBRARIES} clBolt.Runtime ${TBB_LIBRARIES} )
else (BUILD_TBB)
    target_link_libraries( clBolt.Bench.InnerProduct ${Boost_LIBRARIES} clBolt.Runtime )
endif()

set_target_properties( clBolt.Bench.InnerProduct PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( clBolt.Bench.InnerProduct PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET clBolt.Bench.InnerProduct PROPERTY FOLDER "Benchmark/OpenCL")

# CPack configuration; include the executable into the package
install( TARGETS clBolt.Bench.InnerProduct
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}
    )
