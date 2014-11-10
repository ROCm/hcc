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
set( clBolt.Test.Control.Source  ${BOLT_CL_TEST_DIR}/common/stdafx.cpp 
                                  ${BOLT_CL_TEST_DIR}/common/myocl.cpp 
                                   Control.test.cpp )
                                   
set( clBolt.Test.Control.Headers  ${BOLT_CL_TEST_DIR}/common/stdafx.h 
                                   ${BOLT_CL_TEST_DIR}/common/targetver.h 
                                   ${BOLT_CL_TEST_DIR}/common/myocl.h 
                                   ${BOLT_INCLUDE_DIR}/bolt/cl/control.h )

set( clBolt.Test.Control.Files ${clBolt.Test.Control.Source} ${clBolt.Test.Control.Headers} )

add_executable( clBolt.Test.Control ${clBolt.Test.Control.Files} )

if(BUILD_TBB)
    target_link_libraries( clBolt.Test.Control clBolt.Runtime ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  ${TBB_LIBRARIES} )
else (BUILD_TBB)
    target_link_libraries( clBolt.Test.Control clBolt.Runtime ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  )
endif()

set_target_properties( clBolt.Test.Control PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( clBolt.Test.Control PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET clBolt.Test.Control PROPERTY FOLDER "Test/OpenCL")

# CPack configuration; include the executable into the package
install( TARGETS clBolt.Test.Control
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}/import
    )
