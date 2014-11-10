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
set( clBolt.Test.Device.Vector.Source ${BOLT_CL_TEST_DIR}/common/stdafx.cpp 
                                  ${BOLT_CL_TEST_DIR}/common/myocl.cpp
                                  device.vector.test.cpp )
                                  
set( clBolt.Test.Device.Vector.Headers ${BOLT_CL_TEST_DIR}/common/stdafx.h 
                                   ${BOLT_CL_TEST_DIR}/common/targetver.h 
                                   ${BOLT_CL_TEST_DIR}/common/myocl.h 
                                   ${BOLT_INCLUDE_DIR}/bolt/cl/device_vector.h )

set( clBolt.Test.Device.Vector.Files ${clBolt.Test.Device.Vector.Source} ${clBolt.Test.Device.Vector.Headers} )

# Set project specific compile and link options
if( MSVC )
set( CMAKE_CXX_FLAGS "-DAMP_TESTS=0 ${CMAKE_CXX_FLAGS}" )
                set( CMAKE_C_FLAGS "-DAMP_TESTS=0 ${CMAKE_C_FLAGS}" )
endif()

add_executable( clBolt.Test.Device.Vector ${clBolt.Test.Device.Vector.Files} )

if(BUILD_TBB)
    target_link_libraries( clBolt.Test.Device.Vector clBolt.Runtime ${OPENCL_LIBRARIES} ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  ${TBB_LIBRARIES} )
else (BUILD_TBB)
    target_link_libraries( clBolt.Test.Device.Vector clBolt.Runtime ${OPENCL_LIBRARIES} ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  )
endif()

set_target_properties( clBolt.Test.Device.Vector PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( clBolt.Test.Device.Vector PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET clBolt.Test.Device.Vector PROPERTY FOLDER "Test/OpenCL")

# CPack configuration; include the executable into the package
install( TARGETS clBolt.Test.Device.Vector
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}/import
    )
