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

# Set project specific compile and link options
if( MSVC )
set( CMAKE_CXX_FLAGS "-bigobj ${CMAKE_CXX_FLAGS}" )
                set( CMAKE_C_FLAGS "-bigobj ${CMAKE_C_FLAGS}" )
endif()

set( clBolt.Test.Scan.Source
    ${BOLT_CL_TEST_DIR}/common/stdafx.cpp 
    ${BOLT_CL_TEST_DIR}/common/myocl.cpp 
    ScanTest.cpp )
set( clBolt.Test.Scan.Headers   ${BOLT_CL_TEST_DIR}/common/stdafx.h 
                                ${BOLT_CL_TEST_DIR}/common/myocl.cpp 
                                ${BOLT_CL_TEST_DIR}/common/targetver.h 
                                ${BOLT_INCLUDE_DIR}/bolt/cl/scan.h 
                                ${BOLT_INCLUDE_DIR}/bolt/cl/detail/scan.inl )

set( clBolt.Test.Scan.Files ${clBolt.Test.Scan.Source} ${clBolt.Test.Scan.Headers} )

add_executable( clBolt.Test.Scan ${clBolt.Test.Scan.Files} )

if(BUILD_TBB)
    target_link_libraries( clBolt.Test.Scan clBolt.Runtime ${OPENCL_LIBRARIES} ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  ${TBB_LIBRARIES} )
else (BUILD_TBB)
    target_link_libraries( clBolt.Test.Scan clBolt.Runtime ${OPENCL_LIBRARIES} ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  )
endif()

set_target_properties( clBolt.Test.Scan PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( clBolt.Test.Scan PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET clBolt.Test.Scan PROPERTY FOLDER "Test/OpenCL")

# CPack configuration; include the executable into the package
install( TARGETS clBolt.Test.Scan
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}/import
    )
