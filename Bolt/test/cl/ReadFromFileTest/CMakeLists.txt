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
set( clBolt.Test.ReadFromFile.Source ${BOLT_CL_TEST_DIR}/common/stdafx.cpp 
                                  ${BOLT_CL_TEST_DIR}/common/myocl.cpp
                                  ReadFromFile.cpp )
set( clBolt.Test.ReadFromFile.Headers ${BOLT_CL_TEST_DIR}/common/stdafx.h 
                                   ${BOLT_CL_TEST_DIR}/common/targetver.h 
                                   ${BOLT_CL_TEST_DIR}/common/myocl.h 
                                   ${BOLT_INCLUDE_DIR}/bolt/cl/transform.h 
                                   ${BOLT_INCLUDE_DIR}/bolt/cl/detail/transform.inl
                                   ${BOLT_CL_TEST_DIR}/common/utils.h 
                                   saxpy_functor.h )

set( clBolt.Test.ReadFromFile.Files ${clBolt.Test.ReadFromFile.Source} ${clBolt.Test.ReadFromFile.Headers} )

# Include standard OpenCL headers
include_directories( ${OPENCL_INCLUDE_DIRS} )

# Set project specific compile and link options
if( MSVC )
    set( CMAKE_CXX_FLAGS "-bigobj ${CMAKE_CXX_FLAGS}" )
    set( CMAKE_C_FLAGS "-bigobj ${CMAKE_C_FLAGS}" )
endif()

add_executable( clBolt.Test.ReadFromFile ${clBolt.Test.ReadFromFile.Files} )
target_link_libraries( clBolt.Test.ReadFromFile clBolt.Runtime ${OPENCL_LIBRARIES} ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${TBB_LIBRARIES} )

set_target_properties( clBolt.Test.ReadFromFile PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( clBolt.Test.ReadFromFile PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET clBolt.Test.ReadFromFile PROPERTY FOLDER "Test/OpenCL")
        
# CPack configuration; include the executable into the package
install( TARGETS clBolt.Test.ReadFromFile
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}/import
    )

install( FILES       
         saxpy_functor.h   DESTINATION ${BIN_DIR} 
         )

install( FILES       
         saxpy_functor.h DESTINATION ${PROJECT_BINARY_DIR}/staging/\${CMAKE_INSTALL_CONFIG_NAME} 
         )


