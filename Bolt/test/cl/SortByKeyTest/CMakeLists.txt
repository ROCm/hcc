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

set( clBolt.Test.SortByKey.Source SortByKeyTest.cpp 
                             ${BOLT_CL_TEST_DIR}/common/myocl.cpp)
set( clBolt.Test.SortByKey.Headers   ${BOLT_CL_TEST_DIR}/common/myocl.h
                                ${BOLT_INCLUDE_DIR}/bolt/cl/sort_by_key.h
                                ${BOLT_INCLUDE_DIR}/bolt/cl/detail/sort_by_key.inl )

set( clBolt.Test.SortByKey.Files ${clBolt.Test.SortByKey.Source} ${clBolt.Test.SortByKey.Headers} )

# Include standard OpenCL headers
include_directories( ${OPENCL_INCLUDE_DIRS} )

# Set project specific compile and link options
if( MSVC )
set( CMAKE_CXX_FLAGS "-bigobj ${CMAKE_CXX_FLAGS}" )
                set( CMAKE_C_FLAGS "-bigobj ${CMAKE_C_FLAGS}" )
endif()

add_executable( clBolt.Test.SortByKey ${clBolt.Test.SortByKey.Files} )

if(BUILD_TBB)
    target_link_libraries( clBolt.Test.SortByKey ${OPENCL_LIBRARIES} ${GTEST_LIBRARIES} ${Boost_LIBRARIES} clBolt.Runtime  ${TBB_LIBRARIES} )
else (BUILD_TBB)
    target_link_libraries( clBolt.Test.SortByKey ${OPENCL_LIBRARIES} ${GTEST_LIBRARIES} ${Boost_LIBRARIES} clBolt.Runtime )
endif()


set_target_properties( clBolt.Test.SortByKey PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( clBolt.Test.SortByKey PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET clBolt.Test.SortByKey PROPERTY FOLDER "Test/OpenCL")
        
# CPack configuration; include the executable into the package
install( TARGETS clBolt.Test.SortByKey
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}/import
    )
