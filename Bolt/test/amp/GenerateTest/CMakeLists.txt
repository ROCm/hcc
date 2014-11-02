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
set( ampBolt.Test.Generate.Source GenerateTest.cpp )
set( ampBolt.Test.Generate.Headers ${BOLT_INCLUDE_DIR}/bolt/amp/generate.h ${BOLT_INCLUDE_DIR}/bolt/amp/detail/generate.inl )

set( ampBolt.Test.Generate.Files ${ampBolt.Test.Generate.Source} ${ampBolt.Test.Generate.Headers} )

add_executable( ampBolt.Test.Generate ${ampBolt.Test.Generate.Files} )


if( MSVC )
    set( CMAKE_CXX_FLAGS "-bigobj ${CMAKE_CXX_FLAGS}" )
    set( CMAKE_C_FLAGS "-bigobj ${CMAKE_C_FLAGS}" )
endif()


if(BUILD_TBB)
    target_link_libraries( ampBolt.Test.Generate ampBolt.Runtime ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  ${TBB_LIBRARIES} )
else (BUILD_TBB)
    target_link_libraries( ampBolt.Test.Generate ampBolt.Runtime ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  )
endif()

if ( UNIX )
  target_link_libraries( ampBolt.Test.Generate ${CLAMP_LIBRARIES} )
endif()


set_target_properties( ampBolt.Test.Generate PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( ampBolt.Test.Generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET ampBolt.Test.Generate PROPERTY FOLDER "Test/AMP")

# CPack configuration; include the executable into the package
install( TARGETS ampBolt.Test.Generate
	RUNTIME DESTINATION ${BIN_DIR}
	LIBRARY DESTINATION ${LIB_DIR}
	ARCHIVE DESTINATION ${LIB_DIR}/import
	)
