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
set( Bolt.Bench.Benchmark.Source Benchmark.cpp )
set( Bolt.Bench.Benchmark.Headers data_type.h )

set( Bolt.Bench.Benchmark.Files
  ${Bolt.Bench.Benchmark.Source}
  ${Bolt.Bench.Benchmark.Headers}
  )
# Include headers files
include_directories( ${BOLT_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIRS} )
add_executable( Bolt.Bench.Benchmark ${Bolt.Bench.Benchmark.Files} )

# Set project specific compile and link options
if( MSVC )
set( CMAKE_CXX_FLAGS "-bigobj ${CMAKE_CXX_FLAGS}" )
                set( CMAKE_C_FLAGS "-bigobj ${CMAKE_C_FLAGS}" )
endif()

if(BUILD_TBB)
    target_link_libraries( Bolt.Bench.Benchmark ${BOLT_LIBRARIES} ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} ${TBB_LIBRARIES} )
else (BUILD_TBB)
    target_link_libraries( Bolt.Bench.Benchmark ${BOLT_LIBRARIES} ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} )
endif()

set_target_properties( Bolt.Bench.Benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_property( TARGET Bolt.Bench.Benchmark PROPERTY FOLDER "Benchmark/AMP-OpenCL")

# CPack configuration; include the executable into the package
install( TARGETS Bolt.Bench.Benchmark
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}
    )
