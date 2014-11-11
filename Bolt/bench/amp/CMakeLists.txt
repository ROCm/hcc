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
message( STATUS "Setting up AMP specific benchmarking projects" )

include_directories( ${BOLT_INCLUDE_DIR} )

if( BUILD_ampBolt )
    # Include standard OpenCL headers
    #add_subdirectory( Scan )
    #add_subdirectory( Reduce )
    #add_subdirectory( Transform )
    #add_subdirectory( TransformReduce )
endif( )
