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
message( STATUS "Setting up documentation project..." )

set( Doxygen.Dependencies )

if( DOXYGEN_ROOT )
    # Kind of sad that the findDoxygen module has not concept of DOXYGEN_ROOT
    find_program( DOXYGEN_EXECUTABLE
        NAMES doxygen
        HINTS
              ${DOXYGEN_ROOT}
        PATHS
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\doxygen_is1;Inno Setup: App Path]/bin"
        PATH_SUFFIXES
            bin
    )
else( )
    message( "Configure Bolt in <BOLT_ROOT>/bin to build the SuperBuild which will download Doxygen automatically" )	
    message( SEND_ERROR "The Documentation project for Bolt requires DOXYGEN_ROOT to be defined" )	
endif( )



if(UNIX)
configure_file( "${PROJECT_SOURCE_DIR}/doxy/bolt.doxy.lin.cfg.in" "${PROJECT_BINARY_DIR}/doxy/bolt.doxy.cfg" @ONLY )
else()
configure_file( "${PROJECT_SOURCE_DIR}/doxy/bolt.doxy.cfg.in" "${PROJECT_BINARY_DIR}/doxy/bolt.doxy.cfg" @ONLY )
endif()

add_custom_target( 	Bolt.Documentation
                    COMMAND ${DOXYGEN_EXECUTABLE} bolt.doxy.cfg
                    DEPENDS ${Doxygen.Dependencies}
                    COMMENT "Generating Bolt API documentation with Doxygen" VERBATIM
                )

# Copy over the generated html documentation into the install location
install( DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR}/doxy.out/html
        DESTINATION
            docs
        PATTERN "*.chm" EXCLUDE )

# Copy over the generated chm documentation into the install location
if(WIN32)
install( FILES 
            ${CMAKE_CURRENT_BINARY_DIR}/doxy.out/html/bolt.chm
        DESTINATION 
            docs/chm )
endif()
