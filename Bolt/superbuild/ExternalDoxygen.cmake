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

message( STATUS "Configuring Doxygen SuperBuild..." )
include( ExternalProject )

set( Doxygen_Version 1.8.3 )

message( STATUS "Doxygen_Version: " ${Doxygen_Version} )

# Purely for debugging the file downloading URLs
# file( DOWNLOAD "http://ftp.stack.nl/pub/users/dimitri/doxygen-${Doxygen_Version}.windows.bin.zip" 
		# "${CMAKE_CURRENT_BINARY_DIR}/download/Doxygen-${Doxygen_Version}/Doxygen_1_49_0.7z" SHOW_PROGRESS STATUS fileStatus LOG fileLog )
# message( STATUS "status: " ${fileStatus} )
# message( STATUS "log: " ${fileLog} )

if( DEFINED ENV{DOXYGEN_URL} )
    set( ext.Doxygen_URL "$ENV{DOXYGEN_URL}" CACHE STRING "URL to download Doxygen from" )
else( )
   if( UNIX)
	  set( ext.Doxygen_URL "http://ftp.stack.nl/pub/users/dimitri/doxygen-${Doxygen_Version}.linux.bin.tar.gz" CACHE STRING "URL to download Doxygen from" )	
	set( MD5_VAL 0459371bf621ffaaaafb6d0f35848317)
   else()	
	    set( ext.Doxygen_URL "http://ftp.stack.nl/pub/users/dimitri/doxygen-${Doxygen_Version}.windows.bin.zip" CACHE STRING "URL to download Doxygen from" )
	set( MD5_VAL 6cad0f8af783eb92b64aae3da8a7be35)	 	
   endif()
endif( )

mark_as_advanced( ext.Doxygen_URL )

# Below is a fancy CMake command to download, build and install Doxygen on the users computer
ExternalProject_Add(
    Doxygen
	PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external/Doxygen
    URL ${ext.Doxygen_URL}
	URL_MD5 ${MD5_VAL}
    UPDATE_COMMAND ""
#    PATCH_COMMAND ""
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
)

set_property( TARGET Doxygen PROPERTY FOLDER "Externals")

ExternalProject_Get_Property( Doxygen source_dir )
ExternalProject_Get_Property( Doxygen binary_dir )

if(UNIX)
set( DOXYGEN_EXECUTABLE ${binary_dir}/doxygen )
else()
set( DOXYGEN_EXECUTABLE ${binary_dir}/doxygen.exe )
endif()
# set( Doxygen_INCLUDE_DIRS ${source_dir} )
# set( Doxygen_LIBRARIES debug;${binary_dir}/stage/lib/libDoxygen_program_options-vc110-mt-gd-1_49.lib;optimized;${binary_dir}/stage/lib/libDoxygen_program_options-vc110-mt-1_49.lib )
