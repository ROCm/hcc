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

set( examples.Hessian.Source main.cpp hessian.cpp hessian_amp.cpp  hessian_amp_range.cpp hessian_ocl.cpp hessian_tbb.cpp image_utils.cpp hessian_oclcpp.cpp myocl.cpp hessian_cpuserial.cpp hessian_boltcl.cpp )
set( examples.Hessian.Headers bitmap_headers.h Hessian.h hessian_amp.h hessian_amp_range.h  hessian_ocl.h hessian_tbb.h image_utils.h matrix_utils.h hessian_oclcpp.h myocl.h hessian_boltcl.h hessian_cpuserial.h hessian_boltcl.h hessian.cl myocl_bufferpool.h ${BOLTAMP_DIR}/transform_reduce.h  ${BOLTAMP_DIR}/transform_reduce_range.h  ${BOLTAMP_DIR}/synchronized_view.h)

set( examples.Hessian.Files ${examples.Hessian.Source} ${examples.Hessian.Headers} )

set( HESSIAN_DIR "${PROJECT_SOURCE_DIR}/Hessian" )

# For a rainy day, add pre-compiled header support
#if( MSVC )
#	if (USE_MSVC_PCH)
	
#		set_source_files_properties(LungAnalysisPCH.cxx
#			PROPERTIES
#			COMPILE_FLAGS "/YcLungAnalysisPCH.h"
#			)
#		foreach( src_file ${UPMC_LA_SRCS} )
#			set_source_files_properties(
#				${src_file}
#				PROPERTIES
#				COMPILE_FLAGS "/YuLungAnalysisPCH.h"
#				)
#		endforeach( src_file ${UPMC_LA_SRCS} )
		
#		list(APPEND UPMC_LA_SRCS LungAnalysisPCH.cxx)
#		list(APPEND UPMC_LA_HDRS LungAnalysisPCH.h)

#	endif(USE_MSVC_PCH)
#endif (MSVC)
  
#	add_definitions( ${Boost_LIB_DIAGNOSTIC_DEFINITIONS} )
# add_definitions( "/DReduce_EXPORTS" )
add_definitions(-DCPPAMP)

# hacks to get TBB to work correctly.
# Note have to define TBBROOT, ie TBBROOT=F:\Program Files Remote\tbb40_278oss

#set ( TROOT FILE( $ENV{TBBROOT} ) )
STRING(REPLACE "\\" "/" TROOT $ENV{TBBROOT})
set ( TBB_INCLUDE_DIR $ENV{TBBROOT}/include )
#set ( TBB_LIBRARY_DIR $ENV{TBBROOT}/lib/ia32/vc10 )
set ( TBB_LIBRARY_DIR $ENV{TBBROOT}/lib/intel64/vc10 )

# Include standard OpenCL headers
include_directories( ${BOLT_ROOT_DIR}/BoltForOpenCL ${BOLT_ROOT_DIR}/BoltForC++AMP ${OPENCL_INCLUDE_DIRS} ${TBB_INCLUDE_DIR})
link_directories(${TBB_LIBRARY_DIR} )

add_executable( examples.Hessian ${examples.Hessian.Files} )

if(BUILD_TBB)
    target_link_libraries( examples.Hessian ${BOLT_LIBRARIES} ${OPENCL_LIBRARIES} ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  ${TBB_LIBRARIES} )
else (BUILD_TBB)
    target_link_libraries( examples.Hessian ${BOLT_LIBRARIES} ${OPENCL_LIBRARIES} ${GTEST_LIBRARIES} ${Boost_LIBRARIES}  )
endif()


set_target_properties( examples.Hessian PROPERTIES VERSION ${Bolt_VERSION} )
set_target_properties( examples.Hessian PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

            #FILE($ENV{TBBROOT})/bin/intel64/vc10/tbb.dll

install( FILES	
			${BOLTCL_DIR}/transform_reduce_kernels.cl
            ${HESSIAN_DIR}/hessian.cl
            ${HESSIAN_DIR}/inputs/marina.bmp
            ${TROOT}/bin/intel64/vc10/tbb.dll
DESTINATION
        	${BIN_DIR} )
        
install( FILES	
			${BOLTCL_DIR}/transform_reduce_kernels.cl
            ${HESSIAN_DIR}/hessian.cl
            ${HESSIAN_DIR}/inputs/marina.bmp
            ${TROOT}/bin/intel64/vc10/tbb.dll
		DESTINATION
			${PROJECT_BINARY_DIR}/staging/\${CMAKE_INSTALL_CONFIG_NAME} )

install( FILES	
            ${HESSIAN_DIR}/inputs/marina.bmp
            ${HESSIAN_DIR}/hessian.cl
		DESTINATION
			${PROJECT_BINARY_DIR}/Hessian/ )

# CPack configuration; include the executable into the package
install( TARGETS examples.Hessian
	RUNTIME DESTINATION ${BIN_DIR}
	LIBRARY DESTINATION ${LIB_DIR}
	ARCHIVE DESTINATION ${LIB_DIR}/import
	)
