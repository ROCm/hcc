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

set( ampBolt.Include.Dir ${BOLT_INCLUDE_DIR}/bolt/amp )

set( ampBolt.Runtime.Source
        bolt.cpp
        control.cpp
        ${BOLT_LIBRARY_DIR}/statisticalTimer.cpp 
   )

set( ampBolt.Runtime.Headers 
	${ampBolt.Include.Dir}/bolt.h 
	${ampBolt.Include.Dir}/control.h 
	${ampBolt.Include.Dir}/binary_search.h
	${ampBolt.Include.Dir}/count.h
	${ampBolt.Include.Dir}/device_vector.h 
	${ampBolt.Include.Dir}/functional.h 
	${ampBolt.Include.Dir}/merge.h	
	${ampBolt.Include.Dir}/min_element.h
	${ampBolt.Include.Dir}/max_element.h	
	${ampBolt.Include.Dir}/copy.h
	${ampBolt.Include.Dir}/fill.h	
	${ampBolt.Include.Dir}/gather.h
	${ampBolt.Include.Dir}/generate.h	
	${ampBolt.Include.Dir}/inner_product.h	
	${ampBolt.Include.Dir}/pair.h 		
	${ampBolt.Include.Dir}/parallel_iteration.h 
	${ampBolt.Include.Dir}/pool_alloc.h 
	${ampBolt.Include.Dir}/reduce.h 
	${ampBolt.Include.Dir}/reduce_by_key.h 
	${ampBolt.Include.Dir}/scan.h 
	${ampBolt.Include.Dir}/scatter.h 
	${ampBolt.Include.Dir}/scan_by_key.h 
	${ampBolt.Include.Dir}/sort.h
	${ampBolt.Include.Dir}/sort_by_key.h    
	${ampBolt.Include.Dir}/stablesort.h 
	${ampBolt.Include.Dir}/stablesort_by_key.h
	${ampBolt.Include.Dir}/synchronized_view.h 
	${ampBolt.Include.Dir}/transform.h 
	${ampBolt.Include.Dir}/transform_reduce.h 
	${ampBolt.Include.Dir}/transform_reduce_range.h
	${ampBolt.Include.Dir}/transform_scan.h
    )
set( ampBolt.Runtime.Headers.Iterator
	${ampBolt.Include.Dir}/iterator/constant_iterator.h
	${ampBolt.Include.Dir}/iterator/counting_iterator.h
	${ampBolt.Include.Dir}/iterator/iterator_traits.h 
  ${ampBolt.Include.Dir}/iterator/permutation_iterator.h
   )
    
set( ampBolt.Runtime.Headers.Detail
	${ampBolt.Include.Dir}/detail/copy.inl
	${ampBolt.Include.Dir}/detail/count.inl
	${ampBolt.Include.Dir}/detail/binary_search.inl
	${ampBolt.Include.Dir}/detail/fill.inl
	${ampBolt.Include.Dir}/detail/gather.inl
	${ampBolt.Include.Dir}/detail/generate.inl
	${ampBolt.Include.Dir}/detail/inner_product.inl
	${ampBolt.Include.Dir}/detail/merge.inl
	${ampBolt.Include.Dir}/detail/min_element.inl
	${ampBolt.Include.Dir}/detail/pair.inl
	${ampBolt.Include.Dir}/detail/reduce.inl
	${ampBolt.Include.Dir}/detail/reduce_by_key.inl
	${ampBolt.Include.Dir}/detail/scan.inl
	${ampBolt.Include.Dir}/detail/scatter.inl
	${ampBolt.Include.Dir}/detail/scan_by_key.inl
	${ampBolt.Include.Dir}/detail/sort.inl
	${ampBolt.Include.Dir}/detail/sort_by_key.inl
	${ampBolt.Include.Dir}/detail/stablesort.inl
	${ampBolt.Include.Dir}/detail/stablesort_by_key.inl
	${ampBolt.Include.Dir}/detail/transform.inl
	${ampBolt.Include.Dir}/detail/transform_reduce.inl
	${ampBolt.Include.Dir}/detail/transform_scan.inl
   )


# The minidump tech is windows specific; I don't have a solution yet for linux, but google-breakpad looks promising
if( WIN32 )
    list( APPEND ampBolt.Runtime.Source ${BOLT_LIBRARY_DIR}/miniDump.cpp )
    list( APPEND ampBolt.Runtime.Headers ${ampBolt.Include.Dir}/../miniDump.h )
endif( )

source_group( "Inline Files" REGULAR_EXPRESSION .*inl$ )

set( ampBolt.Runtime.Files
  ${ampBolt.Runtime.Source}
  ${ampBolt.Runtime.Headers}
  ${ampBolt.Runtime.Headers.Iterator}  
  ${ampBolt.Runtime.Headers.Detail}
  )

# Include standard OpenCL headers
include_directories(
    ${BOLT_INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}/include
    ${OPENCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS} 
    ${CLAMP_INC_DIRS} )
add_library( ampBolt.Runtime STATIC ${ampBolt.Runtime.Files} )

# Construct a meaningful name for this build of the library
set( boltLibName "ampBolt.runtime" )
if( WIN32 )
    if( MSVC_VERSION VERSION_LESS 1600 )
        set( boltLibName "${boltLibName}.vc90" )
    elseif( MSVC_VERSION VERSION_LESS 1700 )
        set( boltLibName "${boltLibName}.vc100" )
    elseif( MSVC_VERSION VERSION_LESS 1800 )
        set( boltLibName "${boltLibName}.vc110" )
    else( )
        set( boltLibName "${boltLibName}.vc120" )
    endif( )
elseif( UNIX )
    set( boltLibName "${boltLibName}.clang" )
endif( )

set_target_properties( ampBolt.Runtime PROPERTIES VERSION ${Bolt_VERSION} SOVERSION ${Bolt_VERSION_MAJOR} )
set_target_properties( ampBolt.Runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( ampBolt.Runtime PROPERTIES OUTPUT_NAME ${boltLibName} IMPORT_SUFFIX ".import" DEBUG_POSTFIX ".debug" )

install( FILES
            ${ampBolt.Runtime.Headers}
        DESTINATION
            ${INCLUDE_DIR}/bolt/amp )

install( FILES
            ${ampBolt.Runtime.Headers.Iterator}
        DESTINATION
            ${INCLUDE_DIR}/bolt/amp/iterator )
            
install( FILES
            ${ampBolt.Runtime.Headers.Detail}
        DESTINATION
            ${INCLUDE_DIR}/bolt/amp/detail )

# CPack configuration; include the executable into the package
install( TARGETS ampBolt.Runtime
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}
    )
