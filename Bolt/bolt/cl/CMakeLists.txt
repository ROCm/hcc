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

set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_LIBS   ON )
set( Boost_DETAILED_FAILURE_MSG   ON )
# set( Boost_DEBUG ON )
# set( BOOST_LIB_DIAGNOSTIC ON )

if( BOOST_ROOT )
    # The root tree of BOOST was specified on the command line; use it to to find the specific Boost the user points too
    find_package( Boost ${Boost.VERSION} COMPONENTS thread system date_time chrono REQUIRED )
    # This will define Boost_FOUND
else( )
    message( "Configure Bolt in <BOLT_ROOT>/bin to build the SuperBuild which will download and build Boost automatically" )
    message( SEND_ERROR "The OpenCL Bolt library requires BOOST_ROOT to be defined" )
endif( )

# A function to help debug cmake files; this will print each element of a list line by line
function( printList myList )
    foreach( element ${${myList}} )
        message( "${myList} element: [" ${element} "]" )
    endforeach( )
endfunction( )

# A macro to create a new list ${resultList} from a given list of filenames and a path to prepend on each element
macro( prependPath resultList filesList path )
    foreach( kernelFile ${${filesList}} )
        list( APPEND ${resultList} "${path}/${kernelFile}" )
        # message( "prependPath element: [" ${${resultList}} "]" )
    endforeach( )
endmacro( )

# Macro that iterates through a source list, and appends into a result list the results of a substitution
macro( substList match_string replace_string resultList sourceList )
    foreach( sourceString ${${sourceList}} )
        string( REPLACE ${match_string} ${replace_string} sourceString ${sourceString} )
        list( APPEND ${resultList} ${sourceString} )
    endforeach( )
endmacro( )

set( clBolt.Include.Dir ${BOLT_INCLUDE_DIR}/bolt/cl )

set( tbb.Include.Dir ${BOLT_INCLUDE_DIR}/bolt/btbb )

set( clBolt.Runtime.Source
        bolt.cpp
        control.cpp
        ${BOLT_LIBRARY_DIR}/statisticalTimer.cpp
        ${BOLT_LIBRARY_DIR}/AsyncProfiler.cpp
    )

set( clBolt.Runtime.Headers
        ${clBolt.Include.Dir}/bolt.h
        ${clBolt.Include.Dir}/clcode.h
        ${clBolt.Include.Dir}/control.h
        ${clBolt.Include.Dir}/binary_search.h
        ${clBolt.Include.Dir}/copy.h
        ${clBolt.Include.Dir}/count.h
        ${clBolt.Include.Dir}/device_vector.h
        ${clBolt.Include.Dir}/distance.h
        ${clBolt.Include.Dir}/functional.h
        ${clBolt.Include.Dir}/fill.h
        ${clBolt.Include.Dir}/gather.h
        ${clBolt.Include.Dir}/generate.h
        ${clBolt.Include.Dir}/inner_product.h
        ${clBolt.Include.Dir}/max_element.h
        ${clBolt.Include.Dir}/merge.h
        ${clBolt.Include.Dir}/min_element.h
        ${clBolt.Include.Dir}/pair.h
        ${clBolt.Include.Dir}/reduce.h
        ${clBolt.Include.Dir}/reduce_by_key.h
        ${clBolt.Include.Dir}/scan.h
        ${clBolt.Include.Dir}/scan_by_key.h
        ${clBolt.Include.Dir}/scatter.h
        ${clBolt.Include.Dir}/sort.h
        ${clBolt.Include.Dir}/sort_by_key.h
        ${clBolt.Include.Dir}/stablesort.h
        ${clBolt.Include.Dir}/stablesort_by_key.h
        ${clBolt.Include.Dir}/transform.h
        ${clBolt.Include.Dir}/transform_reduce.h
        ${clBolt.Include.Dir}/transform_scan.h
    )

set( clBolt.Runtime.Headers.Iterator
        ${clBolt.Include.Dir}/iterator/addressof.h
        ${clBolt.Include.Dir}/iterator/facade_iterator_category.h
        ${clBolt.Include.Dir}/iterator/iterator_adaptor.h
        ${clBolt.Include.Dir}/iterator/iterator_categories.h
        ${clBolt.Include.Dir}/iterator/iterator_facade.h
        ${clBolt.Include.Dir}/iterator/iterator_traits.h
        ${clBolt.Include.Dir}/iterator/constant_iterator.h
        ${clBolt.Include.Dir}/iterator/counting_iterator.h
        ${clBolt.Include.Dir}/iterator/transform_iterator.h
        ${clBolt.Include.Dir}/iterator/permutation_iterator.h
    )

set( clBolt.Runtime.Headers.Misc
        ${clBolt.Include.Dir}/../countof.h
        ${clBolt.Include.Dir}/../unicode.h
        ${clBolt.Include.Dir}/../BoltLog.h
        ${clBolt.Include.Dir}/../statisticalTimer.h
        ${clBolt.Include.Dir}/../AsyncProfiler.h
    )

set( clBolt.Runtime.Headers.Detail
        ${clBolt.Include.Dir}/detail/binary_search.inl
        ${clBolt.Include.Dir}/detail/copy.inl
        ${clBolt.Include.Dir}/detail/count.inl
        ${clBolt.Include.Dir}/detail/distance.inl
        ${clBolt.Include.Dir}/detail/fill.inl
        ${clBolt.Include.Dir}/detail/gather.inl
        ${clBolt.Include.Dir}/detail/generate.inl
        ${clBolt.Include.Dir}/detail/inner_product.inl
        ${clBolt.Include.Dir}/detail/merge.inl
        ${clBolt.Include.Dir}/detail/min_element.inl
        ${clBolt.Include.Dir}/detail/pair.inl
        ${clBolt.Include.Dir}/detail/reduce.inl
        ${clBolt.Include.Dir}/detail/reduce_by_key.inl
        ${clBolt.Include.Dir}/detail/scan.inl
        ${clBolt.Include.Dir}/detail/scan_by_key.inl
        ${clBolt.Include.Dir}/detail/scatter.inl
        ${clBolt.Include.Dir}/detail/sort.inl
        ${clBolt.Include.Dir}/detail/sort_by_key.inl
        ${clBolt.Include.Dir}/detail/stablesort.inl
        ${clBolt.Include.Dir}/detail/stablesort_by_key.inl
        ${clBolt.Include.Dir}/detail/transform.inl
        ${clBolt.Include.Dir}/detail/transform_reduce.inl
        ${clBolt.Include.Dir}/detail/transform_scan.inl
        ${clBolt.Include.Dir}/detail/type_traits.h
    )

set( clBolt.Runtime.clFiles
        fill_kernels.cl
        copy_kernels.cl
        binary_search_kernels.cl
        count_kernels.cl
        gather_kernels.cl
        generate_kernels.cl
        min_element_kernels.cl
        merge_kernels.cl
        reduce_kernels.cl
        reduce_by_key_kernels.cl
        transform_kernels.cl
        transform_reduce_kernels.cl
        transform_scan_kernels.cl
        scan_kernels.cl
        scan_by_key_kernels.cl
        scatter_kernels.cl
        sort_kernels.cl
        stablesort_kernels.cl
        stablesort_by_key_kernels.cl
        sort_common_kernels.cl
        sort_uint_kernels.cl
        sort_by_key_uint_kernels.cl
        sort_int_kernels.cl
        sort_by_key_int_kernels.cl
        sort_by_key_kernels.cl
    )

set( tbb.Runtime.Headers
    ${tbb.Include.Dir}/binary_search.h
    ${tbb.Include.Dir}/copy.h
    ${tbb.Include.Dir}/count.h
    ${tbb.Include.Dir}/fill.h
    ${tbb.Include.Dir}/gather.h
    ${tbb.Include.Dir}/generate.h
    ${tbb.Include.Dir}/inner_product.h
    ${tbb.Include.Dir}/merge.h
    ${tbb.Include.Dir}/min_element.h
    ${tbb.Include.Dir}/reduce.h
    ${tbb.Include.Dir}/reduce_by_key.h
    ${tbb.Include.Dir}/scan.h
    ${tbb.Include.Dir}/scan_by_key.h
    ${tbb.Include.Dir}/scatter.h
    ${tbb.Include.Dir}/sort.h
    ${tbb.Include.Dir}/sort_by_key.h
    ${tbb.Include.Dir}/stable_sort.h
    ${tbb.Include.Dir}/stable_sort_by_key.h
    ${tbb.Include.Dir}/transform.h
    ${tbb.Include.Dir}/transform_reduce.h
    )

set( tbb.Runtime.Headers.Detail
    ${tbb.Include.Dir}/detail/binary_search.inl
    ${tbb.Include.Dir}/detail/copy.inl
    ${tbb.Include.Dir}/detail/count.inl
    ${tbb.Include.Dir}/detail/fill.inl
    ${tbb.Include.Dir}/detail/gather.inl
    ${tbb.Include.Dir}/detail/generate.inl
    ${tbb.Include.Dir}/detail/inner_product.inl
    ${tbb.Include.Dir}/detail/merge.inl
    ${tbb.Include.Dir}/detail/min_element.inl
    ${tbb.Include.Dir}/detail/reduce.inl
    ${tbb.Include.Dir}/detail/reduce_by_key.inl
    ${tbb.Include.Dir}/detail/scan.inl
    ${tbb.Include.Dir}/detail/scan_by_key.inl
    ${tbb.Include.Dir}/detail/scatter.inl
    ${tbb.Include.Dir}/detail/sort.inl
    ${tbb.Include.Dir}/detail/sort_by_key.inl
    ${tbb.Include.Dir}/detail/stable_sort.inl
    ${tbb.Include.Dir}/detail/stable_sort_by_key.inl
    ${tbb.Include.Dir}/detail/transform.inl
    ${tbb.Include.Dir}/detail/transform_reduce.inl
    )

# Create a list of .cl files that we would like to be a part of the IDE
prependPath( clBolt.Runtime.clFiles.FullPath clBolt.Runtime.clFiles ${clBolt.Include.Dir} )
# printList( clBolt.Runtime.clFiles.FullPath )

# The minidump tech is windows specific; I don't have a solution yet for linux, but google-breakpad looks promising
if( WIN32 )
    list( APPEND clBolt.Runtime.Source ${BOLT_LIBRARY_DIR}/miniDump.cpp )
    list( APPEND clBolt.Runtime.Headers.Misc ${clBolt.Include.Dir}/../miniDump.h )
endif( )

# These create convenient file folders for our inline and kernel files
source_group( "Inline Files" REGULAR_EXPRESSION .*inl$ )
source_group( "Kernel Files" REGULAR_EXPRESSION .*cl$ )

set( clBolt.Runtime.Files
  ${clBolt.Runtime.Source}
  ${clBolt.Runtime.Headers}
  ${clBolt.Runtime.Headers.Detail}
  ${clBolt.Runtime.Headers.Iterator}
  ${clBolt.Runtime.Headers.Misc}
  ${clBolt.Runtime.clFiles.FullPath} )

# Include headers files
include_directories(
    ${BOLT_INCLUDE_DIR}
    ${PROJECT_BINARY_DIR}/include
    ${OPENCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS} )

# Create a list of files that we expect to generate from the kernel files
substList( ".cl" ".hpp" clBolt.Runtime.hppFiles clBolt.Runtime.clFiles )
# printList( clBolt.Runtime.hppFiles )

prependPath( clBolt.Runtime.hppFiles.FullPath clBolt.Runtime.hppFiles ${PROJECT_BINARY_DIR}/include/bolt )
# printList( clBolt.Runtime.hppFiles.FullPath )

add_custom_command(
  OUTPUT ${clBolt.Runtime.hppFiles.FullPath}
  COMMAND clBolt.StringifyKernels -d "${PROJECT_BINARY_DIR}/include/bolt/" ${clBolt.Runtime.clFiles.FullPath}
  DEPENDS clBolt.StringifyKernels ${clBolt.Runtime.clFiles.FullPath}
  COMMENT "Creating kernel header files from .cl files"
  VERBATIM
)

add_library( clBolt.Runtime STATIC ${clBolt.Runtime.Files} ${clBolt.Runtime.hppFiles.FullPath} )
target_link_libraries( clBolt.Runtime ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} )

# Construct a meaningful name for this build of the library


set( boltLibName "clBolt.runtime" )
if( MSVC )
    if( MSVC_VERSION VERSION_LESS 1600 )
        set( boltLibName "${boltLibName}.vc90" )
    elseif( MSVC_VERSION VERSION_LESS 1700 )
        set( boltLibName "${boltLibName}.vc100" )
    elseif( MSVC_VERSION VERSION_LESS 1800 )
        set( boltLibName "${boltLibName}.vc110" )
    else()
        set( boltLibName "${boltLibName}.vc120" )
    endif( )
else()
    set( boltLibName "${boltLibName}.gcc" )
endif()

set_target_properties( clBolt.Runtime PROPERTIES VERSION ${Bolt_VERSION} SOVERSION ${Bolt_VERSION_MAJOR} )
set_target_properties( clBolt.Runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( clBolt.Runtime PROPERTIES OUTPUT_NAME ${boltLibName} IMPORT_SUFFIX ".import" DEBUG_POSTFIX ".debug" )

install( FILES
            ${clBolt.Runtime.Headers}
        DESTINATION
            ${INCLUDE_DIR}/bolt/cl )

install( FILES
            ${clBolt.Runtime.Headers.Iterator}
        DESTINATION
            ${INCLUDE_DIR}/bolt/cl/iterator )

install( FILES
            ${clBolt.Runtime.Headers.Detail}
        DESTINATION
            ${INCLUDE_DIR}/bolt/cl/detail )

install( FILES
            ${clBolt.Runtime.Headers.Misc}
        DESTINATION
            ${INCLUDE_DIR}/bolt )

install( FILES
            ${tbb.Runtime.Headers}
        DESTINATION
            ${INCLUDE_DIR}/bolt/btbb )

install( FILES
            ${tbb.Runtime.Headers.Detail}
        DESTINATION
            ${INCLUDE_DIR}/bolt/btbb/detail )

# Install dependent Boost header files
set( ROOT_EXTERNAL_BOOST ${PROJECT_BINARY_DIR}/../external/boost/src/Boost/boost )

# smart_ptr library
install( DIRECTORY
            ${ROOT_EXTERNAL_BOOST}/smart_ptr
            ${ROOT_EXTERNAL_BOOST}/detail
        DESTINATION
            ${INCLUDE_DIR}/boost )

install( FILES
            ${ROOT_EXTERNAL_BOOST}/enable_shared_from_this.hpp
            ${ROOT_EXTERNAL_BOOST}/get_pointer.hpp
            ${ROOT_EXTERNAL_BOOST}/intrusive_ptr.hpp
            ${ROOT_EXTERNAL_BOOST}/make_shared.hpp
            ${ROOT_EXTERNAL_BOOST}/memory_order.hpp
            ${ROOT_EXTERNAL_BOOST}/pointer_cast.hpp
            ${ROOT_EXTERNAL_BOOST}/pointer_to_other.hpp
            ${ROOT_EXTERNAL_BOOST}/scoped_array.hpp
            ${ROOT_EXTERNAL_BOOST}/scoped_ptr.hpp
            ${ROOT_EXTERNAL_BOOST}/shared_array.hpp
            ${ROOT_EXTERNAL_BOOST}/shared_ptr.hpp
            ${ROOT_EXTERNAL_BOOST}/smart_ptr.hpp
            ${ROOT_EXTERNAL_BOOST}/weak_ptr.hpp
        DESTINATION
            ${INCLUDE_DIR}/boost )

# iterator library
install( DIRECTORY
            ${ROOT_EXTERNAL_BOOST}/iterator
            ${ROOT_EXTERNAL_BOOST}/pending
            ${ROOT_EXTERNAL_BOOST}/config
            ${ROOT_EXTERNAL_BOOST}/mpl
            ${ROOT_EXTERNAL_BOOST}/type_traits
            ${ROOT_EXTERNAL_BOOST}/preprocessor
            ${ROOT_EXTERNAL_BOOST}/utility
            ${ROOT_EXTERNAL_BOOST}/exception
        DESTINATION
            ${INCLUDE_DIR}/boost )

install( FILES
            ${ROOT_EXTERNAL_BOOST}/config.hpp
            ${ROOT_EXTERNAL_BOOST}/function_output_iterator.hpp
            ${ROOT_EXTERNAL_BOOST}/iterator.hpp
            ${ROOT_EXTERNAL_BOOST}/iterator_adaptors.hpp
            ${ROOT_EXTERNAL_BOOST}/pointee.hpp
            ${ROOT_EXTERNAL_BOOST}/shared_container_iterator.hpp
            ${ROOT_EXTERNAL_BOOST}/static_assert.hpp
            ${ROOT_EXTERNAL_BOOST}/utility.hpp
            ${ROOT_EXTERNAL_BOOST}/checked_delete.hpp
            ${ROOT_EXTERNAL_BOOST}/next_prior.hpp
            ${ROOT_EXTERNAL_BOOST}/noncopyable.hpp
            ${ROOT_EXTERNAL_BOOST}/assert.hpp
            ${ROOT_EXTERNAL_BOOST}/current_function.hpp
            ${ROOT_EXTERNAL_BOOST}/throw_exception.hpp
        DESTINATION
            ${INCLUDE_DIR}/boost )

# thread library
install( DIRECTORY
            ${ROOT_EXTERNAL_BOOST}/thread
            ${ROOT_EXTERNAL_BOOST}/system
            ${ROOT_EXTERNAL_BOOST}/bind
            ${ROOT_EXTERNAL_BOOST}/date_time
            ${ROOT_EXTERNAL_BOOST}/chrono
            ${ROOT_EXTERNAL_BOOST}/ratio
            ${ROOT_EXTERNAL_BOOST}/move
        DESTINATION
            ${INCLUDE_DIR}/boost )

install( FILES
            ${ROOT_EXTERNAL_BOOST}/version.hpp
            ${ROOT_EXTERNAL_BOOST}/cstdint.hpp
            ${ROOT_EXTERNAL_BOOST}/operators.hpp
            ${ROOT_EXTERNAL_BOOST}/cerrno.hpp
            ${ROOT_EXTERNAL_BOOST}/bind.hpp
            ${ROOT_EXTERNAL_BOOST}/ref.hpp
            ${ROOT_EXTERNAL_BOOST}/mem_fn.hpp
            ${ROOT_EXTERNAL_BOOST}/type.hpp
            ${ROOT_EXTERNAL_BOOST}/is_placeholder.hpp
            ${ROOT_EXTERNAL_BOOST}/visit_each.hpp
            ${ROOT_EXTERNAL_BOOST}/limits.hpp
            ${ROOT_EXTERNAL_BOOST}/integer_traits.hpp
        DESTINATION
            ${INCLUDE_DIR}/boost )

# Copy over our Boost static library dependencies
install( FILES
    ${Boost_SYSTEM_LIBRARY_DEBUG}
    ${Boost_THREAD_LIBRARY_DEBUG}
    ${Boost_DATE_TIME_LIBRARY_DEBUG}
    ${Boost_CHRONO_LIBRARY_DEBUG}
    DESTINATION ${LIB_DIR}
    CONFIGURATIONS Debug
    )

# Copy over our Boost static library dependencies
install( FILES
    ${Boost_SYSTEM_LIBRARY_RELEASE}
    ${Boost_THREAD_LIBRARY_RELEASE}
    ${Boost_DATE_TIME_LIBRARY_RELEASE}
    ${Boost_CHRONO_LIBRARY_RELEASE}
    DESTINATION ${LIB_DIR}
    CONFIGURATIONS Release
    )

# CPack configuration; include the executable into the package
install( TARGETS clBolt.Runtime
#    EXPORT exportBOLT
    RUNTIME DESTINATION ${BIN_DIR}
    LIBRARY DESTINATION ${LIB_DIR}
    ARCHIVE DESTINATION ${LIB_DIR}
    )

# install( EXPORT exportBOLT
        # DESTINATION exportCmake
        # )
