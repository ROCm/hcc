# Here we have PSTL header files inside experimental directory and impl sub-directory.
# All PSTL headers should be in the same pstl-headers target, but different output dir

# PSTL headers from experimental/
set(PSTL_EXP_HEADERS
  algorithm
  exception_list
  execution_policy
  numeric)

# PSTL headers from experimental/impl/
FILE(GLOB PSTL_IMPL ${CMAKE_CURRENT_SOURCE_DIR}/impl/*.inl)

# Obtain the names of each impl header and add to PSTL
foreach(InFName ${PSTL_IMPL})
  STRING(REGEX REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/ "" OutFName ${InFName})
  set(PSTL_IMPL_HEADERS ${PSTL_IMPL_HEADERS} "${OutFName}")
endforeach(InFName)

# Set location for exp/ output directory
set(exp_output_dir "${PROJECT_BINARY_DIR}/include/experimental")
# Set location for exp/impl/ output directory
set(impl_output_dir "${PROJECT_BINARY_DIR}/include/experimental/impl")

set(exp_out_files)
set(impl_out_files)

foreach( f ${PSTL_EXP_HEADERS} )
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${exp_output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying HCC's ${f}...")
  list(APPEND exp_out_files ${dst})
endforeach( f )

foreach( f ${PSTL_IMPL_HEADERS} )
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${impl_output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying HCC's ${f}...")
  list(APPEND impl_out_files ${dst})
endforeach( f )

# Create target for pstl-headers and set dependencies
add_custom_target(pstl-headers ALL DEPENDS ${exp_out_files} ${impl_out_files})
add_dependencies(world pstl-headers)

# Install command for PSTL exp headers
install(FILES ${PSTL_EXP_HEADERS}
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/experimental)
  
# Install command for PSTL impl headers
install(FILES ${PSTL_IMPL_HEADERS}
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include/experimental/impl)

