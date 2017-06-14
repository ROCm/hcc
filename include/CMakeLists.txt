#install commands for headers

set(H_HEADERS
  amp_math.h
  amp_short_vectors.h
  amp.h
  grid_launch.h
  grid_launch.hpp
  hc_am.hpp
  hc_defines.h
  hc_math.hpp
  hc_printf.hpp
  hc_rt_debug.h
  hc_short_vector.hpp
  hc.hpp
  hsa_atomic.h
  kalmar_aligned_alloc.h
  kalmar_buffer.h
  kalmar_cpu_launch.h
  kalmar_exception.h
  kalmar_index.h
  kalmar_launch.h
  kalmar_math.h
  kalmar_runtime.h
  kalmar_serialize.h
  pinned_vector.hpp)

set(INL_HEADERS
  hc_norm_unorm.inl
  hc_short_vector.inl
  kalmar_short_vectors.inl)

set(HCC_HEADERS
  ${H_HEADERS}
  ${INL_HEADERS})

set (output_dir "${PROJECT_BINARY_DIR}/include")

set(out_files)
foreach( f ${HCC_HEADERS} )
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying HCC's ${f}...")
  list(APPEND out_files ${dst})
endforeach( f )

add_custom_target(hcc-headers ALL DEPENDS ${out_files})
add_dependencies(world hcc-headers)
set_target_properties(hcc-headers PROPERTIES FOLDER "HCC Misc")

install(FILES ${HCC_HEADERS}
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION include)


# N4494 headers
FILE(GLOB N4494 ${CMAKE_CURRENT_SOURCE_DIR}/coordinate
                ${CMAKE_CURRENT_SOURCE_DIR}/array_view)
INSTALL(FILES ${N4494} DESTINATION include)
# Copy the file to directory matching the install directory
file(COPY ${N4494} DESTINATION "${PROJECT_BINARY_DIR}/include")

# PSTL headers
ADD_SUBDIRECTORY(experimental)
