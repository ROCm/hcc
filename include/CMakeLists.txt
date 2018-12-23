# Handle HC Legacy, for HIP compatibility (forever?)
# Put all hcc headers into the hcc-headers target
# .h and .hpp headers
set(HCC_headers
    array_view
    coordinate
    hc_am_internal.hpp
    hc_am.hpp
    hc_defines.h
    hc_math.hpp
    hc_printf.hpp
    hc_rt_debug.h
    hc.hpp
    hcc_features.hpp
    kalmar_aligned_alloc.h
    kalmar_exception.h
    kalmar_runtime.h)

# Set location for output directory
set(output_dir "${PROJECT_BINARY_DIR}/include")
set(out_files)
foreach(f ${HCC_headers})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    set(dst ${output_dir}/${f})
    add_custom_command(
        OUTPUT ${dst}
        DEPENDS ${src}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
        COMMENT "Copying HCC's ${f}...")
    list(APPEND out_files ${dst})
endforeach()

# Create target for hcc-headers and set dependencies
add_custom_target(hcc-headers ALL DEPENDS ${out_files})
add_dependencies(world hcc-headers)

# Install command for headers
install(
    FILES ${HCC_headers}
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DESTINATION include)

# Handle pSTL
add_subdirectory(experimental)

# Handle HC Next (post ROCm 2.0)
add_subdirectory(hc)