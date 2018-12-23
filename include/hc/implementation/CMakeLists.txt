set(
    HC_impl_headers
        hc_code_object_bundle.hpp
        hc_n_way_set_associative_cache.hpp
        hc_program_state.hpp
        hc_raii_handle.hpp
        hc_type_support.hpp)

# Set location for output directory
set(output_dir "${PROJECT_BINARY_DIR}/include/hc/implementation")
set(out_files)
foreach(f ${HC_impl_headers})
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
add_custom_target(hc-impl-headers ALL DEPENDS ${out_files})
add_dependencies(world hc-impl-headers)

# Install command for headers
install(
    FILES ${HC_impl_headers}
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DESTINATION include/hc/implementation)