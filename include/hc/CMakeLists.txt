set(
    HC_headers
        hc_agent_pool.hpp
        hc_aligned_alloc.hpp
        hc_am_internal.hpp
        hc_am.hpp
        hc_atomics.hpp
        hc_callable_attributes.hpp
        hc_defines.hpp
        hc_exception.hpp
        hc_index.hpp
        hc_kernel_emitter.hpp
        hc_launch.hpp
        hc_math.hpp
        hc_norm_unorm.hpp
        hc_printf.hpp
        hc_queue_pool.hpp
        hc_rt_debug.hpp
        hc_runtime.hpp
        hc_short_vector.hpp
        hc_signal_pool.hpp
        hc.hpp
        implementation/hc_code_object_bundle.hpp
        implementation/hc_program_state.hpp
        implementation/hc_raii_handle.hpp
        implementation/hc_type_support.hpp)

# Set location for output directory
set(output_dir "${PROJECT_BINARY_DIR}/include/hc")
set(out_files)
foreach(f ${HC_headers})
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
add_custom_target(hc-headers ALL DEPENDS ${out_files})
add_dependencies(world hc-headers)

# Install command for headers
install(
    FILES ${HC_headers}
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DESTINATION include)