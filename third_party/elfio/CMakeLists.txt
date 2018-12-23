set(elfio_headers
    elf_types.hpp
    elfio_amdgpu.hpp
    elfio_dump.hpp
    elfio_dynamic.hpp
    elfio_header.hpp
    elfio_note.hpp
    elfio_relocation.hpp
    elfio_section.hpp
    elfio_segment.hpp
    elfio_strings.hpp
    elfio_symbols.hpp
    elfio_utils.hpp
    elfio.hpp)
set(elfio_license COPYING)

# Set location for output directory
set(output_dir "${PROJECT_BINARY_DIR}/third_party/elfio")
set(out_files)
foreach(f ${elfio_headers} ${elfio_license})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    set(dst ${output_dir}/${f})
    add_custom_command(
        OUTPUT ${dst}
        DEPENDS ${src}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
        COMMENT "Copying ELFIO's ${f}...")
    list(APPEND out_files ${dst})
endforeach()

# Create target for hcc-headers and set dependencies
add_custom_target(elfio-headers ALL DEPENDS ${out_files})
add_dependencies(world hc-headers hc-impl-headers)

# Install command for headers
install(
    FILES ${elfio_headers}
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DESTINATION third_party/elfio)