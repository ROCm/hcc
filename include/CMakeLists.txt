#install commands for headers

FILE(GLOB H_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h*)

FILE(GLOB INL_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.inl)


set(HCC_HEADERS)

#Obtain the names of each Header File
foreach(InFName ${H_HEADERS} ${INL_HEADERS})
  STRING(REGEX REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/ "" OutFName ${InFName})
  set(HCC_HEADERS ${HCC_HEADERS} "${OutFName}")
endforeach(InFName)


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
