macro(setup_HIP dest_dir name)

if(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
  MESSAGE("HIP passes seem to present.")
else(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
  MESSAGE("Setting up HIP.")
execute_process( COMMAND ln -s ${PROJECT_SOURCE_DIR}/${name} ${dest_dir}/${name} )
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/HIP.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
endmacro()
