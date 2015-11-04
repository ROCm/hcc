macro(setup_HC dest_dir name)

if(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
  MESSAGE("HC passes seem to present.")
else(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
  MESSAGE("Setting up HC.")
execute_process( COMMAND ln -s ${PROJECT_SOURCE_DIR}/${name} ${dest_dir}/${name} )
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/HC.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
endmacro()
