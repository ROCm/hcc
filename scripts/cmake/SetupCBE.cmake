macro(setup_CBackend dest_dir name)

if(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
MESSAGE("CBackend seems to present.")
else(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
MESSAGE("Setting up CBackend.")
execute_process( COMMAND ln -s ${PROJECT_SOURCE_DIR}/${name} ${dest_dir}/${name} )
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/CBackend.diff
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

endif()
endmacro()
