macro(setup_Promote dest_dir name)

if(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
MESSAGE("Promote pass seems to present.")
else(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
MESSAGE("Setting up Promote.")
execute_process( COMMAND ln -s ${PROJECT_SOURCE_DIR}/${name} ${dest_dir}/${name} )
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/Promote.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
#execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/GlobalOpt.patch
#                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
endmacro()
