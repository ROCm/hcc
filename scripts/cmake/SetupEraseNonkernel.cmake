macro(setup_EraseNonkernel dest_dir name)

if(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
MESSAGE("EraseNonkernel pass seems to present.")
else(EXISTS "${dest_dir}/${name}" AND IS_SYMLINK "${dest_dir}/${name}")
MESSAGE("Setting up EraseNonkernel.")
execute_process( COMMAND ln -s ${PROJECT_SOURCE_DIR}/${name} ${dest_dir}/${name} )
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/EraseNonkernel.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()
endmacro()
