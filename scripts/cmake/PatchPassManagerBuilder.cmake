macro(patch_PassManagerBuilder name)

MESSAGE("Patch PassManagerBuilder to fix the tile uniform violation problem of Bolt.")
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/PassManagerBuilder.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

endmacro()
