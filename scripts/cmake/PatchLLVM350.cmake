macro(patch_LLVM350 name)

MESSAGE("Patch LLVM3.5.0 so it can be compiled with gcc 5")
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/LLVM350.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

endmacro()
