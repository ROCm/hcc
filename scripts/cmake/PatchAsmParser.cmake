macro(patch_AsmParser name)

MESSAGE("Patch AsmParser to be compatible with LLVM 3.2.")
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/AsmParser.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

endmacro()
