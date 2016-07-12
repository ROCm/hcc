macro(patch_LLVM name)

MESSAGE("Patch LLVM so it has HCC EnvironmentType in Triple.")
# The patch is made by "git diff". Apply it with "git apply".
execute_process( COMMAND git apply ${PROJECT_SOURCE_DIR}/${name}/LLVM.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/compiler)

endmacro()
