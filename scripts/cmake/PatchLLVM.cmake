macro(patch_LLVM name)

MESSAGE("Patch LLVM so it has HCC EnvironmentType in Triple.")
# The patch is made by "git diff". Apply it with "git apply".
execute_process( COMMAND git apply ${PROJECT_SOURCE_DIR}/${name}/LLVM.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/compiler)

MESSAGE("Patch LLVM to support debugging metadata with address space information.")
execute_process( COMMAND git apply --whitespace=nowarn ${PROJECT_SOURCE_DIR}/${name}/Debugger.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/compiler)


endmacro()
