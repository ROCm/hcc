macro(patch_SimplifyCFG name)

MESSAGE("Patch SimplifyCFG to aware NoDuplicate attribute.")
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/SimplifyCFG.diff
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

endmacro()
