macro(setup_Utils name)

MESSAGE("Patching utils.")
execute_process( COMMAND sh ${PROJECT_SOURCE_DIR}/${name}/TestRunner.patch
                 WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endmacro()
