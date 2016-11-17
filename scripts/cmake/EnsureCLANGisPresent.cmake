macro(ensure_clang_is_present dest_dir name url)

string(COMPARE EQUAL "${url}" "." default_clang)
if(default_clang)
 set(REPO https://github.com/RadeonOpenCompute/hcc-clang.git)
else()
 set(REPO "${url}")
endif()

if(EXISTS "${dest_dir}/${name}/tools/clang")
  MESSAGE("hcc-clang is present.")
else(EXISTS "${dest_dir}/${name}/tools/clang")
  Find_Package(Git)
  Find_Program(GITL_EXECUTABLE git)

  # determine current branch of hcc
  execute_process(COMMAND ${GIT_EXECUTABLE} symbolic-ref --short HEAD
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                  OUTPUT_VARIABLE KALMAR_BRANCH_NAME
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # query if the branch exist
  execute_process(COMMAND ${GIT_EXECUTABLE} ls-remote --heads ${REPO} ${KALMAR_BRANCH_NAME} 
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                  OUTPUT_VARIABLE KALMAR_CLANG_HAS_SAME_BRANCH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(KALMAR_BRANCH_NAME AND KALMAR_CLANG_HAS_SAME_BRANCH)
    # use the same branch as hcc
    MESSAGE("Cloning hcc-clang from branch ${KALMAR_BRANCH_NAME} of ${REPO}...")
    execute_process( COMMAND ${GIT_EXECUTABLE} clone -b ${KALMAR_BRANCH_NAME} ${REPO} ${dest_dir}/${name}/tools/clang )
  else(KALMAR_CLANG_HAS_SAME_BRANCH)
    # branch not found, use default one
    MESSAGE("Cloning hcc-clang from roc-1.3.x branch of ${REPO}...")
    execute_process( COMMAND ${GIT_EXECUTABLE} clone -b roc-1.3.x   ${REPO} ${dest_dir}/${name}/tools/clang )
  endif()

endif(EXISTS "${dest_dir}/${name}/tools/clang")

endmacro()
