function(add_version_info_from_git VERS PATCH_P SDK_COMMIT_P LLVM_COMMIT_P)
  # get date information based on UTC
  # use the last two digits of year + week number + day in the week as HCC_VERSION_PATCH
  # use the commit date, instead of build date
  # add xargs to remove strange trailing newline character
  execute_process(COMMAND git show -s --format=@%ct
                  COMMAND xargs
                  COMMAND date -f - --utc +%y%U%w
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                  OUTPUT_VARIABLE PATCH
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  # get commit information
  execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                OUTPUT_VARIABLE SDK_COMMIT
                OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND git rev-parse --short HEAD
                  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/llvm-project
                  OUTPUT_VARIABLE LLVM_COMMIT
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${VERS} "${${VERS}}.${PATCH}-${SDK_COMMIT}-${LLVM_COMMIT}" PARENT_SCOPE)
  set(${PATCH_P} "${PATCH}" PARENT_SCOPE)
  set(${SDK_COMMIT_P} "${SDK_COMMIT}" PARENT_SCOPE)
  set(${LLVM_COMMIT_P} "${LLVM_COMMIT}" PARENT_SCOPE)
endfunction(add_version_info_from_git)
