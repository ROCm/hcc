macro(ensure_HSAILASM_is_present dest_dir name)

if(EXISTS "${dest_dir}/${name}")
    MESSAGE("HSAILasm already exists.")
elseif(HSA_USE_AMDGPU_BACKEND)
    MESSAGE("AMDGPU backend enable, not using HSAILasm")
else(EXISTS "${dest_dir}/${name}")
    if(EXISTS ${HSAIL_ASSEMBLER_DIR})
      MESSAGE("Symbolic link with specified HSAILasm")
      execute_process(COMMAND ln -fs ${HSAIL_ASSEMBLER_DIR} ${dest_dir}/${name}
                      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

    else(EXISTS ${HSAIL_ASSEMBLER_DIR})

      MESSAGE("Downloading HSAILasm")
      Find_Package(Git)
      if(NOT ${GIT_FOUND})
          message(FATAL_ERROR "upstream HSAILasm is not present at ${dest_dir}/${name} and git could not be found")
      else()
          # git clone
          execute_process( COMMAND git clone https://github.com/HSAFoundation/HSAIL-Tools.git ${dest_dir}/${name} )
      endif()

    endif(EXISTS ${HSAIL_ASSEMBLER_DIR})
endif()
endmacro()
