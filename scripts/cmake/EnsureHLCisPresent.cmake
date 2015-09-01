macro(ensure_HLC_is_present dest_dir name)

if(EXISTS "${dest_dir}/${name}")
    MESSAGE("HLC already exists.")
else(EXISTS "${dest_dir}/${name}")
    MESSAGE("Downloading HLC")
    Find_Package(Git)
    if(NOT ${GIT_FOUND})
        message(FATAL_ERROR "upstream HLC is not present at ${dest_dir}/${name} and git could not be found")
    else()
        # git clone
        execute_process( COMMAND git clone https://github.com/HSAFoundation/HLC-HSAIL-Development-LLVM.git ${dest_dir}/${name} )
        # switch to hsail-stable-3.7 branch
        execute_process( COMMAND git checkout hsail-stable-3.7 WORKING_DIRECTORY ${dest_dir}/${name} )
    endif()
endif()
endmacro()
