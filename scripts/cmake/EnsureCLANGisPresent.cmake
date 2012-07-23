macro(ensure_clang_is_present dest_dir name)

if(EXISTS "${dest_dir}/${name}/tools/clang")
  MESSAGE("clang is present.")
else(EXISTS "${dest_dir}/${name}/tools/clang")
  MESSAGE("Cloning clang at ${dest_dir}/${name}/tools/clang...")
  MESSAGE("Enter your ID of the repository:")
  Find_Package(Mercurial)
  Find_Program(MERCIRUAL_EXECUTABLE hg)
  execute_process( COMMAND ${MERCURIAL_EXECUTABLE} clone https://bitbucket.org/multicoreware/cppamp -u cppamp ${dest_dir}/${name}/tools/clang )
endif(EXISTS "${dest_dir}/${name}/tools/clang")

endmacro()

