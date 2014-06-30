macro(ensure_clang_is_present dest_dir name url)

#TODO: why is CLANG_URL set to "." in CMakeLists.txt:13
string(COMPARE EQUAL "${url}" "." default_clang)
if(default_clang)
 set(REPO https://bitbucket.org/multicoreware/cppamp-ng.git)
else()
 set(REPO "${url}")
endif()

if(EXISTS "${dest_dir}/${name}/tools/clang")
  MESSAGE("clang is present.")
else(EXISTS "${dest_dir}/${name}/tools/clang")
  MESSAGE("Cloning clang from ${REPO}...")
  Find_Package(Git)
  Find_Program(GITL_EXECUTABLE git)
  execute_process( COMMAND ${GIT_EXECUTABLE} clone ${REPO} ${dest_dir}/${name}/tools/clang )
endif(EXISTS "${dest_dir}/${name}/tools/clang")

endmacro()
