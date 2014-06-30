macro(ensure_gmac_is_present dest_dir name url)

#TODO: why is GMAC_URL set to "." in CMakeLists.txt:12
string(COMPARE EQUAL "${url}" "." default_gmac)
if(default_gmac)
 set(REPO https://bitbucket.org/multicoreware/gmac)
else()
 set(REPO "${url}")
endif()

if(EXISTS "${dest_dir}/${name}")
  MESSAGE("gmac is present.")
else(EXISTS "${dest_dir}/${name}")
  MESSAGE("Cloning gmac from ${REPO}...")
  Find_Package(Mercurial)
  Find_Program(MERCIRUAL_EXECUTABLE hg)
  if(NOT MERCIRUAL_EXECUTABLE)
    MESSAGE(FATAL_ERROR "no gmac, and mercurial not found")
  endif()
  execute_process( COMMAND ${MERCURIAL_EXECUTABLE} clone ${REPO} ${dest_dir}/${name} )
endif(EXISTS "${dest_dir}/${name}")

endmacro()
