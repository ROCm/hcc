macro(ensure_hip_is_present dest_dir name)

  SET(LOCAL_HIP_DIR "${dest_dir}/mcwhip")
  SET(REMOTE_HIP_DIR "${dest_dir}/${name}")
  SET(HIP_HEADER_DIR "${dest_dir}/include")
  SET(HIP_LIB_DIR "${dest_dir}/lib/hip")

  macro(create_link src dest)
    execute_process(COMMAND ln -fs "${src}" "${dest}"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  endmacro()

  macro(create_copy src dest)
    execute_process(COMMAND cp "${src}" "${dest}"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  endmacro()


  if(EXISTS "${HIP_HEADER_DIR}/hip_runtime.h")
    MESSAGE("HIP headers already exists.")
  else()
    SET(USE_LOCAL_HIP FALSE)
    if(EXISTS "${REMOTE_HIP_DIR}")
      MESSAGE("HIP already exists.")
    else()
      if(DEFINED ENV{HIP_URL})
        Find_Package(Git)
        Find_Program(GITL_EXECUTABLE git)

        # TODO(Yan-Ming): change upstream url when hip is publicly available
        SET(HIP_URL $ENV{HIP_URL})
        MESSAGE("Downloading HIP from ${HIP_URL}")
        execute_process(COMMAND ${GIT_EXECUTABLE} clone ${HIP_URL} ${REMOTE_HIP_DIR})
      endif()

      if(NOT EXISTS "${REMOTE_HIP_DIR}")
        # git clone failed
        SET(USE_LOCAL_HIP TRUE)
      endif()
    endif()

    if(NOT EXISTS "${HIP_LIB_DIR}")
      execute_process(COMMAND mkdir "${HIP_LIB_DIR}"
                      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()
    create_copy("${LOCAL_HIP_DIR}/CMakeLists.txt" "${HIP_LIB_DIR}")

    if(${USE_LOCAL_HIP})
      MESSAGE("Going to use local HIP implementation")

      # XXX(Yan-Ming): fix filenames
      create_link("${LOCAL_HIP_DIR}/hip_runtime.h" "${HIP_HEADER_DIR}")
      create_link("${LOCAL_HIP_DIR}/hip-cuda.h" "${HIP_HEADER_DIR}")
      create_link("${LOCAL_HIP_DIR}/hip_runtime.cpp" "${HIP_LIB_DIR}")
    else()
      create_link("${REMOTE_HIP_DIR}/include/hip_runtime.h" "${HIP_HEADER_DIR}")
      create_link("${REMOTE_HIP_DIR}/include/hip_runtime_api.h" "${HIP_HEADER_DIR}")
      create_link("${REMOTE_HIP_DIR}/include/hip_runtime_api_hcc.h" "${HIP_HEADER_DIR}")
      create_link("${REMOTE_HIP_DIR}/include/hip_runtime.h" "${HIP_HEADER_DIR}")
      create_link("${REMOTE_HIP_DIR}/include/hip_runtime_hcc.h" "${HIP_HEADER_DIR}")
      create_link("${REMOTE_HIP_DIR}/include/hip_texture_hcc.h" "${HIP_HEADER_DIR}")
      create_link("${REMOTE_HIP_DIR}/hc/include/grid_launch.h" "${HIP_HEADER_DIR}")
      create_link("${REMOTE_HIP_DIR}/hc/include/hc_am.hpp" "${HIP_HEADER_DIR}")
      create_link("${REMOTE_HIP_DIR}/src/hip_hcc.cpp" "${HIP_LIB_DIR}")
      create_link("${REMOTE_HIP_DIR}/hc/src/hc_am.cpp" "${HIP_LIB_DIR}")
      create_link("${REMOTE_HIP_DIR}/hc/src/grid_launch.cpp" "${HIP_LIB_DIR}")
    endif()
  endif()
endmacro()
