####################
# builtin functions bitcode (HSA)
####################
install(FILES hsa_math.bc DESTINATION lib)
install(FILES builtins-hsail.opt.bc DESTINATION lib)
install(FILES hsail-amdgpu-wrapper.ll DESTINATION lib)
# Copy the file to directory matching the install directory
file(COPY hsa_math.bc DESTINATION "${PROJECT_BINARY_DIR}/lib")
file(COPY builtins-hsail.opt.bc DESTINATION "${PROJECT_BINARY_DIR}/lib")
file(COPY hsail-amdgpu-wrapper.ll DESTINATION "${PROJECT_BINARY_DIR}/lib")

####################
# Dynamic memory allocation HSAIL logic (HSA)
####################
install(FILES hsa_builtins.hsail DESTINATION lib)
# Copy the file to directory matching the install directory
file(COPY hsa_builtins.hsail DESTINATION "${PROJECT_BINARY_DIR}/lib")
