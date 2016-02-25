

define linkonce_odr spir_func i32 @amdgcn_wave_rshift_1(i32 %v) #1  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 312, i1 0, i32 15, i32 15)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_rshift_zero_1(i32 %v) #1  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 312, i1 1, i32 15, i32 15)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_rrotate_1(i32 %v) #1  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 316, i1 0, i32 15, i32 15)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_lshift_1(i32 %v) #1  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 304, i1 0, i32 15, i32 15)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_lshift_zero_1(i32 %v) #1  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 304, i1 1, i32 15, i32 15)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_wave_lrotate_1(i32 %v) #1  {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 308, i1 0, i32 15, i32 15)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_row_rshift_1(i32 %v) #1 {
  %call = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %v, i32 273, i1 0, i32 15, i32 15)
  ret i32 %call
}

;llvm.amdgcn.mov.dpp.i32 <src> <dpp_ctrl> <bound_ctrl> <bank_mask> <row_mask>
declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32, i1, i32, i32) #0

attributes #0 = { nounwind readnone convergent }
attributes #1 = { alwaysinline nounwind readnone convergent }

