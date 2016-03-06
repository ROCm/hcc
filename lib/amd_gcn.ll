




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




define linkonce_odr spir_func i32 @amdgcn_row_rshift(i32 %data, i32 %delta) #1 {
  switch i32 %delta, label %31 [
    i32 1, label %1
    i32 2, label %3
    i32 3, label %5
    i32 4, label %7
    i32 5, label %9
    i32 6, label %11
    i32 7, label %13
    i32 8, label %15
    i32 9, label %17
    i32 10, label %19
    i32 11, label %21
    i32 12, label %23
    i32 13, label %25
    i32 14, label %27
    i32 15, label %29
  ]

; <label>:1:                                              ; preds = %0                     
  %2 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 273, i1 0, i32 15, i32 15)
  ret i32 %2

; <label>:3:                                              ; preds = %0                    
  %4 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 274, i1 0, i32 15, i32 15)
  ret i32 %4

; <label>:5:                                              ; preds = %0                     
  %6 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 275, i1 0, i32 15, i32 15)
  ret i32 %6

; <label>:7:                                              ; preds = %0                     
  %8 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 276, i1 0, i32 15, i32 15)
  ret i32 %8

; <label>:9:                                              ; preds = %0                     
  %10 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 277, i1 0, i32 15, i32 15)
  ret i32 %10

; <label>:11:                                              ; preds = %0                     
  %12 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 278, i1 0, i32 15, i32 15)
  ret i32 %12

; <label>:13:                                              ; preds = %0                     
  %14 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 279, i1 0, i32 15, i32 15)
  ret i32 %14

; <label>:15:                                              ; preds = %0                     
  %16 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 280, i1 0, i32 15, i32 15)
  ret i32 %16

; <label>:17:                                              ; preds = %0                     
  %18 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 281, i1 0, i32 15, i32 15)
  ret i32 %18

; <label>:19:                                              ; preds = %0                     
  %20 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 282, i1 0, i32 15, i32 15)
  ret i32 %20

; <label>:21:                                              ; preds = %0                     
  %22 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 283, i1 0, i32 15, i32 15)
  ret i32 %22

; <label>:23:                                              ; preds = %0                     
  %24 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 284, i1 0, i32 15, i32 15)
  ret i32 %24

; <label>:25:                                              ; preds = %0                     
  %26 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 285, i1 0, i32 15, i32 15)
  ret i32 %26

; <label>:27:                                              ; preds = %0                     
  %28 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 286, i1 0, i32 15, i32 15)
  ret i32 %28

; <label>:29:                                              ; preds = %0                     
  %30 = call i32 @llvm.amdgcn.mov.dpp.i32(i32 %data, i32 287, i1 0, i32 15, i32 15)
  ret i32 %30

; <label>:31:
  ret i32 %data
}

define linkonce_odr spir_func i32 @amdgcn_ds_permute(i32 %index, i32 %src) #1  {
  %call = call i32 @llvm.amdgcn.ds.permute(i32 %index, i32 %src)
  ret i32 %call
}

define linkonce_odr spir_func i32 @amdgcn_ds_bpermute(i32 %index, i32 %src) #1  {
  %call = call i32 @llvm.amdgcn.ds.bpermute(i32 %index, i32 %src)
  ret i32 %call
}


;llvm.amdgcn.mov.dpp.i32 <src> <dpp_ctrl> <bound_ctrl> <bank_mask> <row_mask>
declare i32 @llvm.amdgcn.mov.dpp.i32(i32, i32, i1, i32, i32) #0

;llvm.amdgcn.ds.permute <index> <src>
declare i32 @llvm.amdgcn.ds.permute(i32, i32) #0

;llvm.amdgcn.ds.bpermute <index> <src>
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #0

attributes #0 = { nounwind readnone convergent }
attributes #1 = { alwaysinline nounwind readnone convergent }

