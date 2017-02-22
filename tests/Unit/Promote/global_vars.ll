; RUN: %opt -load %llvm_libs_dir/LLVMPromote.so -load %llvm_libs_dir/LLVMEraseNonkernel.so -load %llvm_libs_dir/LLVMTileUniform.so -load %llvm_libs_dir/LLVMRemoveSpecialSection.so  -erase-nonkernels -promote-globals -tile-uniform -dce -globaldce -remove-special-section -S < %s | %FileCheck %s
; ModuleID = 'global_vars'
source_filename = "llvm-link"
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa-hcc"
$G1 = comdat any
$G2 = comdat any
@G1 = linkonce_odr local_unnamed_addr global [9 x float] zeroinitializer, section "clamp_opencl_local", comdat, align 4
@G2 = linkonce_odr local_unnamed_addr global [9 x float] zeroinitializer, section "clamp_opencl_local", comdat, align 4

define amdgpu_kernel void @foo(i32 *%out) #0 align 2 {
  %foo.1 = getelementptr inbounds [9 x float], [9 x float]* @G1, i32 0, i32 3
; CHECK: %foo.1 = getelementptr inbounds [9 x float], [9 x float] addrspace(3)* @foo.G1
  %foo.2 = bitcast float* %foo.1 to i32*
; CHECK: bitcast float addrspace(3)* %foo.1 to i32 addrspace(3)*
  %foo.3 = load i32, i32 *%foo.2, align 4
; CHECK: %foo.3 = load i32, i32 addrspace(3)*
  store i32 %foo.3, i32 *%out, align 4
  ret void
}

define amdgpu_kernel void @bar(i32 *%out) #0 align 2 {
  %bar.1 = getelementptr inbounds [9 x float], [9 x float]* @G2, i32 0, i32 2
; CHECK: %bar.1 = getelementptr inbounds [9 x float], [9 x float] addrspace(3)* @bar.G2
  %bar.2 = bitcast float* %bar.1 to i32*
; CHECK: bitcast float addrspace(3)* %bar.1 to i32 addrspace(3)*
  %bar.3 = load i32, i32 *%bar.2, align 4
; CHECK: %bar.3 = load i32, i32 addrspace(3)*
  store i32 %bar.3, i32 *%out, align 4
  ret void
}

define amdgpu_kernel void @foo2(i32 *%out) #0 align 2 {
  %foo2.1 = getelementptr inbounds [9 x float], [9 x float]* @G1, i32 0, i32 3
; CHECK: %foo2.1 = getelementptr inbounds [9 x float], [9 x float] addrspace(3)* @foo2.G1
  %foo2.2 = bitcast float* %foo2.1 to i32*
; CHECK: bitcast float addrspace(3)* %foo2.1 to i32 addrspace(3)*
  %foo2.3 = load i32, i32 *%foo2.2, align 4
; CHECK: %foo2.3 = load i32, i32 addrspace(3)*
  store i32 %foo2.3, i32 *%out, align 4
  ret void
}

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+fp64-denormals,-fp32-denormals" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.ocl.version = !{!0}
!llvm.ident = !{!1, !1, !1}
!llvm.module.flags = !{!2}
!hcc.kernels = !{!3,!4,!5}
!0 = !{i32 0, i32 0}
!1 = !{!"HCC clang version 4.0.0 (based on HCC 1.0.16405-ec4fcca-12f0f67-6c755a0 )"}
!2 = !{i32 1, !"PIC Level", i32 2}
!3 = !{void (i32*)* @foo}
!4 = !{void (i32*)* @bar}
!5 = !{void (i32*)* @foo2}
