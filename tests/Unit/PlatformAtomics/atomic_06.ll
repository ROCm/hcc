; XFAIL: Linux
; RUN: %hsail %s | tee %t | %FileCheck %s
; ModuleID = 'kernel.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-a0:0:64-s0:64:64-f80:128:128"
target triple = "hsail64-pc-unknown-amdopencl"

define spir_kernel void @ZZN11Concurrency4Test7details8gpu_readIiLi1EEET_RNS_10array_viewIS3_XT0_EEENS_5indexIXT0_EEEENS2_IiLi1EEUlNS7_ILi1EEEE_19__cxxamp_trampolineEPiiiiiiiiSC_iiiiiiiii(i32*, i32, i32, i32, i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
  ret void
}

define spir_kernel void @ZZN11Concurrency4Test7details9gpu_writeIiLi1EEEvRNS_10array_viewIT_XT0_EEENS_5indexIXT0_EEES4_ENS2_IiLi1EEUlNS7_ILi1EEEE_19__cxxamp_trampolineEPiiiiiiiiiii(i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {

test_atomicrmw_sub:
  ;
  ; test atomicrmw sub
  ;

  %sub1 = atomicrmw sub i32* %0, i32 %1 monotonic
;CHECK: atomicnoret_sub_global_rlx_sys_s32 {{.*}}

  %sub2 = atomicrmw sub i32* %0, i32 %1 acquire
;CHECK: atomicnoret_sub_global_scacq_sys_s32 {{.*}}

  %sub3 = atomicrmw sub i32* %0, i32 %1 release
;CHECK: atomicnoret_sub_global_screl_sys_s32 {{.*}}

  %sub4 = atomicrmw sub i32* %0, i32 %1 acq_rel
;CHECK: atomicnoret_sub_global_scar_sys_s32 {{.*}}

  %sub5 = atomicrmw sub i32* %0, i32 %1 seq_cst
;CHECK: atomicnoret_sub_global_scar_sys_s32 {{.*}}

  ret void
}

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!1}
!opencl.kernels = !{!3, !9}

!0 = metadata !{i32 1, i32 2}
!1 = metadata !{}
!2 = metadata !{metadata !"cl_doubles"}
!3 = metadata !{void (i32*, i32, i32, i32, i32, i32, i32, i32, i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @ZZN11Concurrency4Test7details8gpu_readIiLi1EEET_RNS_10array_viewIS3_XT0_EEENS_5indexIXT0_EEEENS2_IiLi1EEUlNS7_ILi1EEEE_19__cxxamp_trampolineEPiiiiiiiiSC_iiiiiiiii, metadata !4, metadata !5, metadata !6, metadata !7, metadata !8}
!4 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!5 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!6 = metadata !{metadata !"kernel_arg_type", metadata !"int*", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int*", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int"}
!7 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!8 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!9 = metadata !{void (i32*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @ZZN11Concurrency4Test7details9gpu_writeIiLi1EEEvRNS_10array_viewIT_XT0_EEENS_5indexIXT0_EEES4_ENS2_IiLi1EEUlNS7_ILi1EEEE_19__cxxamp_trampolineEPiiiiiiiiiii, metadata !10, metadata !11, metadata !12, metadata !13, metadata !14}
!10 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!11 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!12 = metadata !{metadata !"kernel_arg_type", metadata !"int*", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int", metadata !"int"}
!13 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!14 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!15 = metadata !{metadata !"int", metadata !16}
!16 = metadata !{metadata !"omnipotent char", metadata !17}
!17 = metadata !{metadata !"Simple C/C++ TBAA"}
