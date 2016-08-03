; RUN: %spirify %s > %t
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: alwaysinline nounwind uwtable
define void @_ZN6Kalmar5indexILi2EEC1ERKS1_() unnamed_addr #0 align 2 {
entry:
  ret void
}

attributes #0 = { alwaysinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!hcc.kernels = !{!0, !6, !0, !12, !14}
!llvm.ident = !{!20, !20, !20, !20, !20, !20, !20, !20, !20, !20, !20, !20, !20, !20}

!0 = metadata !{null, metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
!1 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!2 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!3 = metadata !{metadata !"kernel_arg_type", metadata !"int", metadata !"int", metadata !"SimFlat*", metadata !"int", metadata !"real_t", metadata !"EamPotential*"}
!4 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!5 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!6 = metadata !{null, metadata !7, metadata !8, metadata !9, metadata !10, metadata !11}
!7 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0}
!8 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!9 = metadata !{metadata !"kernel_arg_type", metadata !"int", metadata !"int", metadata !"SimFlat*", metadata !"EamPotential*"}
!10 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !""}
!11 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !""}
!12 = metadata !{null, metadata !1, metadata !2, metadata !13, metadata !4, metadata !5}
!13 = metadata !{metadata !"kernel_arg_type", metadata !"SimFlat*", metadata !"int", metadata !"real_t", metadata !"real_t", metadata !"real_t", metadata !"real_t"}
!14 = metadata !{null, metadata !15, metadata !16, metadata !17, metadata !18, metadata !19}
!15 = metadata !{metadata !"kernel_arg_addr_space", i32 0}
!16 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none"}
!17 = metadata !{metadata !"kernel_arg_type", metadata !"SimFlat*"}
!18 = metadata !{metadata !"kernel_arg_type_qual", metadata !""}
!19 = metadata !{metadata !"kernel_arg_name", metadata !""}
!20 = metadata !{metadata !"Kalmar clang version 3.5.0 (tags/RELEASE_350/final) (based on Kalmar 0.6.0-c9137a5-5dcbbec LLVM 3.5.0svn)"}
