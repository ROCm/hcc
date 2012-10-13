; RUN: %llc -march=c < %s > %t.c
; RUN: gcc -c %t.c
; ModuleID = 'binomialoptions.cpp'

%"class.Concurrency::index" = type { [1 x i32] }
%"class.Concurrency::tiled_index" = type { %"class.Concurrency::index", %"class.Concurrency::index", %"class.Concurrency::index", %"class.Concurrency::index" }

define void @foo(%"class.Concurrency::tiled_index"* nocapture %tidx) nounwind {
entry:
  %bar = getelementptr inbounds %"class.Concurrency::tiled_index"* %tidx, i32 0, i32 1, i32 0, i32 0
  ret void
}

!1 = metadata !{metadata !"int", metadata !2}
!2 = metadata !{metadata !"omnipotent char", metadata !3}
!3 = metadata !{metadata !"Simple C/C++ TBAA"}
!4 = metadata !{metadata !"any pointer", metadata !2}
!5 = metadata !{metadata !"float", metadata !2}
