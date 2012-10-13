; RUN: %llc -march=c < %s  > %t.c
; RUN: gcc -D__ATTRIBUTE_WEAK__= -D__local=static %t.c -c
; ModuleID = 'BitonicSort.cpp'

@_ZZ16transpose_kernelIiEvRN11Concurrency5arrayIT_Li1EEES4_jjNS0_11tiled_indexILi16ELi16ELi0EEEE21transpose_shared_data = linkonce_odr addrspace(3) global [16 x [16 x i32]] zeroinitializer, align 4

define linkonce_odr void @foo() align 2 {
entry:
  %call3.i = call i32 @get_local_id(i32 1) nounwind readonly
  %call4.i = call i32 @get_local_id(i32 0) nounwind readonly
  %arrayidx13.i.i = getelementptr inbounds [16 x [16 x i32]] addrspace(3)* @_ZZ16transpose_kernelIiEvRN11Concurrency5arrayIT_Li1EEES4_jjNS0_11tiled_indexILi16ELi16ELi0EEEE21transpose_shared_data, i32 0, i32 %call3.i, i32 %call4.i
  ret void
}


declare i32 @get_local_id(i32) nounwind readonly
declare i32 @get_global_id(i32) nounwind readonly


