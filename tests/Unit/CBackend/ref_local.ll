; RUN: %llc -march=c < %s > %t.c
; RUN: gcc -D__local= -c %t.c
; ModuleID = 'binomialoptions.cpp'

@call_a = internal addrspace(3) unnamed_addr global [17 x float] zeroinitializer, align 4

define internal void @bar() align 2 {
entry:
  %0 = load float addrspace(3)* getelementptr inbounds ([17 x float] addrspace(3)* @call_a, i32 0, i32 0), align 4
  ret void
}

