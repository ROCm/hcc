; RUN: %llc -march=c %s -o %t.c
; RUN: %clang %t.c -o %t
; RUN: %t | %FileCheck %s

; ModuleID = 'global_variables.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.student = type { i8*, i32, i32 }

@.str = private unnamed_addr constant [9 x i8] c"xiaoming\00", align 1
@boy_name = global i8* getelementptr inbounds ([9 x i8]* @.str, i32 0, i32 0), align 8
@boy_weight_sub = global i32 5, align 4
@.str1 = private unnamed_addr constant [32 x i8] c"The student named %s is a man.\0A\00", align 1
@.str2 = private unnamed_addr constant [31 x i8] c"boy's heavy three months is %d\00", align 1

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  %student_boy = alloca %struct.student, align 8
  store i32 0, i32* %retval
  store i32 18, i32* %x, align 4
  %0 = load i8** @boy_name, align 8
  %name = getelementptr inbounds %struct.student* %student_boy, i32 0, i32 0
  store i8* %0, i8** %name, align 8
  %age = getelementptr inbounds %struct.student* %student_boy, i32 0, i32 1
  store i32 22, i32* %age, align 4
  %weight = getelementptr inbounds %struct.student* %student_boy, i32 0, i32 2
  store i32 130, i32* %weight, align 4
  %age1 = getelementptr inbounds %struct.student* %student_boy, i32 0, i32 1
  %1 = load i32* %age1, align 4
  %2 = load i32* %x, align 4
  %cmp = icmp sge i32 %1, %2
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %name2 = getelementptr inbounds %struct.student* %student_boy, i32 0, i32 0
  %3 = load i8** %name2, align 8
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([32 x i8]* @.str1, i32 0, i32 0), i8* %3)
  %weight3 = getelementptr inbounds %struct.student* %student_boy, i32 0, i32 2
  %4 = load i32* %weight3, align 4
  %5 = load i32* @boy_weight_sub, align 4
  %add = add nsw i32 %4, %5
  %call4 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([31 x i8]* @.str2, i32 0, i32 0), i32 %add)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}

declare i32 @printf(i8*, ...)

;CHECK: The student named xiaoming is a man.
