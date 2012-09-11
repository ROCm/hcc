; RUN: %llc -march=c < %s | %FileCheck %s

; ModuleID = 'struct_in_struct.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.std_info = type { [7 x i8], [9 x i8], [3 x i8], %struct.date }
%struct.date = type { i32, i32, i32 }

@student = global %struct.std_info { [7 x i8] c"000102\00", [9 x i8] c"sss\00\00\00\00\00\00", [3 x i8] c"nv\00", %struct.date { i32 1980, i32 9, i32 20 } }, align 4
@.str = private unnamed_addr constant [8 x i8] c"No: %s\0A\00", align 1
@.str1 = private unnamed_addr constant [10 x i8] c"Name: %s\0A\00", align 1
@.str2 = private unnamed_addr constant [9 x i8] c"Sex: %s\0A\00", align 1
@.str3 = private unnamed_addr constant [20 x i8] c"Birthday: %d-%d-%d\0A\00", align 1

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %p_std = alloca %struct.std_info*, align 8
  store i32 0, i32* %retval
  store %struct.std_info* @student, %struct.std_info** %p_std, align 8
  %0 = load %struct.std_info** %p_std, align 8
  %no = getelementptr inbounds %struct.std_info* %0, i32 0, i32 0
  %arraydecay = getelementptr inbounds [7 x i8]* %no, i32 0, i32 0
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([8 x i8]* @.str, i32 0, i32 0), i8* %arraydecay)
  %1 = load %struct.std_info** %p_std, align 8
  %name = getelementptr inbounds %struct.std_info* %1, i32 0, i32 1
  %arraydecay1 = getelementptr inbounds [9 x i8]* %name, i32 0, i32 0
  %call2 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str1, i32 0, i32 0), i8* %arraydecay1)
  %2 = load %struct.std_info** %p_std, align 8
  %sex = getelementptr inbounds %struct.std_info* %2, i32 0, i32 2
  %arraydecay3 = getelementptr inbounds [3 x i8]* %sex, i32 0, i32 0
  %call4 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([9 x i8]* @.str2, i32 0, i32 0), i8* %arraydecay3)
  %3 = load %struct.std_info** %p_std, align 8
  %birthday = getelementptr inbounds %struct.std_info* %3, i32 0, i32 3
  %year = getelementptr inbounds %struct.date* %birthday, i32 0, i32 0
  %4 = load i32* %year, align 4
  %5 = load %struct.std_info** %p_std, align 8
  %birthday5 = getelementptr inbounds %struct.std_info* %5, i32 0, i32 3
  %month = getelementptr inbounds %struct.date* %birthday5, i32 0, i32 1
  %6 = load i32* %month, align 4
  %7 = load %struct.std_info** %p_std, align 8
  %birthday6 = getelementptr inbounds %struct.std_info* %7, i32 0, i32 3
  %day = getelementptr inbounds %struct.date* %birthday6, i32 0, i32 2
  %8 = load i32* %day, align 4
  %call7 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([20 x i8]* @.str3, i32 0, i32 0), i32 %4, i32 %6, i32 %8)
  ret i32 0
}

declare i32 @printf(i8*, ...)

;CHECK: main
