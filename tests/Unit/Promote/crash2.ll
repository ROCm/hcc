; RUN: %spirify %s > %t
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

%"class.Concurrency::index.4.164.324.344" = type { %"struct.Concurrency::index_impl.2.163.323.343" }
%"struct.Concurrency::index_impl.2.163.323.343" = type { %"class.Concurrency::__index_leaf.0.162.322.342" }
%"class.Concurrency::__index_leaf.0.162.322.342" = type { i32, i32 }
%"class.Concurrency::index.0.7.167.327.347" = type { %"struct.Concurrency::index_impl.1.6.166.326.346" }
%"struct.Concurrency::index_impl.1.6.166.326.346" = type { %"class.Concurrency::__index_leaf.0.162.322.342", %"class.Concurrency::__index_leaf.2.5.165.325.345" }
%"class.Concurrency::__index_leaf.2.5.165.325.345" = type { i32, i32 }
%"class.Concurrency::index.3.10.170.330.350" = type { %"struct.Concurrency::index_impl.4.9.169.329.349" }
%"struct.Concurrency::index_impl.4.9.169.329.349" = type { %"class.Concurrency::__index_leaf.0.162.322.342", %"class.Concurrency::__index_leaf.2.5.165.325.345", %"class.Concurrency::__index_leaf.5.8.168.328.348" }
%"class.Concurrency::__index_leaf.5.8.168.328.348" = type { i32, i32 }
%"class.Concurrency::tiled_extent.12.172.332.352" = type { %"class.Concurrency::extent.11.171.331.351" }
%"class.Concurrency::extent.11.171.331.351" = type { %"struct.Concurrency::index_impl.2.163.323.343" }
%class.anon.15.175.335.355 = type { %"class.Concurrency::array.14.174.334.354"*, i32 }
%"class.Concurrency::array.14.174.334.354" = type { %"class.Concurrency::extent.11.171.331.351", %"class.Concurrency::_data.13.173.333.353", i32 }
%"class.Concurrency::_data.13.173.333.353" = type { i32* }
%"class.Concurrency::tiled_extent.7.17.177.337.357" = type { %"class.Concurrency::extent.8.16.176.336.356" }
%"class.Concurrency::extent.8.16.176.336.356" = type { %"struct.Concurrency::index_impl.1.6.166.326.346" }
%class.anon.6.18.178.338.358 = type { %"class.Concurrency::array.14.174.334.354"*, %"class.Concurrency::array.14.174.334.354"*, i32, i32 }
%class.anon.10.19.179.339.359 = type { %"class.Concurrency::array.14.174.334.354"*, i32, i32 }
%class.anon.11.20.180.340.360 = type { %"class.Concurrency::array.14.174.334.354"*, %"class.Concurrency::array.14.174.334.354"*, i32, i32 }
%class.anon.12.21.181.341.361 = type { %"class.Concurrency::array.14.174.334.354"*, i32, i32 }

@.str = private unnamed_addr constant [22 x i8] c"__cxxamp_opencl_index\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [55 x i8] c"/home/ijsung/Development/cppamp/test/src/include/amp.h\00", section "llvm.metadata"
@.str2 = private unnamed_addr constant [20 x i8] c"__cxxamp_trampoline\00", section "llvm.metadata"
@.str3 = private unnamed_addr constant [83 x i8] c"/home/ijsung/Development/cppamp/test/src/tests/Samples/BitonicSort/BitonicSort.cpp\00", section "llvm.metadata"
@_ZZ19bitonic_sort_kernelIiEvRN11Concurrency5arrayIT_Li1EEEjjNS0_11tiled_indexILi512ELi0ELi0EEEE7sh_data = linkonce_odr global [512 x i32] zeroinitializer, section "clamp_opencl_local", align 4
@llvm.global.annotations = appending global [8 x { i8*, i8*, i8*, i32 }] [{ i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Concurrency::index.4.164.324.344"*)* @_ZN11Concurrency5indexILi1EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([55 x i8]* @.str1, i32 0, i32 0), i32 552 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Concurrency::index.0.7.167.327.347"*)* @_ZN11Concurrency5indexILi2EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([55 x i8]* @.str1, i32 0, i32 0), i32 552 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Concurrency::index.3.10.170.330.350"*)* @_ZN11Concurrency5indexILi3EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([55 x i8]* @.str1, i32 0, i32 0), i32 552 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE1_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj to i8*), i8* getelementptr inbounds ([20 x i8]* @.str2, i32 0, i32 0), i8* getelementptr inbounds ([83 x i8]* @.str3, i32 0, i32 0), i32 167 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj to i8*), i8* getelementptr inbounds ([20 x i8]* @.str2, i32 0, i32 0), i8* getelementptr inbounds ([83 x i8]* @.str3, i32 0, i32 0), i32 161 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj to i8*), i8* getelementptr inbounds ([20 x i8]* @.str2, i32 0, i32 0), i8* getelementptr inbounds ([83 x i8]* @.str3, i32 0, i32 0), i32 149 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj to i8*), i8* getelementptr inbounds ([20 x i8]* @.str2, i32 0, i32 0), i8* getelementptr inbounds ([83 x i8]* @.str3, i32 0, i32 0), i32 142 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (i32, i32, i32*, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEj to i8*), i8* getelementptr inbounds ([20 x i8]* @.str2, i32 0, i32 0), i8* getelementptr inbounds ([83 x i8]* @.str3, i32 0, i32 0), i32 119 }], section "llvm.metadata"
@llvm.used = appending global [5 x i8*] [i8* bitcast (void (%"class.Concurrency::tiled_extent.12.172.332.352"*, %class.anon.15.175.335.355*)* @_ZN11Concurrency17parallel_for_eachILi512EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi512ELi0ELi0EEEE_EEvNS_12tiled_extentIXT_ELi0ELi0EEERKT0_ to i8*), i8* bitcast (void (%"class.Concurrency::tiled_extent.7.17.177.337.357"*, %class.anon.6.18.178.338.358*)* @_ZN11Concurrency17parallel_for_eachILi16ELi16EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi16ELi16ELi0EEEE_EEvNS_12tiled_extentIXT_EXT0_ELi0EEERKT1_ to i8*), i8* bitcast (void (%"class.Concurrency::tiled_extent.12.172.332.352"*, %class.anon.10.19.179.339.359*)* @_ZN11Concurrency17parallel_for_eachILi512EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi512ELi0ELi0EEEE0_EEvNS_12tiled_extentIXT_ELi0ELi0EEERKT0_ to i8*), i8* bitcast (void (%"class.Concurrency::tiled_extent.7.17.177.337.357"*, %class.anon.11.20.180.340.360*)* @_ZN11Concurrency17parallel_for_eachILi16ELi16EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi16ELi16ELi0EEEE0_EEvNS_12tiled_extentIXT_EXT0_ELi0EEERKT1_ to i8*), i8* bitcast (void (%"class.Concurrency::tiled_extent.12.172.332.352"*, %class.anon.12.21.181.341.361*)* @_ZN11Concurrency17parallel_for_eachILi512EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi512ELi0ELi0EEEE1_EEvNS_12tiled_extentIXT_ELi0ELi0EEERKT0_ to i8*)], section "llvm.metadata"

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EE21__cxxamp_opencl_indexEv(%"class.Concurrency::index.4.164.324.344"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EE21__cxxamp_opencl_indexEv(%"class.Concurrency::index.0.7.167.327.347"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EE21__cxxamp_opencl_indexEv(%"class.Concurrency::index.3.10.170.330.350"* nocapture) #0 align 2

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi512EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi512ELi0ELi0EEEE_EEvNS_12tiled_extentIXT_ELi0ELi0EEERKT0_(%"class.Concurrency::tiled_extent.12.172.332.352"* nocapture, %class.anon.15.175.335.355* nocapture) #1

declare void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEj(i32, i32, i32*, i32, i32) #2 align 2

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi16ELi16EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi16ELi16ELi0EEEE_EEvNS_12tiled_extentIXT_EXT0_ELi0EEERKT1_(%"class.Concurrency::tiled_extent.7.17.177.337.357"* nocapture, %class.anon.6.18.178.338.358* nocapture) #1

declare void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj(i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32) #2 align 2

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi512EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi512ELi0ELi0EEEE0_EEvNS_12tiled_extentIXT_ELi0ELi0EEERKT0_(%"class.Concurrency::tiled_extent.12.172.332.352"* nocapture, %class.anon.10.19.179.339.359* nocapture) #1

define void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj(i32, i32, i32*, i32, i32, i32) #2 align 2 {
entry:
  %arrayidx.i.i = getelementptr inbounds [512 x i32]* @_ZZ19bitonic_sort_kernelIiEvRN11Concurrency5arrayIT_Li1EEEjjNS0_11tiled_indexILi512ELi0ELi0EEEE7sh_data, i32 0, i32 undef
  br i1 undef, label %_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENKS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE0_clESA_.exit, label %for.body.lr.ph.i.i

for.body.lr.ph.i.i:                               ; preds = %entry
  br i1 undef, label %cond.end.i.i, label %cond.true.i.i

cond.true.i.i:                                    ; preds = %for.body.lr.ph.i.i
  %arrayidx9.i.i = getelementptr inbounds [512 x i32]* @_ZZ19bitonic_sort_kernelIiEvRN11Concurrency5arrayIT_Li1EEEjjNS0_11tiled_indexILi512ELi0ELi0EEEE7sh_data, i32 0, i32 undef
  br label %cond.end.i.i

cond.end.i.i:                                     ; preds = %cond.true.i.i, %for.body.lr.ph.i.i
  %cond.in.i.i = phi i32* [ %arrayidx9.i.i, %cond.true.i.i ], [ %arrayidx.i.i, %for.body.lr.ph.i.i ]
  unreachable

_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENKS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE0_clESA_.exit: ; preds = %entry
  ret void
}

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi16ELi16EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi16ELi16ELi0EEEE0_EEvNS_12tiled_extentIXT_EXT0_ELi0EEERKT1_(%"class.Concurrency::tiled_extent.7.17.177.337.357"* nocapture, %class.anon.11.20.180.340.360* nocapture) #1

declare void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj(i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32) #2 align 2

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi512EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi512ELi0ELi0EEEE1_EEvNS_12tiled_extentIXT_ELi0ELi0EEERKT0_(%"class.Concurrency::tiled_extent.12.172.332.352"* nocapture, %class.anon.12.21.181.341.361* nocapture) #1

declare void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE1_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj(i32, i32, i32*, i32, i32, i32) #2 align 2

attributes #0 = { alwaysinline nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!hcc.kernels = !{!0, !1, !2, !3, !4}

!0 = metadata !{void (i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE1_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj}
!1 = metadata !{void (i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj}
!2 = metadata !{void (i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj}
!3 = metadata !{void (i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj}
!4 = metadata !{void (i32, i32, i32*, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi512ELi0ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEj}
