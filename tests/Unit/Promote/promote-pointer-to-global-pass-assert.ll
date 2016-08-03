; RUN: %spirify %s > %t
; ModuleID = '/tmp/reduced.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.Kalmar::index" = type { %"struct.Kalmar::index_impl" }
%"struct.Kalmar::index_impl" = type { %"class.Kalmar::__index_leaf" }
%"class.Kalmar::__index_leaf" = type { i32, i32 }
%"class.Kalmar::index.0" = type { %"struct.Kalmar::index_impl.1" }
%"struct.Kalmar::index_impl.1" = type { %"class.Kalmar::__index_leaf", %"class.Kalmar::__index_leaf.2" }
%"class.Kalmar::__index_leaf.2" = type { i32, i32 }
%"class.Kalmar::index.3" = type { %"struct.Kalmar::index_impl.4" }
%"struct.Kalmar::index_impl.4" = type { %"class.Kalmar::__index_leaf", %"class.Kalmar::__index_leaf.2", %"class.Kalmar::__index_leaf.5" }
%"class.Kalmar::__index_leaf.5" = type { i32, i32 }
%"class.Concurrency::accelerator_view" = type { %"class.std::__1::shared_ptr" }
%"class.std::__1::shared_ptr" = type { %"class.Kalmar::KalmarQueue"*, %"class.std::__1::__shared_weak_count"* }
%"class.Kalmar::KalmarQueue" = type { i32 (...)**, %"class.Kalmar::KalmarDevice"*, i32 }
%"class.Kalmar::KalmarDevice" = type { i32 (...)**, i32, %"class.std::__1::shared_ptr", %"struct.std::__1::once_flag" }
%"struct.std::__1::once_flag" = type { i64 }
%"class.std::__1::__shared_weak_count" = type { %"class.std::__1::__shared_count", i64 }
%"class.std::__1::__shared_count" = type { i32 (...)**, i64 }
%"class.Concurrency::tiled_extent" = type { %"class.Concurrency::extent" }
%"class.Concurrency::extent" = type { %"struct.Kalmar::index_impl" }
%class.anon = type { i32, i32, %struct.SimFlatSt*, i32, double, %struct.EamPotentialSt* }
%struct.SimFlatSt = type { i32, i32, double, %struct.DomainSt*, %struct.LinkCellSt*, %struct.AtomsSt*, %struct.SpeciesDataSt*, double, double, %struct.BasePotentialSt*, %struct.HaloExchangeSt* }
%struct.DomainSt = type { [3 x i32], [3 x i32], [3 x double], [3 x double], [3 x double], [3 x double], [3 x double], [3 x double] }
%struct.LinkCellSt = type { [3 x i32], i32, i32, i32, [3 x double], [3 x double], [3 x double], [3 x double], i32*, i32** }
%struct.AtomsSt = type { i32, i32, i32*, i32*, [3 x double]*, [3 x double]*, [3 x double]*, double* }
%struct.SpeciesDataSt = type { [3 x i8], i32, double }
%struct.BasePotentialSt = type { double, double, double, [8 x i8], [3 x i8], i32, i32 (%struct.SimFlatSt*)*, void (%struct._IO_FILE*, %struct.BasePotentialSt*)*, void (%struct.BasePotentialSt**)* }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.HaloExchangeSt = type { [6 x i32], i32, i32 (i8*, i8*, i32, i8*)*, void (i8*, i8*, i32, i32, i8*)*, void (i8*)*, i8* }
%struct.EamPotentialSt = type { double, double, double, [8 x i8], [3 x i8], i32, i32 (%struct.SimFlatSt*)*, void (%struct._IO_FILE*, %struct.BasePotentialSt*)*, void (%struct.BasePotentialSt**)*, %struct.InterpolationObjectSt*, %struct.InterpolationObjectSt*, %struct.InterpolationObjectSt*, double*, double*, %struct.HaloExchangeSt*, %struct.ForceExchangeDataSt* }
%struct.InterpolationObjectSt = type { i32, double, double, double* }
%struct.ForceExchangeDataSt = type { double*, %struct.LinkCellSt* }
%class.anon.13 = type { i32, i32, %struct.SimFlatSt*, %struct.EamPotentialSt* }
%class.anon.14 = type { i32, i32, %struct.SimFlatSt*, i32, double, %struct.EamPotentialSt* }
%struct.SimFlatSt.34 = type { i32, i32, double, %struct.DomainSt*, %struct.LinkCellSt*, %struct.AtomsSt*, %struct.SpeciesDataSt*, double, double, %struct.BasePotentialSt.35*, %struct.HaloExchangeSt* }
%struct.BasePotentialSt.35 = type { double, double, double, [8 x i8], [3 x i8], i32, i32 (%struct.SimFlatSt.34*)*, void (%struct._IO_FILE*, %struct.BasePotentialSt.35*)*, {}* }
%class.anon.44 = type { %struct.SimFlatSt.34* }
%class.anon.13.46 = type { %struct.SimFlatSt.34*, i32, double, double, double, double }

@.str = private unnamed_addr constant [22 x i8] c"__cxxamp_opencl_index\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [62 x i8] c"/home/marsenau/src/cppamp-driver-ng-35/include/kalmar_index.h\00", section "llvm.metadata"
@.str7 = private unnamed_addr constant [20 x i8] c"__cxxamp_trampoline\00", section "llvm.metadata"
@.str8 = private unnamed_addr constant [8 x i8] c"eam.cpp\00", section "llvm.metadata"
@llvm.global.annotations = appending global [14 x { i8*, i8*, i8*, i32 }] [{ i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index"*)* @_ZN6Kalmar5indexILi1EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([62 x i8]* @.str1, i32 0, i32 0), i32 318 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.0"*)* @_ZN6Kalmar5indexILi2EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([62 x i8]* @.str1, i32 0, i32 0), i32 318 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.3"*)* @_ZN6Kalmar5indexILi3EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([62 x i8]* @.str1, i32 0, i32 0), i32 318 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (i32, i32, %struct.SimFlatSt*, i32, double, %struct.EamPotentialSt*)* @"_ZZL8eamForceP9SimFlatStEN3$_219__cxxamp_trampolineEiiS0_idP14EamPotentialSt" to i8*), i8* getelementptr inbounds ([20 x i8]* @.str7, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str8, i32 0, i32 0), i32 402 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (i32, i32, %struct.SimFlatSt*, %struct.EamPotentialSt*)* @"_ZZL8eamForceP9SimFlatStEN3$_119__cxxamp_trampolineEiiS0_P14EamPotentialSt" to i8*), i8* getelementptr inbounds ([20 x i8]* @.str7, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str8, i32 0, i32 0), i32 359 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (i32, i32, %struct.SimFlatSt*, i32, double, %struct.EamPotentialSt*)* @"_ZZL8eamForceP9SimFlatStEN3$_019__cxxamp_trampolineEiiS0_idP14EamPotentialSt" to i8*), i8* getelementptr inbounds ([20 x i8]* @.str7, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8]* @.str8, i32 0, i32 0), i32 289 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index"*)* @_ZN6Kalmar5indexILi1EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str41, i32 0, i32 0), i8* getelementptr inbounds ([62 x i8]* @.str142, i32 0, i32 0), i32 318 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.0"*)* @_ZN6Kalmar5indexILi2EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str41, i32 0, i32 0), i8* getelementptr inbounds ([62 x i8]* @.str142, i32 0, i32 0), i32 318 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.3"*)* @_ZN6Kalmar5indexILi3EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str41, i32 0, i32 0), i8* getelementptr inbounds ([62 x i8]* @.str142, i32 0, i32 0), i32 318 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%struct.SimFlatSt.34*, i32, double, double, double, double)* @"_ZZL7ljForceP9SimFlatStEN3$_119__cxxamp_trampolineES0_idddd" to i8*), i8* getelementptr inbounds ([20 x i8]* @.str446, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str547, i32 0, i32 0), i32 285 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%struct.SimFlatSt.34*)* @"_ZZL7ljForceP9SimFlatStEN3$_019__cxxamp_trampolineES0_" to i8*), i8* getelementptr inbounds ([20 x i8]* @.str446, i32 0, i32 0), i8* getelementptr inbounds ([12 x i8]* @.str547, i32 0, i32 0), i32 174 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index"*)* @_ZN6Kalmar5indexILi1EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str213, i32 0, i32 0), i8* getelementptr inbounds ([62 x i8]* @.str1214, i32 0, i32 0), i32 318 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.0"*)* @_ZN6Kalmar5indexILi2EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str213, i32 0, i32 0), i8* getelementptr inbounds ([62 x i8]* @.str1214, i32 0, i32 0), i32 318 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.3"*)* @_ZN6Kalmar5indexILi3EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str213, i32 0, i32 0), i8* getelementptr inbounds ([62 x i8]* @.str1214, i32 0, i32 0), i32 318 }], section "llvm.metadata"
@.str41 = private unnamed_addr constant [22 x i8] c"__cxxamp_opencl_index\00", section "llvm.metadata"
@.str142 = private unnamed_addr constant [62 x i8] c"/home/marsenau/src/cppamp-driver-ng-35/include/kalmar_index.h\00", section "llvm.metadata"
@.str446 = private unnamed_addr constant [20 x i8] c"__cxxamp_trampoline\00", section "llvm.metadata"
@.str547 = private unnamed_addr constant [12 x i8] c"ljForce.cpp\00", section "llvm.metadata"
@.str213 = private unnamed_addr constant [22 x i8] c"__cxxamp_opencl_index\00", section "llvm.metadata"
@.str1214 = private unnamed_addr constant [62 x i8] c"/home/marsenau/src/cppamp-driver-ng-35/include/kalmar_index.h\00", section "llvm.metadata"
@llvm.used = appending global [5 x i8*] [i8* bitcast (void (%"class.Concurrency::accelerator_view"*, %"class.Concurrency::tiled_extent"*, %class.anon*)* @"_ZN11Concurrency17parallel_for_eachILi64EZL8eamForceP9SimFlatStE3$_0EEvRKNS_16accelerator_viewENS_12tiled_extentIXT_ELi0ELi0EEERKT0_" to i8*), i8* bitcast (void (%"class.Concurrency::accelerator_view"*, %"class.Concurrency::tiled_extent"*, %class.anon.13*)* @"_ZN11Concurrency17parallel_for_eachILi64EZL8eamForceP9SimFlatStE3$_1EEvRKNS_16accelerator_viewENS_12tiled_extentIXT_ELi0ELi0EEERKT0_" to i8*), i8* bitcast (void (%"class.Concurrency::accelerator_view"*, %"class.Concurrency::tiled_extent"*, %class.anon.14*)* @"_ZN11Concurrency17parallel_for_eachILi64EZL8eamForceP9SimFlatStE3$_2EEvRKNS_16accelerator_viewENS_12tiled_extentIXT_ELi0ELi0EEERKT0_" to i8*), i8* bitcast (void (%"class.Concurrency::accelerator_view"*, %"class.Concurrency::extent"*, %class.anon.44*)* @"_ZN11Concurrency17parallel_for_eachIZL7ljForceP9SimFlatStE3$_0EEvRKNS_16accelerator_viewENS_6extentILi1EEERKT_" to i8*), i8* bitcast (void (%"class.Concurrency::accelerator_view"*, %"class.Concurrency::tiled_extent"*, %class.anon.13.46*)* @"_ZN11Concurrency17parallel_for_eachILi64EZL7ljForceP9SimFlatStE3$_1EEvRKNS_16accelerator_viewENS_12tiled_extentIXT_ELi0ELi0EEERKT0_" to i8*)], section "llvm.metadata"

; Function Attrs: alwaysinline nounwind readnone uwtable
define internal void @_ZN6Kalmar5indexILi1EE21__cxxamp_opencl_indexEv(%"class.Kalmar::index"* nocapture %this) #0 align 2 {
  ret void
}

; Function Attrs: alwaysinline nounwind readnone uwtable
define internal void @_ZN6Kalmar5indexILi2EE21__cxxamp_opencl_indexEv(%"class.Kalmar::index.0"* nocapture %this) #0 align 2 {
  ret void
}

; Function Attrs: alwaysinline nounwind readnone uwtable
define internal void @_ZN6Kalmar5indexILi3EE21__cxxamp_opencl_indexEv(%"class.Kalmar::index.3"* nocapture %this) #0 align 2 {
entry:
  ret void
}

; Function Attrs: noinline nounwind readnone uwtable
define internal void @"_ZN11Concurrency17parallel_for_eachILi64EZL8eamForceP9SimFlatStE3$_0EEvRKNS_16accelerator_viewENS_12tiled_extentIXT_ELi0ELi0EEERKT0_"(%"class.Concurrency::accelerator_view"* nocapture dereferenceable(16) %av, %"class.Concurrency::tiled_extent"* nocapture %compute_domain, %class.anon* nocapture dereferenceable(40) %f) #1 {
entry:
  ret void
}

; Function Attrs: nounwind readnone uwtable
define internal spir_kernel void @"_ZZL8eamForceP9SimFlatStEN3$_019__cxxamp_trampolineEiiS0_idP14EamPotentialSt"(i32, i32, %struct.SimFlatSt* nocapture readonly, i32, double, %struct.EamPotentialSt* nocapture readonly) #2 align 2 {
entry:
  ret void
}

; Function Attrs: noinline nounwind readnone uwtable
define internal void @"_ZN11Concurrency17parallel_for_eachILi64EZL8eamForceP9SimFlatStE3$_1EEvRKNS_16accelerator_viewENS_12tiled_extentIXT_ELi0ELi0EEERKT0_"(%"class.Concurrency::accelerator_view"* nocapture dereferenceable(16) %av, %"class.Concurrency::tiled_extent"* nocapture %compute_domain, %class.anon.13* nocapture dereferenceable(24) %f) #1 {
entry:
  ret void
}

; Function Attrs: nounwind readnone uwtable
define internal spir_kernel void @"_ZZL8eamForceP9SimFlatStEN3$_119__cxxamp_trampolineEiiS0_P14EamPotentialSt"(i32, i32, %struct.SimFlatSt* nocapture readonly, %struct.EamPotentialSt* nocapture readonly) #2 align 2 {
entry:
  ret void
}

; Function Attrs: noinline nounwind readnone uwtable
define internal void @"_ZN11Concurrency17parallel_for_eachILi64EZL8eamForceP9SimFlatStE3$_2EEvRKNS_16accelerator_viewENS_12tiled_extentIXT_ELi0ELi0EEERKT0_"(%"class.Concurrency::accelerator_view"* nocapture dereferenceable(16) %av, %"class.Concurrency::tiled_extent"* nocapture %compute_domain, %class.anon.14* nocapture dereferenceable(40) %f) #1 {
entry:
  ret void
}

; Function Attrs: nounwind readnone uwtable
define internal spir_kernel void @"_ZZL8eamForceP9SimFlatStEN3$_219__cxxamp_trampolineEiiS0_idP14EamPotentialSt"(i32, i32, %struct.SimFlatSt* nocapture readonly, i32, double, %struct.EamPotentialSt* nocapture readonly) #2 align 2 {
entry:
  ret void
}

; Function Attrs: nounwind readnone uwtable
define internal spir_kernel void @"_ZZL7ljForceP9SimFlatStEN3$_119__cxxamp_trampolineES0_idddd"(%struct.SimFlatSt.34* nocapture readonly, i32, double, double, double, double) #2 align 2 {
entry:
  ret void
}

; Function Attrs: nounwind readnone uwtable
define internal spir_kernel void @"_ZZL7ljForceP9SimFlatStEN3$_019__cxxamp_trampolineES0_"(%struct.SimFlatSt.34* nocapture readonly) #2 align 2 {
  ret void
}

; Function Attrs: noinline nounwind readnone uwtable
define internal void @"_ZN11Concurrency17parallel_for_eachIZL7ljForceP9SimFlatStE3$_0EEvRKNS_16accelerator_viewENS_6extentILi1EEERKT_"(%"class.Concurrency::accelerator_view"* nocapture dereferenceable(16) %av, %"class.Concurrency::extent"* nocapture %compute_domain, %class.anon.44* nocapture dereferenceable(8) %f) #1 {
entry:
  ret void
}

; Function Attrs: noinline nounwind readnone uwtable
define internal void @"_ZN11Concurrency17parallel_for_eachILi64EZL7ljForceP9SimFlatStE3$_1EEvRKNS_16accelerator_viewENS_12tiled_extentIXT_ELi0ELi0EEERKT0_"(%"class.Concurrency::accelerator_view"* nocapture dereferenceable(16) %av, %"class.Concurrency::tiled_extent"* nocapture %compute_domain, %class.anon.13.46* nocapture dereferenceable(48) %f) #1 {
entry:
  ret void
}

attributes #0 = { alwaysinline nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!hcc.kernels = !{!0, !6, !12, !13, !15}
!llvm.ident = !{!21, !21, !21, !21, !21, !21, !21, !21, !21, !21, !21, !21, !21, !21}

!0 = metadata !{void (i32, i32, %struct.SimFlatSt*, i32, double, %struct.EamPotentialSt*)* @"_ZZL8eamForceP9SimFlatStEN3$_219__cxxamp_trampolineEiiS0_idP14EamPotentialSt", metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
!1 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!2 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!3 = metadata !{metadata !"kernel_arg_type", metadata !"int", metadata !"int", metadata !"SimFlat*", metadata !"int", metadata !"real_t", metadata !"EamPotential*"}
!4 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!5 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!6 = metadata !{void (i32, i32, %struct.SimFlatSt*, %struct.EamPotentialSt*)* @"_ZZL8eamForceP9SimFlatStEN3$_119__cxxamp_trampolineEiiS0_P14EamPotentialSt", metadata !7, metadata !8, metadata !9, metadata !10, metadata !11}
!7 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0}
!8 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!9 = metadata !{metadata !"kernel_arg_type", metadata !"int", metadata !"int", metadata !"SimFlat*", metadata !"EamPotential*"}
!10 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !""}
!11 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !""}
!12 = metadata !{void (i32, i32, %struct.SimFlatSt*, i32, double, %struct.EamPotentialSt*)* @"_ZZL8eamForceP9SimFlatStEN3$_019__cxxamp_trampolineEiiS0_idP14EamPotentialSt", metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
!13 = metadata !{void (%struct.SimFlatSt.34*, i32, double, double, double, double)* @"_ZZL7ljForceP9SimFlatStEN3$_119__cxxamp_trampolineES0_idddd", metadata !1, metadata !2, metadata !14, metadata !4, metadata !5}
!14 = metadata !{metadata !"kernel_arg_type", metadata !"SimFlat*", metadata !"int", metadata !"real_t", metadata !"real_t", metadata !"real_t", metadata !"real_t"}
!15 = metadata !{void (%struct.SimFlatSt.34*)* @"_ZZL7ljForceP9SimFlatStEN3$_019__cxxamp_trampolineES0_", metadata !16, metadata !17, metadata !18, metadata !19, metadata !20}
!16 = metadata !{metadata !"kernel_arg_addr_space", i32 0}
!17 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none"}
!18 = metadata !{metadata !"kernel_arg_type", metadata !"SimFlat*"}
!19 = metadata !{metadata !"kernel_arg_type_qual", metadata !""}
!20 = metadata !{metadata !"kernel_arg_name", metadata !""}
!21 = metadata !{metadata !"Kalmar clang version 3.5.0 (tags/RELEASE_350/final) (based on Kalmar 0.6.0-c9137a5-5dcbbec LLVM 3.5.0svn)"}
