; RUN: %spirify %s | tee %t | %FileCheck %s
; ModuleID = 'dump.input.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::__1::set" = type { %"class.std::__1::__tree" }
%"class.std::__1::__tree" = type { %"class.std::__1::__tree_node"*, %"class.std::__1::__compressed_pair", %"class.std::__1::__compressed_pair.6" }
%"class.std::__1::__tree_node" = type { %"class.std::__1::__tree_node_base.base", %"class.std::__1::basic_string" }
%"class.std::__1::__tree_node_base.base" = type <{ %"class.std::__1::__tree_end_node", %"class.std::__1::__tree_node_base"*, %"class.std::__1::__tree_node_base"*, i8 }>
%"class.std::__1::__tree_end_node" = type { %"class.std::__1::__tree_node_base"* }
%"class.std::__1::__tree_node_base" = type { %"class.std::__1::__tree_end_node", %"class.std::__1::__tree_node_base"*, %"class.std::__1::__tree_node_base"*, i8 }
%"class.std::__1::basic_string" = type { %"class.std::__1::__compressed_pair.8" }
%"class.std::__1::__compressed_pair.8" = type { %"class.std::__1::__libcpp_compressed_pair_imp.9" }
%"class.std::__1::__libcpp_compressed_pair_imp.9" = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep" = type { %union.anon }
%union.anon = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long" = type { i64, i64, i8* }
%"class.std::__1::__compressed_pair" = type { %"class.std::__1::__libcpp_compressed_pair_imp" }
%"class.std::__1::__libcpp_compressed_pair_imp" = type { %"class.std::__1::__tree_end_node" }
%"class.std::__1::__compressed_pair.6" = type { %"class.std::__1::__libcpp_compressed_pair_imp.7" }
%"class.std::__1::__libcpp_compressed_pair_imp.7" = type { i64 }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%"class.Kalmar::index" = type { %"struct.Kalmar::index_impl" }
%"struct.Kalmar::index_impl" = type { %"class.Kalmar::__index_leaf" }
%"class.Kalmar::__index_leaf" = type { i32, i32 }
%"class.Kalmar::index.0" = type { %"struct.Kalmar::index_impl.1" }
%"struct.Kalmar::index_impl.1" = type { %"class.Kalmar::__index_leaf", %"class.Kalmar::__index_leaf.2" }
%"class.Kalmar::__index_leaf.2" = type { i32, i32 }
%"class.Kalmar::index.3" = type { %"struct.Kalmar::index_impl.4" }
%"struct.Kalmar::index_impl.4" = type { %"class.Kalmar::__index_leaf", %"class.Kalmar::__index_leaf.2", %"class.Kalmar::__index_leaf.5" }
%"class.Kalmar::__index_leaf.5" = type { i32, i32 }
%struct.tid_s = type { i32, i32*, i32* }
%"class.hc::accelerator_view" = type { %"class.std::__1::shared_ptr.13" }
%"class.std::__1::shared_ptr.13" = type { %"class.Kalmar::KalmarQueue"*, %"class.std::__1::__shared_weak_count"* }
%"class.Kalmar::KalmarQueue" = type { i32 (...)**, %"class.Kalmar::KalmarDevice"*, i32 }
%"class.Kalmar::KalmarDevice" = type { i32 (...)**, i32, %"class.std::__1::map", %"class.std::__1::mutex" }
%"class.std::__1::map" = type { %"class.std::__1::__tree.14.26" }
%"class.std::__1::__tree.14.26" = type { %"class.std::__1::__tree_node.15"*, %"class.std::__1::__compressed_pair", %"class.std::__1::__compressed_pair.20" }
%"class.std::__1::__tree_node.15" = type { %"class.std::__1::__tree_node_base.base", %"union.std::__1::__value_type" }
%"union.std::__1::__value_type" = type { %"struct.std::__1::pair" }
%"struct.std::__1::pair" = type { %"class.std::__1::__thread_id", %"class.std::__1::shared_ptr.13" }
%"class.std::__1::__thread_id" = type { i64 }
%"class.std::__1::__compressed_pair.20" = type { %"class.std::__1::__libcpp_compressed_pair_imp.21" }
%"class.std::__1::__libcpp_compressed_pair_imp.21" = type { i64 }
%"class.std::__1::mutex" = type { %union.pthread_mutex_t }
%union.pthread_mutex_t = type { %"struct.(anonymous union)::__pthread_mutex_s" }
%"struct.(anonymous union)::__pthread_mutex_s" = type { i32, i32, i32, i32, i32, i16, i16, %struct.__pthread_internal_list }
%struct.__pthread_internal_list = type { %struct.__pthread_internal_list*, %struct.__pthread_internal_list* }
%"class.std::__1::__shared_weak_count" = type { %"class.std::__1::__shared_count", i64 }
%"class.std::__1::__shared_count" = type { i32 (...)**, i64 }
%"class.hc::completion_future" = type { %"class.std::__1::shared_future", %"class.std::__1::thread"*, %"class.std::__1::shared_ptr" }
%"class.std::__1::shared_future" = type { %"class.std::__1::__assoc_sub_state"* }
%"class.std::__1::__assoc_sub_state" = type { %"class.std::__1::__shared_count", %"class.std::exception_ptr", %"class.std::__1::mutex", %"class.std::__1::condition_variable", i32 }
%"class.std::exception_ptr" = type { i8* }
%"class.std::__1::condition_variable" = type { %union.pthread_cond_t }
%union.pthread_cond_t = type { %struct.anon }
%struct.anon = type { i32, i32, i64, i64, i64, i8*, i32, i32 }
%"class.std::__1::thread" = type { i64 }
%"class.std::__1::shared_ptr" = type { %"class.Kalmar::KalmarAsyncOp"*, %"class.std::__1::__shared_weak_count"* }
%"class.Kalmar::KalmarAsyncOp" = type { i32 (...)** }
%"class.Kalmar::KalmarContext" = type { i32 (...)**, %"class.Kalmar::KalmarDevice"*, %"class.std::__1::vector" }
%"class.std::__1::vector" = type { %"class.std::__1::__vector_base" }
%"class.std::__1::__vector_base" = type { %"class.Kalmar::KalmarDevice"**, %"class.Kalmar::KalmarDevice"**, %"class.std::__1::__compressed_pair.25" }
%"class.std::__1::__compressed_pair.25" = type { %"class.std::__1::__libcpp_compressed_pair_imp.26" }
%"class.std::__1::__libcpp_compressed_pair_imp.26" = type { %"class.Kalmar::KalmarDevice"** }
%"class.hc::extent" = type { %"struct.Kalmar::index_impl" }
%"struct.(anonymous namespace)::foo_func" = type { %struct.tid_s, i32*, i32 }

@.str = private unnamed_addr constant [22 x i8] c"__cxxamp_opencl_index\00", section "llvm.metadata"
@.str1 = private unnamed_addr constant [50 x i8] c"/home/whchung/cppamp35/src/include/kalmar_index.h\00", section "llvm.metadata"
@_ZN6KalmarL20__mcw_cxxamp_kernelsE = internal global %"class.std::__1::set" zeroinitializer, align 8
@__dso_handle = external global i8
@.str2 = private unnamed_addr constant [5 x i8] c"%d \0A\00", align 1
@.str3 = private unnamed_addr constant [10 x i8] c"success!\0A\00", align 1
@.str4 = private unnamed_addr constant [12 x i8] c"no success\0A\00", align 1
@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_test.cpp, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_test_camp.cpp, i8* null }]
@llvm.global.annotations = appending global [7 x { i8*, i8*, i8*, i32 }] [{ i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index"*)* @_ZN6Kalmar5indexILi1EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([50 x i8]* @.str1, i32 0, i32 0), i32 452 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.0"*)* @_ZN6Kalmar5indexILi2EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([50 x i8]* @.str1, i32 0, i32 0), i32 452 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.3"*)* @_ZN6Kalmar5indexILi3EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([50 x i8]* @.str1, i32 0, i32 0), i32 452 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index"*)* @_ZN6Kalmar5indexILi1EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str5, i32 0, i32 0), i8* getelementptr inbounds ([50 x i8]* @.str16, i32 0, i32 0), i32 452 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.0"*)* @_ZN6Kalmar5indexILi2EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str5, i32 0, i32 0), i8* getelementptr inbounds ([50 x i8]* @.str16, i32 0, i32 0), i32 452 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (%"class.Kalmar::index.3"*)* @_ZN6Kalmar5indexILi3EE21__cxxamp_opencl_indexEv to i8*), i8* getelementptr inbounds ([22 x i8]* @.str5, i32 0, i32 0), i8* getelementptr inbounds ([50 x i8]* @.str16, i32 0, i32 0), i32 452 }, { i8*, i8*, i8*, i32 } { i8* bitcast (void (i32, i32*, i32*, i32*, i32)* @_ZN12_GLOBAL__N_18foo_func19__cxxamp_trampolineEiPiS1_S1_i to i8*), i8* getelementptr inbounds ([20 x i8]* @.str28, i32 0, i32 0), i8* getelementptr inbounds ([14 x i8]* @.str39, i32 0, i32 0), i32 7 }], section "llvm.metadata"
@.str5 = private unnamed_addr constant [22 x i8] c"__cxxamp_opencl_index\00", section "llvm.metadata"
@.str16 = private unnamed_addr constant [50 x i8] c"/home/whchung/cppamp35/src/include/kalmar_index.h\00", section "llvm.metadata"
@_ZN6KalmarL20__mcw_cxxamp_kernelsE7 = internal global %"class.std::__1::set" zeroinitializer, align 8
@.str28 = private unnamed_addr constant [20 x i8] c"__cxxamp_trampoline\00", section "llvm.metadata"
@.str39 = private unnamed_addr constant [14 x i8] c"test_camp.cpp\00", section "llvm.metadata"
@stderr = external global %struct._IO_FILE*
@.str410 = private unnamed_addr constant [54 x i8] c"There is no device can be used to do the computation\0A\00", align 1
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%"class.hc::completion_future"*, %"class.hc::accelerator_view"*, %"class.hc::extent"*, %"struct.(anonymous namespace)::foo_func"*)* @_ZN2hc17parallel_for_eachIN12_GLOBAL__N_18foo_funcEEENS_17completion_futureERKNS_16accelerator_viewERKNS_6extentILi1EEERKT_ to i8*)], section "llvm.metadata"

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2Ev(%"class.Kalmar::index"* nocapture %this) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 0, i32* %1, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1Ev(%"class.Kalmar::index"* nocapture %this) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 0, i32* %1, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2ERKS1_(%"class.Kalmar::index"* nocapture %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %other) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %other, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %3, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1ERKS1_(%"class.Kalmar::index"* nocapture %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %other) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %other, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %3, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2Ei(%"class.Kalmar::index"* nocapture %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %i0, i32* %1, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1Ei(%"class.Kalmar::index"* nocapture %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %i0, i32* %1, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2EPKi(%"class.Kalmar::index"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1EPKi(%"class.Kalmar::index"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2EPi(%"class.Kalmar::index"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1EPi(%"class.Kalmar::index"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEaSERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %other) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %other, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %3, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind readonly uwtable
define weak_odr i32 @_ZNK6Kalmar5indexILi1EEixEj(%"class.Kalmar::index"* nocapture readonly %this, i32 %c) #1 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0
  %2 = zext i32 %c to i64
  %3 = getelementptr inbounds %"struct.Kalmar::index_impl"* %1, i64 %2, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !12
  ret i32 %4
}

; Function Attrs: alwaysinline nounwind readnone uwtable
define weak_odr dereferenceable(4) i32* @_ZN6Kalmar5indexILi1EEixEj(%"class.Kalmar::index"* readnone %this, i32 %c) #2 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0
  %2 = zext i32 %c to i64
  %3 = getelementptr inbounds %"struct.Kalmar::index_impl"* %1, i64 %2, i32 0, i32 0
  ret i32* %3
}

; Function Attrs: alwaysinline nounwind readonly uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi1EEeqERKS1_(%"class.Kalmar::index"* nocapture readonly %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %other) #1 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %other, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = icmp eq i32 %2, %4
  ret i1 %5
}

; Function Attrs: alwaysinline nounwind readonly uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi1EEneERKS1_(%"class.Kalmar::index"* nocapture readonly %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %other) #1 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %other, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = icmp ne i32 %2, %4
  ret i1 %5
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEpLERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %rhs) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %rhs, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = add nsw i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmIERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %rhs) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %rhs, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = sub nsw i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmLERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %__r) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %__r, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = mul nsw i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEdVERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %__r) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %__r, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = sdiv i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EErMERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* nocapture readonly dereferenceable(8) %__r) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %__r, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = srem i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEpLEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = add nsw i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmIEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = sub nsw i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmLEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = mul nsw i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEdVEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = sdiv i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EErMEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = srem i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEppEv(%"class.Kalmar::index"* %this) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEppEi(%"class.Kalmar::index"* noalias nocapture sret %agg.result, %"class.Kalmar::index"* nocapture %this, i32) #0 align 2 {
  %2 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index"* %agg.result, i64 0, i32 0, i32 0, i32 0
  store i32 %3, i32* %4, align 4, !tbaa !7
  %5 = add nsw i32 %3, 1
  store i32 %5, i32* %2, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmmEv(%"class.Kalmar::index"* %this) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = add nsw i32 %2, -1
  store i32 %3, i32* %1, align 4, !tbaa !7
  ret %"class.Kalmar::index"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEmmEi(%"class.Kalmar::index"* noalias nocapture sret %agg.result, %"class.Kalmar::index"* nocapture %this, i32) #0 align 2 {
  %2 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index"* %agg.result, i64 0, i32 0, i32 0, i32 0
  store i32 %3, i32* %4, align 4, !tbaa !7
  %5 = add nsw i32 %3, -1
  store i32 %5, i32* %2, align 4, !tbaa !7
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EE21__cxxamp_opencl_indexEv(%"class.Kalmar::index"* nocapture %this) #0 align 2 {
  %1 = tail call i64 @amp_get_global_id(i32 0) #15
  %2 = trunc i64 %1 to i32
  %3 = getelementptr inbounds %"class.Kalmar::index"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %3, align 4, !tbaa !12
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2Ev(%"class.Kalmar::index.0"* nocapture %this) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 0, i32* %1, align 4, !tbaa !7
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 0, i32* %2, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1Ev(%"class.Kalmar::index.0"* nocapture %this) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 0, i32* %1, align 4, !tbaa !7
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 0, i32* %2, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2ERKS1_(%"class.Kalmar::index.0"* nocapture %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %other) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 1, i32 0
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %5, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %6, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1ERKS1_(%"class.Kalmar::index.0"* nocapture %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %other) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 1, i32 0
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %5, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %6, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2Ei(%"class.Kalmar::index.0"* nocapture %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %i0, i32* %1, align 4, !tbaa !7
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %i0, i32* %2, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1Ei(%"class.Kalmar::index.0"* nocapture %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %i0, i32* %1, align 4, !tbaa !7
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %i0, i32* %2, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2EPKi(%"class.Kalmar::index.0"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  %3 = getelementptr inbounds i32* %components, i64 1
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %5, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1EPKi(%"class.Kalmar::index.0"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  %3 = getelementptr inbounds i32* %components, i64 1
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %5, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2EPi(%"class.Kalmar::index.0"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  %3 = getelementptr inbounds i32* %components, i64 1
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %5, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1EPi(%"class.Kalmar::index.0"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  %3 = getelementptr inbounds i32* %components, i64 1
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %5, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEaSERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %other) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %3, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !12
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %5, i32* %6, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind readonly uwtable
define weak_odr i32 @_ZNK6Kalmar5indexILi2EEixEj(%"class.Kalmar::index.0"* nocapture readonly %this, i32 %c) #1 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0
  %2 = zext i32 %c to i64
  %3 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 %2, i32 0
  %4 = load i32* %3, align 4, !tbaa !12
  ret i32 %4
}

; Function Attrs: alwaysinline nounwind readnone uwtable
define weak_odr dereferenceable(4) i32* @_ZN6Kalmar5indexILi2EEixEj(%"class.Kalmar::index.0"* readnone %this, i32 %c) #2 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0
  %2 = zext i32 %c to i64
  %3 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 %2, i32 0
  ret i32* %3
}

; Function Attrs: alwaysinline nounwind readonly uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi2EEeqERKS1_(%"class.Kalmar::index.0"* nocapture readonly %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %other) #1 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0
  %2 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 1, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 0
  %5 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %4, i64 1, i32 0
  %6 = load i32* %5, align 4, !tbaa !12
  %7 = icmp eq i32 %3, %6
  br i1 %7, label %8, label %_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_.exit

; <label>:8                                       ; preds = %0
  %9 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %10 = load i32* %9, align 4, !tbaa !12
  %11 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 0, i32 0
  %12 = load i32* %11, align 4, !tbaa !12
  %13 = icmp eq i32 %10, %12
  br label %_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_.exit

_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_.exit: ; preds = %8, %0
  %14 = phi i1 [ false, %0 ], [ %13, %8 ]
  ret i1 %14
}

; Function Attrs: alwaysinline nounwind readonly uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi2EEneERKS1_(%"class.Kalmar::index.0"* nocapture readonly %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %other) #1 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0
  %2 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 1, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 0
  %5 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %4, i64 1, i32 0
  %6 = load i32* %5, align 4, !tbaa !12
  %7 = icmp eq i32 %3, %6
  br i1 %7, label %8, label %_ZNK6Kalmar5indexILi2EEeqERKS1_.exit

; <label>:8                                       ; preds = %0
  %9 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %10 = load i32* %9, align 4, !tbaa !12
  %11 = getelementptr inbounds %"class.Kalmar::index.0"* %other, i64 0, i32 0, i32 0, i32 0
  %12 = load i32* %11, align 4, !tbaa !12
  %phitmp = icmp ne i32 %10, %12
  br label %_ZNK6Kalmar5indexILi2EEeqERKS1_.exit

_ZNK6Kalmar5indexILi2EEeqERKS1_.exit:             ; preds = %8, %0
  %13 = phi i1 [ true, %0 ], [ %phitmp, %8 ]
  ret i1 %13
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEpLERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %rhs) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %rhs, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = add nsw i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %rhs, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = add nsw i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmIERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %rhs) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %rhs, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = sub nsw i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %rhs, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = sub nsw i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmLERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %__r) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %__r, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = mul nsw i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %__r, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = mul nsw i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEdVERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %__r) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %__r, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = sdiv i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %__r, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = sdiv i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EErMERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* nocapture readonly dereferenceable(16) %__r) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %__r, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = srem i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %__r, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = srem i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEpLEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = add nsw i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = add nsw i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmIEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = sub nsw i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = sub nsw i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmLEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = mul nsw i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = mul nsw i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEdVEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = sdiv i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = sdiv i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EErMEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = srem i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = srem i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEppEv(%"class.Kalmar::index.0"* %this) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = add nsw i32 %5, 1
  store i32 %6, i32* %4, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEppEi(%"class.Kalmar::index.0"* noalias nocapture sret %agg.result, %"class.Kalmar::index.0"* nocapture %this, i32) #0 align 2 {
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !12
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %agg.result, i64 0, i32 0, i32 0, i32 0
  store i32 %3, i32* %6, align 4, !tbaa !7
  %7 = getelementptr inbounds %"class.Kalmar::index.0"* %agg.result, i64 0, i32 0, i32 1, i32 0
  store i32 %5, i32* %7, align 4, !tbaa !13
  %8 = add nsw i32 %3, 1
  store i32 %8, i32* %2, align 4, !tbaa !7
  %9 = add nsw i32 %5, 1
  store i32 %9, i32* %4, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmmEv(%"class.Kalmar::index.0"* %this) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = add nsw i32 %2, -1
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = add nsw i32 %5, -1
  store i32 %6, i32* %4, align 4, !tbaa !13
  ret %"class.Kalmar::index.0"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEmmEi(%"class.Kalmar::index.0"* noalias nocapture sret %agg.result, %"class.Kalmar::index.0"* nocapture %this, i32) #0 align 2 {
  %2 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !12
  %6 = getelementptr inbounds %"class.Kalmar::index.0"* %agg.result, i64 0, i32 0, i32 0, i32 0
  store i32 %3, i32* %6, align 4, !tbaa !7
  %7 = getelementptr inbounds %"class.Kalmar::index.0"* %agg.result, i64 0, i32 0, i32 1, i32 0
  store i32 %5, i32* %7, align 4, !tbaa !13
  %8 = add nsw i32 %3, -1
  store i32 %8, i32* %2, align 4, !tbaa !7
  %9 = add nsw i32 %5, -1
  store i32 %9, i32* %4, align 4, !tbaa !13
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EE21__cxxamp_opencl_indexEv(%"class.Kalmar::index.0"* nocapture %this) #0 align 2 {
  %1 = tail call i64 @amp_get_global_id(i32 0) #15
  %2 = trunc i64 %1 to i32
  %3 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0
  %4 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %3, i64 1, i32 0
  store i32 %2, i32* %4, align 4, !tbaa !12
  %5 = tail call i64 @amp_get_global_id(i32 1) #15
  %6 = trunc i64 %5 to i32
  %7 = getelementptr inbounds %"class.Kalmar::index.0"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %6, i32* %7, align 4, !tbaa !12
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2Ev(%"class.Kalmar::index.3"* nocapture %this) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 0, i32* %1, align 4, !tbaa !7
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 0, i32* %2, align 4, !tbaa !13
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 0, i32* %3, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1Ev(%"class.Kalmar::index.3"* nocapture %this) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 0, i32* %1, align 4, !tbaa !7
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 0, i32* %2, align 4, !tbaa !13
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 0, i32* %3, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2ERKS1_(%"class.Kalmar::index.3"* nocapture %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %other) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 1, i32 0
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 2, i32 0
  %6 = load i32* %5, align 4, !tbaa !12
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %7, align 4, !tbaa !7
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %8, align 4, !tbaa !13
  %9 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 %6, i32* %9, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1ERKS1_(%"class.Kalmar::index.3"* nocapture %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %other) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 1, i32 0
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 2, i32 0
  %6 = load i32* %5, align 4, !tbaa !12
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %7, align 4, !tbaa !7
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %8, align 4, !tbaa !13
  %9 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 %6, i32* %9, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2Ei(%"class.Kalmar::index.3"* nocapture %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %i0, i32* %1, align 4, !tbaa !7
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %i0, i32* %2, align 4, !tbaa !13
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 %i0, i32* %3, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1Ei(%"class.Kalmar::index.3"* nocapture %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %i0, i32* %1, align 4, !tbaa !7
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %i0, i32* %2, align 4, !tbaa !13
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 %i0, i32* %3, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2EPKi(%"class.Kalmar::index.3"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  %3 = getelementptr inbounds i32* %components, i64 1
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %5, align 4, !tbaa !13
  %6 = getelementptr inbounds i32* %components, i64 2
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 %7, i32* %8, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1EPKi(%"class.Kalmar::index.3"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  %3 = getelementptr inbounds i32* %components, i64 1
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %5, align 4, !tbaa !13
  %6 = getelementptr inbounds i32* %components, i64 2
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 %7, i32* %8, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2EPi(%"class.Kalmar::index.3"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  %3 = getelementptr inbounds i32* %components, i64 1
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %5, align 4, !tbaa !13
  %6 = getelementptr inbounds i32* %components, i64 2
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 %7, i32* %8, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1EPi(%"class.Kalmar::index.3"* nocapture %this, i32* nocapture readonly %components) unnamed_addr #0 align 2 {
  %1 = load i32* %components, align 4, !tbaa !12
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %1, i32* %2, align 4, !tbaa !7
  %3 = getelementptr inbounds i32* %components, i64 1
  %4 = load i32* %3, align 4, !tbaa !12
  %5 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %4, i32* %5, align 4, !tbaa !13
  %6 = getelementptr inbounds i32* %components, i64 2
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 %7, i32* %8, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEaSERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %other) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %2, i32* %3, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !12
  %6 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  store i32 %5, i32* %6, align 4, !tbaa !13
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 2, i32 0
  %8 = load i32* %7, align 4, !tbaa !12
  %9 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  store i32 %8, i32* %9, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind readonly uwtable
define weak_odr i32 @_ZNK6Kalmar5indexILi3EEixEj(%"class.Kalmar::index.3"* nocapture readonly %this, i32 %c) #1 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0
  %2 = zext i32 %c to i64
  %3 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 %2, i32 0
  %4 = load i32* %3, align 4, !tbaa !12
  ret i32 %4
}

; Function Attrs: alwaysinline nounwind readnone uwtable
define weak_odr dereferenceable(4) i32* @_ZN6Kalmar5indexILi3EEixEj(%"class.Kalmar::index.3"* readnone %this, i32 %c) #2 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0
  %2 = zext i32 %c to i64
  %3 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 %2, i32 0
  ret i32* %3
}

; Function Attrs: alwaysinline nounwind readonly uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi3EEeqERKS1_(%"class.Kalmar::index.3"* nocapture readonly %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %other) #1 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0
  %2 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 2, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 0
  %5 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %4, i64 2, i32 0
  %6 = load i32* %5, align 4, !tbaa !12
  %7 = icmp eq i32 %3, %6
  br i1 %7, label %8, label %_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_.exit

; <label>:8                                       ; preds = %0
  %9 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 1, i32 0
  %10 = load i32* %9, align 4, !tbaa !12
  %11 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %4, i64 1, i32 0
  %12 = load i32* %11, align 4, !tbaa !12
  %13 = icmp eq i32 %10, %12
  br i1 %13, label %14, label %_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_.exit

; <label>:14                                      ; preds = %8
  %15 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %16 = load i32* %15, align 4, !tbaa !12
  %17 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 0, i32 0
  %18 = load i32* %17, align 4, !tbaa !12
  %19 = icmp eq i32 %16, %18
  br label %_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_.exit

_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_.exit: ; preds = %14, %8, %0
  %20 = phi i1 [ false, %0 ], [ false, %8 ], [ %19, %14 ]
  ret i1 %20
}

; Function Attrs: alwaysinline nounwind readonly uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi3EEneERKS1_(%"class.Kalmar::index.3"* nocapture readonly %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %other) #1 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0
  %2 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 2, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 0
  %5 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %4, i64 2, i32 0
  %6 = load i32* %5, align 4, !tbaa !12
  %7 = icmp eq i32 %3, %6
  br i1 %7, label %8, label %_ZNK6Kalmar5indexILi3EEeqERKS1_.exit

; <label>:8                                       ; preds = %0
  %9 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %1, i64 1, i32 0
  %10 = load i32* %9, align 4, !tbaa !12
  %11 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %4, i64 1, i32 0
  %12 = load i32* %11, align 4, !tbaa !12
  %13 = icmp eq i32 %10, %12
  br i1 %13, label %14, label %_ZNK6Kalmar5indexILi3EEeqERKS1_.exit

; <label>:14                                      ; preds = %8
  %15 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %16 = load i32* %15, align 4, !tbaa !12
  %17 = getelementptr inbounds %"class.Kalmar::index.3"* %other, i64 0, i32 0, i32 0, i32 0
  %18 = load i32* %17, align 4, !tbaa !12
  %phitmp = icmp ne i32 %16, %18
  br label %_ZNK6Kalmar5indexILi3EEeqERKS1_.exit

_ZNK6Kalmar5indexILi3EEeqERKS1_.exit:             ; preds = %14, %8, %0
  %19 = phi i1 [ true, %0 ], [ true, %8 ], [ %phitmp, %14 ]
  ret i1 %19
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEpLERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %rhs) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %rhs, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = add nsw i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.3"* %rhs, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = add nsw i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  %11 = getelementptr inbounds %"class.Kalmar::index.3"* %rhs, i64 0, i32 0, i32 2, i32 0
  %12 = load i32* %11, align 4, !tbaa !12
  %13 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %14 = load i32* %13, align 4, !tbaa !15
  %15 = add nsw i32 %14, %12
  store i32 %15, i32* %13, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmIERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %rhs) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %rhs, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = sub nsw i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.3"* %rhs, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = sub nsw i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  %11 = getelementptr inbounds %"class.Kalmar::index.3"* %rhs, i64 0, i32 0, i32 2, i32 0
  %12 = load i32* %11, align 4, !tbaa !12
  %13 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %14 = load i32* %13, align 4, !tbaa !15
  %15 = sub nsw i32 %14, %12
  store i32 %15, i32* %13, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmLERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %__r) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %__r, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = mul nsw i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.3"* %__r, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = mul nsw i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  %11 = getelementptr inbounds %"class.Kalmar::index.3"* %__r, i64 0, i32 0, i32 2, i32 0
  %12 = load i32* %11, align 4, !tbaa !12
  %13 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %14 = load i32* %13, align 4, !tbaa !15
  %15 = mul nsw i32 %14, %12
  store i32 %15, i32* %13, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEdVERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %__r) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %__r, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = sdiv i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.3"* %__r, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = sdiv i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  %11 = getelementptr inbounds %"class.Kalmar::index.3"* %__r, i64 0, i32 0, i32 2, i32 0
  %12 = load i32* %11, align 4, !tbaa !12
  %13 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %14 = load i32* %13, align 4, !tbaa !15
  %15 = sdiv i32 %14, %12
  store i32 %15, i32* %13, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EErMERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* nocapture readonly dereferenceable(24) %__r) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %__r, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !12
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %4 = load i32* %3, align 4, !tbaa !7
  %5 = srem i32 %4, %2
  store i32 %5, i32* %3, align 4, !tbaa !7
  %6 = getelementptr inbounds %"class.Kalmar::index.3"* %__r, i64 0, i32 0, i32 1, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load i32* %8, align 4, !tbaa !13
  %10 = srem i32 %9, %7
  store i32 %10, i32* %8, align 4, !tbaa !13
  %11 = getelementptr inbounds %"class.Kalmar::index.3"* %__r, i64 0, i32 0, i32 2, i32 0
  %12 = load i32* %11, align 4, !tbaa !12
  %13 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %14 = load i32* %13, align 4, !tbaa !15
  %15 = srem i32 %14, %12
  store i32 %15, i32* %13, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEpLEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = add nsw i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = add nsw i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %8 = load i32* %7, align 4, !tbaa !15
  %9 = add nsw i32 %8, %value
  store i32 %9, i32* %7, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmIEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = sub nsw i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = sub nsw i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %8 = load i32* %7, align 4, !tbaa !15
  %9 = sub nsw i32 %8, %value
  store i32 %9, i32* %7, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmLEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = mul nsw i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = mul nsw i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %8 = load i32* %7, align 4, !tbaa !15
  %9 = mul nsw i32 %8, %value
  store i32 %9, i32* %7, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEdVEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = sdiv i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = sdiv i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %8 = load i32* %7, align 4, !tbaa !15
  %9 = sdiv i32 %8, %value
  store i32 %9, i32* %7, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EErMEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = srem i32 %2, %value
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = srem i32 %5, %value
  store i32 %6, i32* %4, align 4, !tbaa !13
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %8 = load i32* %7, align 4, !tbaa !15
  %9 = srem i32 %8, %value
  store i32 %9, i32* %7, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEppEv(%"class.Kalmar::index.3"* %this) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = add nsw i32 %5, 1
  store i32 %6, i32* %4, align 4, !tbaa !13
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %8 = load i32* %7, align 4, !tbaa !15
  %9 = add nsw i32 %8, 1
  store i32 %9, i32* %7, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEppEi(%"class.Kalmar::index.3"* noalias nocapture sret %agg.result, %"class.Kalmar::index.3"* nocapture %this, i32) #0 align 2 {
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !12
  %6 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %agg.result, i64 0, i32 0, i32 0, i32 0
  store i32 %3, i32* %8, align 4, !tbaa !7
  %9 = getelementptr inbounds %"class.Kalmar::index.3"* %agg.result, i64 0, i32 0, i32 1, i32 0
  store i32 %5, i32* %9, align 4, !tbaa !13
  %10 = getelementptr inbounds %"class.Kalmar::index.3"* %agg.result, i64 0, i32 0, i32 2, i32 0
  store i32 %7, i32* %10, align 4, !tbaa !15
  %11 = add nsw i32 %3, 1
  store i32 %11, i32* %2, align 4, !tbaa !7
  %12 = add nsw i32 %5, 1
  store i32 %12, i32* %4, align 4, !tbaa !13
  %13 = add nsw i32 %7, 1
  store i32 %13, i32* %6, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmmEv(%"class.Kalmar::index.3"* %this) #0 align 2 {
  %1 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %2 = load i32* %1, align 4, !tbaa !7
  %3 = add nsw i32 %2, -1
  store i32 %3, i32* %1, align 4, !tbaa !7
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !13
  %6 = add nsw i32 %5, -1
  store i32 %6, i32* %4, align 4, !tbaa !13
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %8 = load i32* %7, align 4, !tbaa !15
  %9 = add nsw i32 %8, -1
  store i32 %9, i32* %7, align 4, !tbaa !15
  ret %"class.Kalmar::index.3"* %this
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEmmEi(%"class.Kalmar::index.3"* noalias nocapture sret %agg.result, %"class.Kalmar::index.3"* nocapture %this, i32) #0 align 2 {
  %2 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  %3 = load i32* %2, align 4, !tbaa !12
  %4 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 1, i32 0
  %5 = load i32* %4, align 4, !tbaa !12
  %6 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 2, i32 0
  %7 = load i32* %6, align 4, !tbaa !12
  %8 = getelementptr inbounds %"class.Kalmar::index.3"* %agg.result, i64 0, i32 0, i32 0, i32 0
  store i32 %3, i32* %8, align 4, !tbaa !7
  %9 = getelementptr inbounds %"class.Kalmar::index.3"* %agg.result, i64 0, i32 0, i32 1, i32 0
  store i32 %5, i32* %9, align 4, !tbaa !13
  %10 = getelementptr inbounds %"class.Kalmar::index.3"* %agg.result, i64 0, i32 0, i32 2, i32 0
  store i32 %7, i32* %10, align 4, !tbaa !15
  %11 = add nsw i32 %3, -1
  store i32 %11, i32* %2, align 4, !tbaa !7
  %12 = add nsw i32 %5, -1
  store i32 %12, i32* %4, align 4, !tbaa !13
  %13 = add nsw i32 %7, -1
  store i32 %13, i32* %6, align 4, !tbaa !15
  ret void
}

; Function Attrs: alwaysinline nounwind uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EE21__cxxamp_opencl_indexEv(%"class.Kalmar::index.3"* nocapture %this) #0 align 2 {
  %1 = tail call i64 @amp_get_global_id(i32 0) #15
  %2 = trunc i64 %1 to i32
  %3 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0
  %4 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %3, i64 2, i32 0
  store i32 %2, i32* %4, align 4, !tbaa !12
  %5 = tail call i64 @amp_get_global_id(i32 1) #15
  %6 = trunc i64 %5 to i32
  %7 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %3, i64 1, i32 0
  store i32 %6, i32* %7, align 4, !tbaa !12
  %8 = tail call i64 @amp_get_global_id(i32 2) #15
  %9 = trunc i64 %8 to i32
  %10 = getelementptr inbounds %"class.Kalmar::index.3"* %this, i64 0, i32 0, i32 0, i32 0
  store i32 %9, i32* %10, align 4, !tbaa !12
  ret void
}

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #3

; Function Attrs: alwaysinline nounwind uwtable
define void @foo(%struct.tid_s* byval nocapture readonly align 8 %t, i32* nocapture %a, i32 %vecsize) #0 {
  %1 = getelementptr inbounds %struct.tid_s* %t, i64 0, i32 0
  %2 = load i32* %1, align 8, !tbaa !17
  %3 = icmp slt i32 %2, %vecsize
  br i1 %3, label %4, label %7

; <label>:4                                       ; preds = %0
  %5 = sext i32 %2 to i64
  %6 = getelementptr inbounds i32* %a, i64 %5
  store i32 %2, i32* %6, align 4, !tbaa !12
  br label %7

; <label>:7                                       ; preds = %4, %0
  ret void
}

; Function Attrs: uwtable
define i32 @main() #4 {
  %t = alloca %struct.tid_s, align 8
  %1 = alloca %struct.tid_s, align 8
  %2 = tail call noalias i8* @_Znam(i64 40000) #16
  %3 = bitcast i8* %2 to i32*
  %4 = bitcast %struct.tid_s* %1 to i8*
  %5 = bitcast %struct.tid_s* %t to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %4, i8* %5, i64 24, i32 8, i1 false), !tbaa.struct !20
  call void @launch_foo(%struct.tid_s* byval align 8 %1, i32* %3, i32 10000) #17
  br label %9

; <label>:6                                       ; preds = %9
  %7 = trunc i64 %indvars.iv.next to i32
  %8 = icmp slt i32 %7, 10000
  br i1 %8, label %9, label %14

; <label>:9                                       ; preds = %6, %0
  %indvars.iv = phi i64 [ 0, %0 ], [ %indvars.iv.next, %6 ]
  %10 = getelementptr inbounds i32* %3, i64 %indvars.iv
  %11 = load i32* %10, align 4, !tbaa !12
  %12 = trunc i64 %indvars.iv to i32
  %13 = icmp eq i32 %11, %12
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 %13, label %6, label %16

; <label>:14                                      ; preds = %6
  %15 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str3, i64 0, i64 0)) #17
  br label %19

; <label>:16                                      ; preds = %9
  %.lcssa = phi i32 [ %12, %9 ]
  %17 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([5 x i8]* @.str2, i64 0, i64 0), i32 %.lcssa) #17
  %18 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([12 x i8]* @.str4, i64 0, i64 0)) #17
  br label %19

; <label>:19                                      ; preds = %16, %14
  %.0 = phi i32 [ 0, %14 ], [ 1, %16 ]
  ret i32 %.0
}

; Function Attrs: nobuiltin
declare noalias i8* @_Znam(i64) #5

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #3

declare i32 @printf(i8*, ...) #6

; Function Attrs: nounwind readnone
declare i64 @amp_get_global_id(i32) #7

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) #8

; Function Attrs: uwtable
define void @launch_foo(%struct.tid_s* byval nocapture readnone align 8 %t, i32* %a, i32 %vecsize) #4 {
  %1 = alloca %"class.hc::accelerator_view", align 8
  %2 = alloca %"class.hc::completion_future", align 8
  %3 = bitcast %"class.hc::accelerator_view"* %1 to i8*
  call void @llvm.lifetime.start(i64 16, i8* %3)
  call void @_ZN2hc11accelerator23get_auto_selection_viewEv(%"class.hc::accelerator_view"* sret %1) #17
  %4 = getelementptr inbounds %"class.hc::accelerator_view"* %1, i64 0, i32 0, i32 1
  %5 = load %"class.std::__1::__shared_weak_count"** %4, align 8, !tbaa !22
  %6 = icmp eq %"class.std::__1::__shared_weak_count"* %5, null
  br i1 %6, label %8, label %7

; <label>:7                                       ; preds = %0
  tail call void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"* %5) #18
  br label %8

; <label>:8                                       ; preds = %7, %0
  call void @llvm.lifetime.end(i64 16, i8* %3)
  invoke void @_ZNSt3__117__assoc_sub_state4waitEv(%"class.std::__1::__assoc_sub_state"* undef) #17
          to label %_ZNK2hc17completion_future4waitEv.exit unwind label %9

_ZNK2hc17completion_future4waitEv.exit:           ; preds = %8
  call void @_ZN2hc17completion_futureD2Ev(%"class.hc::completion_future"* %2) #18
  ret void

; <label>:9                                       ; preds = %8
  %10 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  call void @_ZN2hc17completion_futureD2Ev(%"class.hc::completion_future"* %2) #18
  resume { i8*, i32 } %10
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"*) #9

declare %"class.Kalmar::KalmarContext"* @_ZN6Kalmar10getContextEv() #6

; Function Attrs: nounwind
declare void @_ZNSt3__119__shared_weak_count12__add_sharedEv(%"class.std::__1::__shared_weak_count"*) #9

declare void @_ZNSt3__15mutex4lockEv(%"class.std::__1::mutex"*) #6

; Function Attrs: nounwind
declare void @_ZNSt3__15mutex6unlockEv(%"class.std::__1::mutex"*) #9

declare i8* @__cxa_begin_catch(i8*)

declare void @_ZSt9terminatev()

; Function Attrs: nobuiltin
declare noalias i8* @_Znwm(i64) #5

; Function Attrs: nounwind readnone
declare i64 @pthread_self() #7

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...) #6

; Function Attrs: noreturn nounwind
declare void @exit(i32) #10

declare void @_ZNSt3__16thread4joinEv(%"class.std::__1::thread"*) #6

; Function Attrs: nounwind
declare void @_ZNSt3__16threadD1Ev(%"class.std::__1::thread"*) #9

; Function Attrs: nounwind
declare void @_ZNSt3__113shared_futureIvED1Ev(%"class.std::__1::shared_future"*) #9

declare void @_ZNSt3__117__assoc_sub_state4waitEv(%"class.std::__1::__assoc_sub_state"*) #6

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #3

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #3

; Function Attrs: uwtable
define linkonce_odr void @_ZN2hc11accelerator23get_auto_selection_viewEv(%"class.hc::accelerator_view"* noalias nocapture sret %agg.result) #4 align 2 {
  %1 = alloca %"class.std::__1::shared_ptr.13", align 8
  %2 = tail call %"class.Kalmar::KalmarContext"* @_ZN6Kalmar10getContextEv() #17
  %3 = getelementptr inbounds %"class.Kalmar::KalmarContext"* %2, i64 0, i32 1
  %4 = load %"class.Kalmar::KalmarDevice"** %3, align 8, !tbaa !24
  %5 = icmp eq %"class.Kalmar::KalmarDevice"* %4, null
  br i1 %5, label %6, label %_ZN6Kalmar13KalmarContext11auto_selectEv.exit

; <label>:6                                       ; preds = %0
  %7 = getelementptr inbounds %"class.Kalmar::KalmarContext"* %2, i64 0, i32 2, i32 0, i32 1
  %8 = load %"class.Kalmar::KalmarDevice"*** %7, align 8, !tbaa !27
  %9 = getelementptr inbounds %"class.Kalmar::KalmarContext"* %2, i64 0, i32 2, i32 0, i32 0
  %10 = load %"class.Kalmar::KalmarDevice"*** %9, align 8, !tbaa !30
  %11 = ptrtoint %"class.Kalmar::KalmarDevice"** %8 to i64
  %12 = ptrtoint %"class.Kalmar::KalmarDevice"** %10 to i64
  %13 = sub i64 %11, %12
  %14 = ashr exact i64 %13, 3
  %15 = icmp ult i64 %14, 2
  br i1 %15, label %16, label %19

; <label>:16                                      ; preds = %6
  %17 = load %struct._IO_FILE** @stderr, align 8, !tbaa !21
  %18 = tail call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %17, i8* getelementptr inbounds ([54 x i8]* @.str410, i64 0, i64 0)) #17
  tail call void @exit(i32 -1) #19
  unreachable

; <label>:19                                      ; preds = %6
  %20 = getelementptr inbounds %"class.Kalmar::KalmarDevice"** %10, i64 1
  %21 = load %"class.Kalmar::KalmarDevice"** %20, align 8, !tbaa !21
  store %"class.Kalmar::KalmarDevice"* %21, %"class.Kalmar::KalmarDevice"** %3, align 8, !tbaa !24
  br label %_ZN6Kalmar13KalmarContext11auto_selectEv.exit

_ZN6Kalmar13KalmarContext11auto_selectEv.exit:    ; preds = %19, %0
  %22 = phi %"class.Kalmar::KalmarDevice"* [ %4, %0 ], [ %21, %19 ]
  call void @_ZN6Kalmar12KalmarDevice17get_default_queueEv(%"class.std::__1::shared_ptr.13"* sret %1, %"class.Kalmar::KalmarDevice"* %22) #17
  %23 = getelementptr inbounds %"class.hc::accelerator_view"* %agg.result, i64 0, i32 0, i32 0
  %24 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %1, i64 0, i32 0
  %25 = load %"class.Kalmar::KalmarQueue"** %24, align 8, !tbaa !31
  store %"class.Kalmar::KalmarQueue"* %25, %"class.Kalmar::KalmarQueue"** %23, align 8, !tbaa !31
  %26 = getelementptr inbounds %"class.hc::accelerator_view"* %agg.result, i64 0, i32 0, i32 1
  %27 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %1, i64 0, i32 1
  %28 = load %"class.std::__1::__shared_weak_count"** %27, align 8, !tbaa !22
  store %"class.std::__1::__shared_weak_count"* %28, %"class.std::__1::__shared_weak_count"** %26, align 8, !tbaa !22
  %29 = icmp eq %"class.std::__1::__shared_weak_count"* %28, null
  br i1 %29, label %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEED2Ev.exit, label %30

; <label>:30                                      ; preds = %_ZN6Kalmar13KalmarContext11auto_selectEv.exit
  tail call void @_ZNSt3__119__shared_weak_count12__add_sharedEv(%"class.std::__1::__shared_weak_count"* %28) #18
  tail call void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"* %28) #18
  br label %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEED2Ev.exit

_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEED2Ev.exit: ; preds = %30, %_ZN6Kalmar13KalmarContext11auto_selectEv.exit
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN2hc17completion_futureD2Ev(%"class.hc::completion_future"* %this) unnamed_addr #11 align 2 {
  %1 = getelementptr inbounds %"class.hc::completion_future"* %this, i64 0, i32 1
  %2 = load %"class.std::__1::thread"** %1, align 8, !tbaa !32
  %3 = icmp eq %"class.std::__1::thread"* %2, null
  br i1 %3, label %.thread, label %4

; <label>:4                                       ; preds = %0
  invoke void @_ZNSt3__16thread4joinEv(%"class.std::__1::thread"* %2) #17
          to label %13 unwind label %5

; <label>:5                                       ; preds = %4
  %6 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  %7 = extractvalue { i8*, i32 } %6, 0
  %8 = getelementptr inbounds %"class.hc::completion_future"* %this, i64 0, i32 2, i32 1
  %9 = load %"class.std::__1::__shared_weak_count"** %8, align 8, !tbaa !36
  %10 = icmp eq %"class.std::__1::__shared_weak_count"* %9, null
  br i1 %10, label %_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit, label %11

; <label>:11                                      ; preds = %5
  tail call void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"* %9) #18
  br label %_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit

_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit: ; preds = %11, %5
  %12 = getelementptr inbounds %"class.hc::completion_future"* %this, i64 0, i32 0
  tail call void @_ZNSt3__113shared_futureIvED1Ev(%"class.std::__1::shared_future"* %12) #18
  tail call void @__clang_call_terminate(i8* %7) #20
  unreachable

; <label>:13                                      ; preds = %4
  %.pr = load %"class.std::__1::thread"** %1, align 8, !tbaa !32
  %14 = icmp eq %"class.std::__1::thread"* %.pr, null
  br i1 %14, label %.thread, label %15

; <label>:15                                      ; preds = %13
  tail call void @_ZNSt3__16threadD1Ev(%"class.std::__1::thread"* %.pr) #18
  %16 = bitcast %"class.std::__1::thread"* %.pr to i8*
  tail call void @_ZdlPv(i8* %16) #21
  br label %.thread

.thread:                                          ; preds = %15, %13, %0
  store %"class.std::__1::thread"* null, %"class.std::__1::thread"** %1, align 8, !tbaa !32
  %17 = getelementptr inbounds %"class.hc::completion_future"* %this, i64 0, i32 2, i32 0
  %18 = load %"class.Kalmar::KalmarAsyncOp"** %17, align 8, !tbaa !37
  %19 = icmp eq %"class.Kalmar::KalmarAsyncOp"* %18, null
  br i1 %19, label %.thread._ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2_crit_edge, label %20

.thread._ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2_crit_edge: ; preds = %.thread
  %.pre = getelementptr inbounds %"class.hc::completion_future"* %this, i64 0, i32 2, i32 1
  br label %_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2

; <label>:20                                      ; preds = %.thread
  store %"class.Kalmar::KalmarAsyncOp"* null, %"class.Kalmar::KalmarAsyncOp"** %17, align 8, !tbaa !21
  %21 = getelementptr inbounds %"class.hc::completion_future"* %this, i64 0, i32 2, i32 1
  %22 = load %"class.std::__1::__shared_weak_count"** %21, align 8, !tbaa !21
  store %"class.std::__1::__shared_weak_count"* null, %"class.std::__1::__shared_weak_count"** %21, align 8, !tbaa !21
  %23 = icmp eq %"class.std::__1::__shared_weak_count"* %22, null
  br i1 %23, label %_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2, label %24

; <label>:24                                      ; preds = %20
  tail call void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"* %22) #18
  br label %_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2

_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2: ; preds = %24, %20, %.thread._ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2_crit_edge
  %.pre-phi = phi %"class.std::__1::__shared_weak_count"** [ %.pre, %.thread._ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2_crit_edge ], [ %21, %20 ], [ %21, %24 ]
  %25 = load %"class.std::__1::__shared_weak_count"** %.pre-phi, align 8, !tbaa !36
  %26 = icmp eq %"class.std::__1::__shared_weak_count"* %25, null
  br i1 %26, label %_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit1, label %27

; <label>:27                                      ; preds = %_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2
  tail call void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"* %25) #18
  br label %_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit1

_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit1: ; preds = %27, %_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev.exit2
  %28 = getelementptr inbounds %"class.hc::completion_future"* %this, i64 0, i32 0
  tail call void @_ZNSt3__113shared_futureIvED1Ev(%"class.std::__1::shared_future"* %28) #18
  ret void
}

; Function Attrs: uwtable
define internal spir_kernel void @_ZN12_GLOBAL__N_18foo_func19__cxxamp_trampolineEiPiS1_S1_i(i32, i32*, i32*, i32*, i32) #4 align 2 {
  %6 = alloca %struct.tid_s, align 8
  ; CHECK: %7 = alloca { i32, i32 addrspace(1)*, i32 addrspace(1)* }
  ; CHECK: %8 = alloca { i32, i32*, i32 addrspace(1)* }
  %7 = alloca %struct.tid_s, align 8
  ; CHECK-NOT: %7 = alloca %struct.tid_s, align 8
  ; CHECK: %9 = alloca %struct.tid_s, align 8
  %8 = tail call i64 @amp_get_global_id(i32 0) #15
  %9 = trunc i64 %8 to i32
  ; CHECK: %12 = bitcast { i32, i32 addrspace(1)*, i32 addrspace(1)* }* %7 to i8*
  ; CHECK: %13 = bitcast { i32, i32*, i32 addrspace(1)* }* %8 to i8*
  %10 = bitcast %struct.tid_s* %7 to i8*
  ; CHECK-NOT: %10 = bitcast %struct.tid_s* %7 to i8*
  ; CHECK: %14 = bitcast %struct.tid_s* %9 to i8*
  call void @llvm.lifetime.start(i64 24, i8* %10)
  %11 = getelementptr inbounds %struct.tid_s* %7, i64 0, i32 0
  ; CHECK-NOT: %11 = getelementptr inbounds %struct.tid_s* %7, i64 0, i32 0
  ; CHECK: %15 = getelementptr inbounds { i32, i32 addrspace(1)*, i32 addrspace(1)* }* %7, i64 0, i32 0
  store i32 %9, i32* %11, align 8
  %12 = getelementptr inbounds %struct.tid_s* %7, i64 0, i32 1
  ; CHECK-NOT: %12 = getelementptr inbounds %struct.tid_s* %7, i64 0, i32 1
  ; CHECK: %16 = getelementptr inbounds { i32, i32 addrspace(1)*, i32 addrspace(1)* }* %7, i64 0, i32 1
  store i32* %1, i32** %12, align 8
  %13 = getelementptr inbounds %struct.tid_s* %7, i64 0, i32 2
  ; CHECK-NOT: %13 = getelementptr inbounds %struct.tid_s* %7, i64 0, i32 2
  ; CHECK: %17 = getelementptr inbounds { i32, i32 addrspace(1)*, i32 addrspace(1)* }* %7, i64 0, i32 2
  store i32* %2, i32** %13, align 8
  %14 = bitcast %struct.tid_s* %6 to i8*
  call void @llvm.lifetime.start(i64 24, i8* %14)
  %tmp = bitcast %struct.tid_s* %6 to i8*
  ; CHECK: %19 = bitcast { i32, i32 addrspace(1)*, i32 addrspace(1)* }* %7 to i8*
  ; CHECK: %20 = bitcast { i32, i32*, i32 addrspace(1)* }* %8 to i8*
  %tmp1 = bitcast %struct.tid_s* %7 to i8*
  ; CHECK-NOT: %tmp1 = bitcast %struct.tid_s* %7 to i8*
  ; CHECK: %tmp1 = bitcast %struct.tid_s* %9 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp, i8* %tmp1, i64 24, i32 1, i1 false)
  ; CHECK-NOT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* tmp, i8* tmp1, i64 24, i32 1, i1 false)
  ; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp, i8* %19, i64 24, i32 1, i1 false)
  %15 = getelementptr inbounds %struct.tid_s* %6, i64 0, i32 0
  %16 = load i32* %15, align 8, !tbaa !17
  %17 = icmp slt i32 %16, %4
  br i1 %17, label %18, label %foo.exit

; <label>:18                                      ; preds = %5
  %19 = sext i32 %16 to i64
  %20 = getelementptr inbounds i32* %3, i64 %19
  store i32 %16, i32* %20, align 4, !tbaa !12
  br label %foo.exit

foo.exit:                                         ; preds = %18, %5
  %21 = bitcast %struct.tid_s* %6 to i8*
  call void @llvm.lifetime.end(i64 24, i8* %21)
  call void @llvm.lifetime.end(i64 24, i8* %10)
  ret void
}

; Function Attrs: nounwind
define internal void @_GLOBAL__sub_I_test.cpp() #3 section ".text.startup" {
  store %"class.std::__1::__tree_node_base"* null, %"class.std::__1::__tree_node_base"** getelementptr inbounds (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0), align 8, !tbaa !38
  store i64 0, i64* getelementptr inbounds (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE, i64 0, i32 0, i32 2, i32 0, i32 0), align 8, !tbaa !40
  store %"class.std::__1::__tree_node"* bitcast (%"class.std::__1::__tree_end_node"* getelementptr inbounds (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE, i64 0, i32 0, i32 1, i32 0, i32 0) to %"class.std::__1::__tree_node"*), %"class.std::__1::__tree_node"** getelementptr inbounds (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE, i64 0, i32 0, i32 0), align 8, !tbaa !21
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::__1::set"*)* @_ZNSt3__13setINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEED2Ev to void (i8*)*), i8* bitcast (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE to i8*), i8* @__dso_handle) #3
  ret void
}

; Function Attrs: nounwind
define internal void @_GLOBAL__sub_I_test_camp.cpp() #3 section ".text.startup" {
  store %"class.std::__1::__tree_node_base"* null, %"class.std::__1::__tree_node_base"** getelementptr inbounds (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE7, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0), align 8, !tbaa !38
  store i64 0, i64* getelementptr inbounds (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE7, i64 0, i32 0, i32 2, i32 0, i32 0), align 8, !tbaa !40
  store %"class.std::__1::__tree_node"* bitcast (%"class.std::__1::__tree_end_node"* getelementptr inbounds (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE7, i64 0, i32 0, i32 1, i32 0, i32 0) to %"class.std::__1::__tree_node"*), %"class.std::__1::__tree_node"** getelementptr inbounds (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE7, i64 0, i32 0, i32 0), align 8, !tbaa !21
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::__1::set"*)* @_ZNSt3__13setINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEED2Ev to void (i8*)*), i8* bitcast (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE7 to i8*), i8* @__dso_handle) #3
  ret void
}

; Function Attrs: noinline nounwind readnone uwtable
define internal void @_ZN2hc17parallel_for_eachIN12_GLOBAL__N_18foo_funcEEENS_17completion_futureERKNS_16accelerator_viewERKNS_6extentILi1EEERKT_(%"class.hc::completion_future"* noalias nocapture sret %agg.result, %"class.hc::accelerator_view"* nocapture dereferenceable(16) %av, %"class.hc::extent"* nocapture dereferenceable(8) %compute_domain, %"struct.(anonymous namespace)::foo_func"* nocapture dereferenceable(40) %f) #12 {
  ret void
}

; Function Attrs: uwtable
define linkonce_odr void @_ZN6Kalmar12KalmarDevice17get_default_queueEv(%"class.std::__1::shared_ptr.13"* noalias nocapture sret %agg.result, %"class.Kalmar::KalmarDevice"* %this) #4 align 2 {
  %tid = alloca %"class.std::__1::__thread_id", align 8
  %1 = alloca %"class.std::__1::shared_ptr.13", align 8
  %2 = tail call i64 @pthread_self() #15
  %3 = getelementptr %"class.std::__1::__thread_id"* %tid, i64 0, i32 0
  store i64 %2, i64* %3, align 8
  %4 = getelementptr inbounds %"class.Kalmar::KalmarDevice"* %this, i64 0, i32 3
  tail call void @_ZNSt3__15mutex4lockEv(%"class.std::__1::mutex"* %4) #17
  %5 = getelementptr inbounds %"class.Kalmar::KalmarDevice"* %this, i64 0, i32 2
  %6 = getelementptr inbounds %"class.Kalmar::KalmarDevice"* %this, i64 0, i32 2, i32 0, i32 1, i32 0, i32 0, i32 0
  %7 = load %"class.std::__1::__tree_node_base"** %6, align 8, !tbaa !38
  %8 = getelementptr inbounds %"class.Kalmar::KalmarDevice"* %this, i64 0, i32 2, i32 0, i32 1, i32 0, i32 0
  %9 = bitcast %"class.std::__1::__tree_end_node"* %8 to %"class.std::__1::__tree_node.15"*
  %10 = icmp eq %"class.std::__1::__tree_node_base"* %7, null
  br i1 %10, label %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE4findERSA_.exit.thread, label %.lr.ph.i.i.i.preheader

.lr.ph.i.i.i.preheader:                           ; preds = %0
  br label %.lr.ph.i.i.i

.lr.ph.i.i.i:                                     ; preds = %.outer.i.i.i, %.lr.ph.i.i.i.preheader
  %.0.ph6.i.i.i = phi %"class.std::__1::__tree_node.15"* [ %.013.i.le.i.i, %.outer.i.i.i ], [ %9, %.lr.ph.i.i.i.preheader ]
  %.01.ph5.i.in.i.i = phi %"class.std::__1::__tree_node_base"* [ %17, %.outer.i.i.i ], [ %7, %.lr.ph.i.i.i.preheader ]
  br label %11

; <label>:11                                      ; preds = %19, %.lr.ph.i.i.i
  %.013.i.in.i.i = phi %"class.std::__1::__tree_node_base"* [ %.01.ph5.i.in.i.i, %.lr.ph.i.i.i ], [ %21, %19 ]
  %12 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %.013.i.in.i.i, i64 1
  %13 = bitcast %"class.std::__1::__tree_node_base"* %12 to i64*
  %14 = load i64* %13, align 8
  %15 = icmp ult i64 %14, %2
  br i1 %15, label %19, label %.outer.i.i.i

.outer.i.i.i:                                     ; preds = %11
  %.013.i.in.i.i.lcssa = phi %"class.std::__1::__tree_node_base"* [ %.013.i.in.i.i, %11 ]
  %.013.i.le.i.i = bitcast %"class.std::__1::__tree_node_base"* %.013.i.in.i.i.lcssa to %"class.std::__1::__tree_node.15"*
  %16 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %.013.i.in.i.i.lcssa, i64 0, i32 0, i32 0
  %17 = load %"class.std::__1::__tree_node_base"** %16, align 8, !tbaa !38
  %18 = icmp eq %"class.std::__1::__tree_node_base"* %17, null
  br i1 %18, label %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i.loopexit10, label %.lr.ph.i.i.i

; <label>:19                                      ; preds = %11
  %20 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %.013.i.in.i.i, i64 0, i32 1
  %21 = load %"class.std::__1::__tree_node_base"** %20, align 8, !tbaa !43
  %22 = icmp eq %"class.std::__1::__tree_node_base"* %21, null
  br i1 %22, label %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i.loopexit, label %11

_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i.loopexit: ; preds = %19
  %.0.ph6.i.i.i.lcssa12 = phi %"class.std::__1::__tree_node.15"* [ %.0.ph6.i.i.i, %19 ]
  br label %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i

_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i.loopexit10: ; preds = %.outer.i.i.i
  %.013.i.le.i.i.lcssa = phi %"class.std::__1::__tree_node.15"* [ %.013.i.le.i.i, %.outer.i.i.i ]
  br label %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i

_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i: ; preds = %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i.loopexit10, %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i.loopexit
  %.0.ph.lcssa.i.i.i = phi %"class.std::__1::__tree_node.15"* [ %.0.ph6.i.i.i.lcssa12, %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i.loopexit ], [ %.013.i.le.i.i.lcssa, %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i.loopexit10 ]
  %23 = icmp eq %"class.std::__1::__tree_node.15"* %.0.ph.lcssa.i.i.i, %9
  br i1 %23, label %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE4findERSA_.exit.thread, label %24

; <label>:24                                      ; preds = %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i
  %25 = getelementptr inbounds %"class.std::__1::__tree_node.15"* %.0.ph.lcssa.i.i.i, i64 0, i32 1, i32 0, i32 0, i32 0
  %26 = load i64* %25, align 8
  %27 = icmp ult i64 %2, %26
  br i1 %27, label %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE4findERSA_.exit.thread, label %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEED2Ev.exit

_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE4findERSA_.exit.thread: ; preds = %24, %_ZNSt3__16__treeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEENS_19__map_value_compareIS2_S7_NS_4lessIS2_EELb1EEENS_9allocatorIS7_EEE13__lower_boundIS2_EENS_15__tree_iteratorIS7_PNS_11__tree_nodeIS7_PvEElEERKT_SK_SK_.exit.i.i, %0
  %28 = call dereferenceable(16) %"class.std::__1::shared_ptr.13"* @_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEEixERSA_(%"class.std::__1::map"* %5, %"class.std::__1::__thread_id"* dereferenceable(8) %tid) #17
  %29 = bitcast %"class.Kalmar::KalmarDevice"* %this to void (%"class.std::__1::shared_ptr.13"*, %"class.Kalmar::KalmarDevice"*)***
  %30 = load void (%"class.std::__1::shared_ptr.13"*, %"class.Kalmar::KalmarDevice"*)*** %29, align 8, !tbaa !46
  %31 = getelementptr inbounds void (%"class.std::__1::shared_ptr.13"*, %"class.Kalmar::KalmarDevice"*)** %30, i64 11
  %32 = load void (%"class.std::__1::shared_ptr.13"*, %"class.Kalmar::KalmarDevice"*)** %31, align 8
  call void %32(%"class.std::__1::shared_ptr.13"* sret %1, %"class.Kalmar::KalmarDevice"* %this) #17
  %33 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %1, i64 0, i32 0
  %34 = load %"class.Kalmar::KalmarQueue"** %33, align 8, !tbaa !31
  %35 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %1, i64 0, i32 1
  %36 = load %"class.std::__1::__shared_weak_count"** %35, align 8, !tbaa !22
  store %"class.Kalmar::KalmarQueue"* null, %"class.Kalmar::KalmarQueue"** %33, align 8, !tbaa !31
  store %"class.std::__1::__shared_weak_count"* null, %"class.std::__1::__shared_weak_count"** %35, align 8, !tbaa !22
  %37 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %28, i64 0, i32 0
  store %"class.Kalmar::KalmarQueue"* %34, %"class.Kalmar::KalmarQueue"** %37, align 8, !tbaa !21
  %38 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %28, i64 0, i32 1
  %39 = load %"class.std::__1::__shared_weak_count"** %38, align 8, !tbaa !21
  store %"class.std::__1::__shared_weak_count"* %36, %"class.std::__1::__shared_weak_count"** %38, align 8, !tbaa !21
  %40 = icmp eq %"class.std::__1::__shared_weak_count"* %39, null
  br i1 %40, label %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEEaSEOS3_.exit, label %41

; <label>:41                                      ; preds = %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE4findERSA_.exit.thread
  call void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"* %39) #18
  br label %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEEaSEOS3_.exit

_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEEaSEOS3_.exit: ; preds = %41, %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE4findERSA_.exit.thread
  %42 = load %"class.std::__1::__shared_weak_count"** %35, align 8, !tbaa !22
  %43 = icmp eq %"class.std::__1::__shared_weak_count"* %42, null
  br i1 %43, label %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEED2Ev.exit, label %44

; <label>:44                                      ; preds = %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEEaSEOS3_.exit
  call void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"* %42) #18
  br label %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEED2Ev.exit

_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEED2Ev.exit: ; preds = %44, %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEEaSEOS3_.exit, %24
  %45 = call dereferenceable(16) %"class.std::__1::shared_ptr.13"* @_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEEixERSA_(%"class.std::__1::map"* %5, %"class.std::__1::__thread_id"* dereferenceable(8) %tid) #17
  %46 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %agg.result, i64 0, i32 0
  %47 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %45, i64 0, i32 0
  %48 = load %"class.Kalmar::KalmarQueue"** %47, align 8, !tbaa !31
  store %"class.Kalmar::KalmarQueue"* %48, %"class.Kalmar::KalmarQueue"** %46, align 8, !tbaa !31
  %49 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %agg.result, i64 0, i32 1
  %50 = getelementptr inbounds %"class.std::__1::shared_ptr.13"* %45, i64 0, i32 1
  %51 = load %"class.std::__1::__shared_weak_count"** %50, align 8, !tbaa !22
  store %"class.std::__1::__shared_weak_count"* %51, %"class.std::__1::__shared_weak_count"** %49, align 8, !tbaa !22
  %52 = icmp eq %"class.std::__1::__shared_weak_count"* %51, null
  br i1 %52, label %54, label %53

; <label>:53                                      ; preds = %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEED2Ev.exit
  call void @_ZNSt3__119__shared_weak_count12__add_sharedEv(%"class.std::__1::__shared_weak_count"* %51) #18
  br label %54

; <label>:54                                      ; preds = %53, %_ZNSt3__110shared_ptrIN6Kalmar11KalmarQueueEED2Ev.exit
  call void @_ZNSt3__15mutex6unlockEv(%"class.std::__1::mutex"* %4) #18
  ret void
}

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) #13 {
  %2 = tail call i8* @__cxa_begin_catch(i8* %0) #3
  tail call void @_ZSt9terminatev() #20
  unreachable
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZNSt3__13setINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEED2Ev(%"class.std::__1::set"* readonly %this) unnamed_addr #14 align 2 {
  %1 = getelementptr inbounds %"class.std::__1::set"* %this, i64 0, i32 0
  %2 = getelementptr inbounds %"class.std::__1::set"* %this, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %3 = load %"class.std::__1::__tree_node_base"** %2, align 8, !tbaa !38
  %4 = bitcast %"class.std::__1::__tree_node_base"* %3 to %"class.std::__1::__tree_node"*
  tail call void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEE7destroyEPNS_11__tree_nodeIS6_PvEE(%"class.std::__1::__tree"* %1, %"class.std::__1::__tree_node"* %4) #18
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dereferenceable(16) %"class.std::__1::shared_ptr.13"* @_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEEixERSA_(%"class.std::__1::map"* %this, %"class.std::__1::__thread_id"* nocapture readonly dereferenceable(8) %__k) #4 align 2 {
  %__parent = alloca %"class.std::__1::__tree_node_base"*, align 8
  %1 = getelementptr inbounds %"class.std::__1::map"* %this, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %2 = load %"class.std::__1::__tree_node_base"** %1, align 8, !tbaa !38
  %3 = icmp eq %"class.std::__1::__tree_node_base"* %2, null
  br i1 %3, label %23, label %.preheader.i

.preheader.i:                                     ; preds = %0
  %4 = getelementptr inbounds %"class.std::__1::__thread_id"* %__k, i64 0, i32 0
  %5 = load i64* %4, align 8
  br label %.backedge.i

.backedge.i:                                      ; preds = %.backedge.i.backedge, %.preheader.i
  %__nd.0.in.i = phi %"class.std::__1::__tree_node_base"* [ %2, %.preheader.i ], [ %__nd.0.in.i.be, %.backedge.i.backedge ]
  %6 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %__nd.0.in.i, i64 1
  %7 = bitcast %"class.std::__1::__tree_node_base"* %6 to i64*
  %8 = load i64* %7, align 8
  %9 = icmp ult i64 %5, %8
  br i1 %9, label %10, label %15

; <label>:10                                      ; preds = %.backedge.i
  %11 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %__nd.0.in.i, i64 0, i32 0, i32 0
  %12 = load %"class.std::__1::__tree_node_base"** %11, align 8, !tbaa !38
  %13 = icmp eq %"class.std::__1::__tree_node_base"* %12, null
  br i1 %13, label %14, label %.backedge.i.backedge

.backedge.i.backedge:                             ; preds = %17, %10
  %__nd.0.in.i.be = phi %"class.std::__1::__tree_node_base"* [ %12, %10 ], [ %19, %17 ]
  br label %.backedge.i

; <label>:14                                      ; preds = %10
  %.lcssa12 = phi %"class.std::__1::__tree_node_base"** [ %11, %10 ]
  %__nd.0.in.i.lcssa11 = phi %"class.std::__1::__tree_node_base"* [ %__nd.0.in.i, %10 ]
  store %"class.std::__1::__tree_node_base"* %__nd.0.in.i.lcssa11, %"class.std::__1::__tree_node_base"** %__parent, align 8
  br label %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE16__find_equal_keyERPNS_16__tree_node_baseIPvEERSA_.exit

; <label>:15                                      ; preds = %.backedge.i
  %16 = icmp ult i64 %8, %5
  br i1 %16, label %17, label %22

; <label>:17                                      ; preds = %15
  %18 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %__nd.0.in.i, i64 0, i32 1
  %19 = load %"class.std::__1::__tree_node_base"** %18, align 8, !tbaa !43
  %20 = icmp eq %"class.std::__1::__tree_node_base"* %19, null
  br i1 %20, label %21, label %.backedge.i.backedge

; <label>:21                                      ; preds = %17
  %.lcssa = phi %"class.std::__1::__tree_node_base"** [ %18, %17 ]
  %__nd.0.in.i.lcssa10 = phi %"class.std::__1::__tree_node_base"* [ %__nd.0.in.i, %17 ]
  store %"class.std::__1::__tree_node_base"* %__nd.0.in.i.lcssa10, %"class.std::__1::__tree_node_base"** %__parent, align 8
  br label %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE16__find_equal_keyERPNS_16__tree_node_baseIPvEERSA_.exit

; <label>:22                                      ; preds = %15
  %__nd.0.in.i.lcssa = phi %"class.std::__1::__tree_node_base"* [ %__nd.0.in.i, %15 ]
  store %"class.std::__1::__tree_node_base"* %__nd.0.in.i.lcssa, %"class.std::__1::__tree_node_base"** %__parent, align 8
  br label %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE16__find_equal_keyERPNS_16__tree_node_baseIPvEERSA_.exit

; <label>:23                                      ; preds = %0
  %24 = getelementptr inbounds %"class.std::__1::map"* %this, i64 0, i32 0, i32 1, i32 0, i32 0
  %25 = bitcast %"class.std::__1::__tree_end_node"* %24 to %"class.std::__1::__tree_node_base"*
  store %"class.std::__1::__tree_node_base"* %25, %"class.std::__1::__tree_node_base"** %__parent, align 8
  %26 = getelementptr inbounds %"class.std::__1::__tree_end_node"* %24, i64 0, i32 0
  br label %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE16__find_equal_keyERPNS_16__tree_node_baseIPvEERSA_.exit

_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE16__find_equal_keyERPNS_16__tree_node_baseIPvEERSA_.exit: ; preds = %23, %22, %21, %14
  %27 = phi %"class.std::__1::__tree_node_base"* [ %__nd.0.in.i.lcssa11, %14 ], [ %__nd.0.in.i.lcssa10, %21 ], [ %__nd.0.in.i.lcssa, %22 ], [ %25, %23 ]
  %.0.i = phi %"class.std::__1::__tree_node_base"** [ %.lcssa12, %14 ], [ %.lcssa, %21 ], [ %__parent, %22 ], [ %26, %23 ]
  %28 = load %"class.std::__1::__tree_node_base"** %.0.i, align 8, !tbaa !21
  %29 = bitcast %"class.std::__1::__tree_node_base"* %28 to %"class.std::__1::__tree_node.15"*
  %30 = icmp eq %"class.std::__1::__tree_node_base"* %28, null
  br i1 %30, label %31, label %60

; <label>:31                                      ; preds = %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE16__find_equal_keyERPNS_16__tree_node_baseIPvEERSA_.exit
  %32 = tail call noalias i8* @_Znwm(i64 56) #17
  %33 = bitcast i8* %32 to %"class.std::__1::__tree_node.15"*
  %34 = getelementptr inbounds i8* %32, i64 32
  %35 = getelementptr inbounds %"class.std::__1::__thread_id"* %__k, i64 0, i32 0
  %36 = load i64* %35, align 8, !tbaa !48
  %37 = bitcast i8* %34 to i64*
  store i64 %36, i64* %37, align 8, !tbaa !48
  %38 = getelementptr inbounds i8* %32, i64 40
  %39 = bitcast i8* %38 to %"class.Kalmar::KalmarQueue"**
  store %"class.Kalmar::KalmarQueue"* null, %"class.Kalmar::KalmarQueue"** %39, align 8, !tbaa !31
  %40 = getelementptr inbounds i8* %32, i64 48
  %41 = bitcast i8* %40 to %"class.std::__1::__shared_weak_count"**
  store %"class.std::__1::__shared_weak_count"* null, %"class.std::__1::__shared_weak_count"** %41, align 8, !tbaa !22
  %42 = bitcast i8* %32 to %"class.std::__1::__tree_node_base"*
  %43 = bitcast i8* %32 to %"class.std::__1::__tree_node_base"**
  store %"class.std::__1::__tree_node_base"* null, %"class.std::__1::__tree_node_base"** %43, align 8, !tbaa !38
  %44 = getelementptr inbounds i8* %32, i64 8
  %45 = bitcast i8* %44 to %"class.std::__1::__tree_node_base"**
  store %"class.std::__1::__tree_node_base"* null, %"class.std::__1::__tree_node_base"** %45, align 8, !tbaa !43
  %46 = getelementptr inbounds i8* %32, i64 16
  %47 = bitcast i8* %46 to %"class.std::__1::__tree_node_base"**
  store %"class.std::__1::__tree_node_base"* %27, %"class.std::__1::__tree_node_base"** %47, align 8, !tbaa !49
  store %"class.std::__1::__tree_node_base"* %42, %"class.std::__1::__tree_node_base"** %.0.i, align 8, !tbaa !21
  %48 = getelementptr inbounds %"class.std::__1::map"* %this, i64 0, i32 0, i32 0
  %49 = load %"class.std::__1::__tree_node.15"** %48, align 8, !tbaa !21
  %50 = getelementptr inbounds %"class.std::__1::__tree_node.15"* %49, i64 0, i32 0, i32 0, i32 0
  %51 = load %"class.std::__1::__tree_node_base"** %50, align 8, !tbaa !38
  %52 = icmp eq %"class.std::__1::__tree_node_base"* %51, null
  br i1 %52, label %_ZNSt3__110unique_ptrINS_11__tree_nodeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEEPvEENS_21__map_node_destructorINS_9allocatorISA_EEEEED2Ev.exit, label %53

; <label>:53                                      ; preds = %31
  %54 = bitcast %"class.std::__1::__tree_node_base"* %51 to %"class.std::__1::__tree_node.15"*
  store %"class.std::__1::__tree_node.15"* %54, %"class.std::__1::__tree_node.15"** %48, align 8, !tbaa !21
  %.pre.i = load %"class.std::__1::__tree_node_base"** %.0.i, align 8, !tbaa !21
  br label %_ZNSt3__110unique_ptrINS_11__tree_nodeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEEPvEENS_21__map_node_destructorINS_9allocatorISA_EEEEED2Ev.exit

_ZNSt3__110unique_ptrINS_11__tree_nodeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEEPvEENS_21__map_node_destructorINS_9allocatorISA_EEEEED2Ev.exit: ; preds = %53, %31
  %55 = phi %"class.std::__1::__tree_node_base"* [ %42, %31 ], [ %.pre.i, %53 ]
  %56 = load %"class.std::__1::__tree_node_base"** %1, align 8, !tbaa !38
  tail call void @_ZNSt3__127__tree_balance_after_insertIPNS_16__tree_node_baseIPvEEEEvT_S5_(%"class.std::__1::__tree_node_base"* %56, %"class.std::__1::__tree_node_base"* %55) #18
  %57 = getelementptr inbounds %"class.std::__1::map"* %this, i64 0, i32 0, i32 2, i32 0, i32 0
  %58 = load i64* %57, align 8, !tbaa !48
  %59 = add i64 %58, 1
  store i64 %59, i64* %57, align 8, !tbaa !48
  br label %60

; <label>:60                                      ; preds = %_ZNSt3__110unique_ptrINS_11__tree_nodeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEEPvEENS_21__map_node_destructorINS_9allocatorISA_EEEEED2Ev.exit, %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE16__find_equal_keyERPNS_16__tree_node_baseIPvEERSA_.exit
  %__r.0 = phi %"class.std::__1::__tree_node.15"* [ %33, %_ZNSt3__110unique_ptrINS_11__tree_nodeINS_12__value_typeINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEEEEPvEENS_21__map_node_destructorINS_9allocatorISA_EEEEED2Ev.exit ], [ %29, %_ZNSt3__13mapINS_11__thread_idENS_10shared_ptrIN6Kalmar11KalmarQueueEEENS_4lessIS1_EENS_9allocatorINS_4pairIKS1_S5_EEEEE16__find_equal_keyERPNS_16__tree_node_baseIPvEERSA_.exit ]
  %61 = getelementptr inbounds %"class.std::__1::__tree_node.15"* %__r.0, i64 0, i32 1, i32 0, i32 1
  ret %"class.std::__1::shared_ptr.13"* %61
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEE7destroyEPNS_11__tree_nodeIS6_PvEE(%"class.std::__1::__tree"* readnone %this, %"class.std::__1::__tree_node"* %__nd) #11 align 2 {
  %1 = icmp eq %"class.std::__1::__tree_node"* %__nd, null
  br i1 %1, label %18, label %2

; <label>:2                                       ; preds = %0
  %3 = getelementptr inbounds %"class.std::__1::__tree_node"* %__nd, i64 0, i32 0, i32 0, i32 0
  %4 = load %"class.std::__1::__tree_node_base"** %3, align 8, !tbaa !38
  %5 = bitcast %"class.std::__1::__tree_node_base"* %4 to %"class.std::__1::__tree_node"*
  tail call void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEE7destroyEPNS_11__tree_nodeIS6_PvEE(%"class.std::__1::__tree"* %this, %"class.std::__1::__tree_node"* %5) #18
  %6 = getelementptr inbounds %"class.std::__1::__tree_node"* %__nd, i64 0, i32 0, i32 1
  %7 = load %"class.std::__1::__tree_node_base"** %6, align 8, !tbaa !43
  %8 = bitcast %"class.std::__1::__tree_node_base"* %7 to %"class.std::__1::__tree_node"*
  tail call void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEE7destroyEPNS_11__tree_nodeIS6_PvEE(%"class.std::__1::__tree"* %this, %"class.std::__1::__tree_node"* %8) #18
  %9 = getelementptr inbounds %"class.std::__1::__tree_node"* %__nd, i64 0, i32 1
  %10 = bitcast %"class.std::__1::basic_string"* %9 to i8*
  %11 = load i8* %10, align 1, !tbaa !50
  %12 = and i8 %11, 1
  %13 = icmp eq i8 %12, 0
  br i1 %13, label %_ZNSt3__116allocator_traitsINS_9allocatorINS_11__tree_nodeINS_12basic_stringIcNS_11char_traitsIcEENS1_IcEEEEPvEEEEE7destroyIS7_EEvRSA_PT_.exit, label %14

; <label>:14                                      ; preds = %2
  %15 = getelementptr inbounds %"class.std::__1::__tree_node"* %__nd, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 2
  %16 = load i8** %15, align 8, !tbaa !51
  tail call void @_ZdlPv(i8* %16) #18
  br label %_ZNSt3__116allocator_traitsINS_9allocatorINS_11__tree_nodeINS_12basic_stringIcNS_11char_traitsIcEENS1_IcEEEEPvEEEEE7destroyIS7_EEvRSA_PT_.exit

_ZNSt3__116allocator_traitsINS_9allocatorINS_11__tree_nodeINS_12basic_stringIcNS_11char_traitsIcEENS1_IcEEEEPvEEEEE7destroyIS7_EEvRSA_PT_.exit: ; preds = %14, %2
  %17 = bitcast %"class.std::__1::__tree_node"* %__nd to i8*
  tail call void @_ZdlPv(i8* %17) #18
  br label %18

; <label>:18                                      ; preds = %_ZNSt3__116allocator_traitsINS_9allocatorINS_11__tree_nodeINS_12basic_stringIcNS_11char_traitsIcEENS1_IcEEEEPvEEEEE7destroyIS7_EEvRSA_PT_.exit, %0
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZNSt3__127__tree_balance_after_insertIPNS_16__tree_node_baseIPvEEEEvT_S5_(%"class.std::__1::__tree_node_base"* readnone %__root, %"class.std::__1::__tree_node_base"* %__x) #11 {
  %1 = icmp eq %"class.std::__1::__tree_node_base"* %__x, %__root
  %2 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %__x, i64 0, i32 3
  %3 = zext i1 %1 to i8
  store i8 %3, i8* %2, align 1, !tbaa !53
  br i1 %1, label %.critedge, label %.lr.ph.preheader

.lr.ph.preheader:                                 ; preds = %0
  br label %.lr.ph

.lr.ph:                                           ; preds = %.backedge, %.lr.ph.preheader
  %.07 = phi %"class.std::__1::__tree_node_base"* [ %11, %.backedge ], [ %__x, %.lr.ph.preheader ]
  %4 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %.07, i64 0, i32 2
  %5 = load %"class.std::__1::__tree_node_base"** %4, align 8, !tbaa !49
  %6 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %5, i64 0, i32 3
  %7 = load i8* %6, align 1, !tbaa !53, !range !54
  %8 = icmp eq i8 %7, 0
  br i1 %8, label %9, label %.critedge.loopexit

; <label>:9                                       ; preds = %.lr.ph
  %10 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %5, i64 0, i32 2
  %11 = load %"class.std::__1::__tree_node_base"** %10, align 8, !tbaa !49
  %12 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %11, i64 0, i32 0, i32 0
  %13 = load %"class.std::__1::__tree_node_base"** %12, align 8, !tbaa !38
  %14 = icmp eq %"class.std::__1::__tree_node_base"* %13, %5
  br i1 %14, label %15, label %72

; <label>:15                                      ; preds = %9
  %16 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %11, i64 0, i32 1
  %17 = load %"class.std::__1::__tree_node_base"** %16, align 8, !tbaa !43
  %18 = icmp eq %"class.std::__1::__tree_node_base"* %17, null
  br i1 %18, label %27, label %19

; <label>:19                                      ; preds = %15
  %20 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %17, i64 0, i32 3
  %21 = load i8* %20, align 1, !tbaa !53, !range !54
  %22 = icmp eq i8 %21, 0
  br i1 %22, label %23, label %27

; <label>:23                                      ; preds = %19
  store i8 1, i8* %6, align 1, !tbaa !53
  %24 = icmp eq %"class.std::__1::__tree_node_base"* %11, %__root
  %25 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %11, i64 0, i32 3
  %26 = zext i1 %24 to i8
  store i8 %26, i8* %25, align 1, !tbaa !53
  store i8 1, i8* %20, align 1, !tbaa !53
  br label %.backedge

; <label>:27                                      ; preds = %19, %15
  %.lcssa36 = phi %"class.std::__1::__tree_node_base"* [ %11, %19 ], [ %11, %15 ]
  %.lcssa34 = phi %"class.std::__1::__tree_node_base"** [ %10, %19 ], [ %10, %15 ]
  %.lcssa33 = phi %"class.std::__1::__tree_node_base"* [ %5, %19 ], [ %5, %15 ]
  %.07.lcssa29 = phi %"class.std::__1::__tree_node_base"* [ %.07, %19 ], [ %.07, %15 ]
  %28 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %.lcssa33, i64 0, i32 0, i32 0
  %29 = load %"class.std::__1::__tree_node_base"** %28, align 8, !tbaa !38
  %30 = icmp eq %"class.std::__1::__tree_node_base"* %29, %.07.lcssa29
  br i1 %30, label %49, label %31

; <label>:31                                      ; preds = %27
  %32 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %.lcssa33, i64 0, i32 1
  %33 = load %"class.std::__1::__tree_node_base"** %32, align 8, !tbaa !43
  %34 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %33, i64 0, i32 0, i32 0
  %35 = load %"class.std::__1::__tree_node_base"** %34, align 8, !tbaa !38
  store %"class.std::__1::__tree_node_base"* %35, %"class.std::__1::__tree_node_base"** %32, align 8, !tbaa !43
  %36 = icmp eq %"class.std::__1::__tree_node_base"* %35, null
  br i1 %36, label %39, label %37

; <label>:37                                      ; preds = %31
  %38 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %35, i64 0, i32 2
  store %"class.std::__1::__tree_node_base"* %.lcssa33, %"class.std::__1::__tree_node_base"** %38, align 8, !tbaa !49
  %.pre16 = load %"class.std::__1::__tree_node_base"** %.lcssa34, align 8, !tbaa !49
  br label %39

; <label>:39                                      ; preds = %37, %31
  %40 = phi %"class.std::__1::__tree_node_base"* [ %.pre16, %37 ], [ %.lcssa36, %31 ]
  %41 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %33, i64 0, i32 2
  store %"class.std::__1::__tree_node_base"* %40, %"class.std::__1::__tree_node_base"** %41, align 8, !tbaa !49
  %42 = load %"class.std::__1::__tree_node_base"** %.lcssa34, align 8, !tbaa !49
  %43 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %42, i64 0, i32 0, i32 0
  %44 = load %"class.std::__1::__tree_node_base"** %43, align 8, !tbaa !38
  %45 = icmp eq %"class.std::__1::__tree_node_base"* %44, %.lcssa33
  br i1 %45, label %46, label %47

; <label>:46                                      ; preds = %39
  store %"class.std::__1::__tree_node_base"* %33, %"class.std::__1::__tree_node_base"** %43, align 8, !tbaa !38
  br label %_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit3

; <label>:47                                      ; preds = %39
  %48 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %42, i64 0, i32 1
  store %"class.std::__1::__tree_node_base"* %33, %"class.std::__1::__tree_node_base"** %48, align 8, !tbaa !43
  br label %_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit3

_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit3: ; preds = %47, %46
  store %"class.std::__1::__tree_node_base"* %.lcssa33, %"class.std::__1::__tree_node_base"** %34, align 8, !tbaa !38
  store %"class.std::__1::__tree_node_base"* %33, %"class.std::__1::__tree_node_base"** %.lcssa34, align 8, !tbaa !49
  %.pre = load %"class.std::__1::__tree_node_base"** %41, align 8, !tbaa !49
  %.phi.trans.insert = getelementptr inbounds %"class.std::__1::__tree_node_base"* %.pre, i64 0, i32 0, i32 0
  %.pre15 = load %"class.std::__1::__tree_node_base"** %.phi.trans.insert, align 8, !tbaa !38
  br label %49

; <label>:49                                      ; preds = %_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit3, %27
  %50 = phi %"class.std::__1::__tree_node_base"* [ %.lcssa33, %27 ], [ %.pre15, %_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit3 ]
  %51 = phi %"class.std::__1::__tree_node_base"* [ %.lcssa36, %27 ], [ %.pre, %_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit3 ]
  %52 = phi %"class.std::__1::__tree_node_base"* [ %.lcssa33, %27 ], [ %33, %_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit3 ]
  %53 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %52, i64 0, i32 3
  store i8 1, i8* %53, align 1, !tbaa !53
  %54 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %51, i64 0, i32 3
  store i8 0, i8* %54, align 1, !tbaa !53
  %55 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %51, i64 0, i32 0, i32 0
  %56 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %50, i64 0, i32 1
  %57 = load %"class.std::__1::__tree_node_base"** %56, align 8, !tbaa !43
  store %"class.std::__1::__tree_node_base"* %57, %"class.std::__1::__tree_node_base"** %55, align 8, !tbaa !38
  %58 = icmp eq %"class.std::__1::__tree_node_base"* %57, null
  br i1 %58, label %61, label %59

; <label>:59                                      ; preds = %49
  %60 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %57, i64 0, i32 2
  store %"class.std::__1::__tree_node_base"* %51, %"class.std::__1::__tree_node_base"** %60, align 8, !tbaa !49
  br label %61

; <label>:61                                      ; preds = %59, %49
  %62 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %51, i64 0, i32 2
  %63 = load %"class.std::__1::__tree_node_base"** %62, align 8, !tbaa !49
  %64 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %50, i64 0, i32 2
  store %"class.std::__1::__tree_node_base"* %63, %"class.std::__1::__tree_node_base"** %64, align 8, !tbaa !49
  %65 = load %"class.std::__1::__tree_node_base"** %62, align 8, !tbaa !49
  %66 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %65, i64 0, i32 0, i32 0
  %67 = load %"class.std::__1::__tree_node_base"** %66, align 8, !tbaa !38
  %68 = icmp eq %"class.std::__1::__tree_node_base"* %67, %51
  br i1 %68, label %69, label %70

; <label>:69                                      ; preds = %61
  store %"class.std::__1::__tree_node_base"* %50, %"class.std::__1::__tree_node_base"** %66, align 8, !tbaa !38
  br label %_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit2

; <label>:70                                      ; preds = %61
  %71 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %65, i64 0, i32 1
  store %"class.std::__1::__tree_node_base"* %50, %"class.std::__1::__tree_node_base"** %71, align 8, !tbaa !43
  br label %_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit2

_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit2: ; preds = %70, %69
  store %"class.std::__1::__tree_node_base"* %51, %"class.std::__1::__tree_node_base"** %56, align 8, !tbaa !43
  store %"class.std::__1::__tree_node_base"* %50, %"class.std::__1::__tree_node_base"** %62, align 8, !tbaa !49
  br label %.critedge

; <label>:72                                      ; preds = %9
  %73 = icmp eq %"class.std::__1::__tree_node_base"* %13, null
  br i1 %73, label %83, label %74

; <label>:74                                      ; preds = %72
  %75 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %13, i64 0, i32 3
  %76 = load i8* %75, align 1, !tbaa !53, !range !54
  %77 = icmp eq i8 %76, 0
  br i1 %77, label %78, label %83

; <label>:78                                      ; preds = %74
  store i8 1, i8* %6, align 1, !tbaa !53
  %79 = icmp eq %"class.std::__1::__tree_node_base"* %11, %__root
  %80 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %11, i64 0, i32 3
  %81 = zext i1 %79 to i8
  store i8 %81, i8* %80, align 1, !tbaa !53
  store i8 1, i8* %75, align 1, !tbaa !53
  br label %.backedge

.backedge:                                        ; preds = %78, %23
  %82 = icmp eq %"class.std::__1::__tree_node_base"* %11, %__root
  br i1 %82, label %.critedge.loopexit, label %.lr.ph

; <label>:83                                      ; preds = %74, %72
  %.lcssa35 = phi %"class.std::__1::__tree_node_base"* [ %11, %74 ], [ %11, %72 ]
  %.lcssa = phi %"class.std::__1::__tree_node_base"** [ %10, %74 ], [ %10, %72 ]
  %.lcssa32 = phi %"class.std::__1::__tree_node_base"* [ %5, %74 ], [ %5, %72 ]
  %.lcssa30 = phi %"class.std::__1::__tree_node_base"** [ %4, %74 ], [ %4, %72 ]
  %.07.lcssa28 = phi %"class.std::__1::__tree_node_base"* [ %.07, %74 ], [ %.07, %72 ]
  %84 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %.lcssa32, i64 0, i32 0, i32 0
  %85 = load %"class.std::__1::__tree_node_base"** %84, align 8, !tbaa !38
  %86 = icmp eq %"class.std::__1::__tree_node_base"* %85, %.07.lcssa28
  br i1 %86, label %87, label %102

; <label>:87                                      ; preds = %83
  %88 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %.07.lcssa28, i64 0, i32 1
  %89 = load %"class.std::__1::__tree_node_base"** %88, align 8, !tbaa !43
  store %"class.std::__1::__tree_node_base"* %89, %"class.std::__1::__tree_node_base"** %84, align 8, !tbaa !38
  %90 = icmp eq %"class.std::__1::__tree_node_base"* %89, null
  br i1 %90, label %93, label %91

; <label>:91                                      ; preds = %87
  %92 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %89, i64 0, i32 2
  store %"class.std::__1::__tree_node_base"* %.lcssa32, %"class.std::__1::__tree_node_base"** %92, align 8, !tbaa !49
  %.pre17 = load %"class.std::__1::__tree_node_base"** %.lcssa, align 8, !tbaa !49
  br label %93

; <label>:93                                      ; preds = %91, %87
  %94 = phi %"class.std::__1::__tree_node_base"* [ %.pre17, %91 ], [ %.lcssa35, %87 ]
  store %"class.std::__1::__tree_node_base"* %94, %"class.std::__1::__tree_node_base"** %.lcssa30, align 8, !tbaa !49
  %95 = load %"class.std::__1::__tree_node_base"** %.lcssa, align 8, !tbaa !49
  %96 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %95, i64 0, i32 0, i32 0
  %97 = load %"class.std::__1::__tree_node_base"** %96, align 8, !tbaa !38
  %98 = icmp eq %"class.std::__1::__tree_node_base"* %97, %.lcssa32
  br i1 %98, label %99, label %100

; <label>:99                                      ; preds = %93
  store %"class.std::__1::__tree_node_base"* %.07.lcssa28, %"class.std::__1::__tree_node_base"** %96, align 8, !tbaa !38
  br label %_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit

; <label>:100                                     ; preds = %93
  %101 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %95, i64 0, i32 1
  store %"class.std::__1::__tree_node_base"* %.07.lcssa28, %"class.std::__1::__tree_node_base"** %101, align 8, !tbaa !43
  br label %_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit

_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit: ; preds = %100, %99
  store %"class.std::__1::__tree_node_base"* %.lcssa32, %"class.std::__1::__tree_node_base"** %88, align 8, !tbaa !43
  store %"class.std::__1::__tree_node_base"* %.07.lcssa28, %"class.std::__1::__tree_node_base"** %.lcssa, align 8, !tbaa !49
  %.pre18 = load %"class.std::__1::__tree_node_base"** %.lcssa30, align 8, !tbaa !49
  br label %102

; <label>:102                                     ; preds = %_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit, %83
  %103 = phi %"class.std::__1::__tree_node_base"* [ %.pre18, %_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit ], [ %.lcssa35, %83 ]
  %104 = phi %"class.std::__1::__tree_node_base"* [ %.07.lcssa28, %_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit ], [ %.lcssa32, %83 ]
  %105 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %104, i64 0, i32 3
  store i8 1, i8* %105, align 1, !tbaa !53
  %106 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %103, i64 0, i32 3
  store i8 0, i8* %106, align 1, !tbaa !53
  %107 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %103, i64 0, i32 1
  %108 = load %"class.std::__1::__tree_node_base"** %107, align 8, !tbaa !43
  %109 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %108, i64 0, i32 0, i32 0
  %110 = load %"class.std::__1::__tree_node_base"** %109, align 8, !tbaa !38
  store %"class.std::__1::__tree_node_base"* %110, %"class.std::__1::__tree_node_base"** %107, align 8, !tbaa !43
  %111 = icmp eq %"class.std::__1::__tree_node_base"* %110, null
  br i1 %111, label %114, label %112

; <label>:112                                     ; preds = %102
  %113 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %110, i64 0, i32 2
  store %"class.std::__1::__tree_node_base"* %103, %"class.std::__1::__tree_node_base"** %113, align 8, !tbaa !49
  br label %114

; <label>:114                                     ; preds = %112, %102
  %115 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %103, i64 0, i32 2
  %116 = load %"class.std::__1::__tree_node_base"** %115, align 8, !tbaa !49
  %117 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %108, i64 0, i32 2
  store %"class.std::__1::__tree_node_base"* %116, %"class.std::__1::__tree_node_base"** %117, align 8, !tbaa !49
  %118 = load %"class.std::__1::__tree_node_base"** %115, align 8, !tbaa !49
  %119 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %118, i64 0, i32 0, i32 0
  %120 = load %"class.std::__1::__tree_node_base"** %119, align 8, !tbaa !38
  %121 = icmp eq %"class.std::__1::__tree_node_base"* %120, %103
  br i1 %121, label %122, label %123

; <label>:122                                     ; preds = %114
  store %"class.std::__1::__tree_node_base"* %108, %"class.std::__1::__tree_node_base"** %119, align 8, !tbaa !38
  br label %_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit

; <label>:123                                     ; preds = %114
  %124 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %118, i64 0, i32 1
  store %"class.std::__1::__tree_node_base"* %108, %"class.std::__1::__tree_node_base"** %124, align 8, !tbaa !43
  br label %_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit

_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit: ; preds = %123, %122
  store %"class.std::__1::__tree_node_base"* %103, %"class.std::__1::__tree_node_base"** %109, align 8, !tbaa !38
  store %"class.std::__1::__tree_node_base"* %108, %"class.std::__1::__tree_node_base"** %115, align 8, !tbaa !49
  br label %.critedge

.critedge.loopexit:                               ; preds = %.backedge, %.lr.ph
  br label %.critedge

.critedge:                                        ; preds = %.critedge.loopexit, %_ZNSt3__118__tree_left_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit, %_ZNSt3__119__tree_right_rotateIPNS_16__tree_node_baseIPvEEEEvT_.exit2, %0
  ret void
}

attributes #0 = { alwaysinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind readonly uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { alwaysinline nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nobuiltin "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nobuiltin nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { noreturn nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { noinline nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #13 = { noinline noreturn nounwind }
attributes #14 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #15 = { nobuiltin nounwind readnone }
attributes #16 = { builtin nobuiltin }
attributes #17 = { nobuiltin }
attributes #18 = { nobuiltin nounwind }
attributes #19 = { nobuiltin noreturn nounwind }
attributes #20 = { noreturn nounwind }
attributes #21 = { builtin nobuiltin nounwind }

!llvm.ident = !{!0, !0}
!hcc.kernels = !{!1}

!0 = metadata !{metadata !"HCC clang version 3.5.0 (tags/RELEASE_350/final) (based on HCC 0.8.1541-1e50a32-348ebf7 LLVM 3.5.0svn)"}
!1 = metadata !{void (i32, i32*, i32*, i32*, i32)* @_ZN12_GLOBAL__N_18foo_func19__cxxamp_trampolineEiPiS1_S1_i, metadata !2, metadata !3, metadata !4, metadata !5, metadata !6}
!2 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0}
!3 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!4 = metadata !{metadata !"kernel_arg_type", metadata !"int", metadata !"int*", metadata !"int*", metadata !"int*", metadata !"int"}
!5 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!6 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!7 = metadata !{metadata !8, metadata !9, i64 0}
!8 = metadata !{metadata !"_ZTSN6Kalmar12__index_leafILi0EEE", metadata !9, i64 0, metadata !9, i64 4}
!9 = metadata !{metadata !"int", metadata !10, i64 0}
!10 = metadata !{metadata !"omnipotent char", metadata !11, i64 0}
!11 = metadata !{metadata !"Simple C/C++ TBAA"}
!12 = metadata !{metadata !9, metadata !9, i64 0}
!13 = metadata !{metadata !14, metadata !9, i64 0}
!14 = metadata !{metadata !"_ZTSN6Kalmar12__index_leafILi1EEE", metadata !9, i64 0, metadata !9, i64 4}
!15 = metadata !{metadata !16, metadata !9, i64 0}
!16 = metadata !{metadata !"_ZTSN6Kalmar12__index_leafILi2EEE", metadata !9, i64 0, metadata !9, i64 4}
!17 = metadata !{metadata !18, metadata !9, i64 0}
!18 = metadata !{metadata !"_ZTS5tid_s", metadata !9, i64 0, metadata !19, i64 8, metadata !19, i64 16}
!19 = metadata !{metadata !"any pointer", metadata !10, i64 0}
!20 = metadata !{i64 0, i64 4, metadata !12, i64 8, i64 8, metadata !21, i64 16, i64 8, metadata !21}
!21 = metadata !{metadata !19, metadata !19, i64 0}
!22 = metadata !{metadata !23, metadata !19, i64 8}
!23 = metadata !{metadata !"_ZTSNSt3__110shared_ptrIN6Kalmar11KalmarQueueEEE", metadata !19, i64 0, metadata !19, i64 8}
!24 = metadata !{metadata !25, metadata !19, i64 8}
!25 = metadata !{metadata !"_ZTSN6Kalmar13KalmarContextE", metadata !19, i64 8, metadata !26, i64 16}
!26 = metadata !{metadata !"_ZTSNSt3__16vectorIPN6Kalmar12KalmarDeviceENS_9allocatorIS3_EEEE"}
!27 = metadata !{metadata !28, metadata !19, i64 8}
!28 = metadata !{metadata !"_ZTSNSt3__113__vector_baseIPN6Kalmar12KalmarDeviceENS_9allocatorIS3_EEEE", metadata !19, i64 0, metadata !19, i64 8, metadata !29, i64 16}
!29 = metadata !{metadata !"_ZTSNSt3__117__compressed_pairIPPN6Kalmar12KalmarDeviceENS_9allocatorIS3_EEEE"}
!30 = metadata !{metadata !28, metadata !19, i64 0}
!31 = metadata !{metadata !23, metadata !19, i64 0}
!32 = metadata !{metadata !33, metadata !19, i64 8}
!33 = metadata !{metadata !"_ZTSN2hc17completion_futureE", metadata !34, i64 0, metadata !19, i64 8, metadata !35, i64 16}
!34 = metadata !{metadata !"_ZTSNSt3__113shared_futureIvEE", metadata !19, i64 0}
!35 = metadata !{metadata !"_ZTSNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEEE", metadata !19, i64 0, metadata !19, i64 8}
!36 = metadata !{metadata !35, metadata !19, i64 8}
!37 = metadata !{metadata !35, metadata !19, i64 0}
!38 = metadata !{metadata !39, metadata !19, i64 0}
!39 = metadata !{metadata !"_ZTSNSt3__115__tree_end_nodeIPNS_16__tree_node_baseIPvEEEE", metadata !19, i64 0}
!40 = metadata !{metadata !41, metadata !42, i64 0}
!41 = metadata !{metadata !"_ZTSNSt3__128__libcpp_compressed_pair_impImNS_4lessINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEEEELj2EEE", metadata !42, i64 0}
!42 = metadata !{metadata !"long", metadata !10, i64 0}
!43 = metadata !{metadata !44, metadata !19, i64 8}
!44 = metadata !{metadata !"_ZTSNSt3__116__tree_node_baseIPvEE", metadata !19, i64 8, metadata !19, i64 16, metadata !45, i64 24}
!45 = metadata !{metadata !"bool", metadata !10, i64 0}
!46 = metadata !{metadata !47, metadata !47, i64 0}
!47 = metadata !{metadata !"vtable pointer", metadata !11, i64 0}
!48 = metadata !{metadata !42, metadata !42, i64 0}
!49 = metadata !{metadata !44, metadata !19, i64 16}
!50 = metadata !{metadata !10, metadata !10, i64 0}
!51 = metadata !{metadata !52, metadata !19, i64 16}
!52 = metadata !{metadata !"_ZTSNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__longE", metadata !42, i64 0, metadata !42, i64 8, metadata !19, i64 16}
!53 = metadata !{metadata !44, metadata !45, i64 24}
!54 = metadata !{i8 0, i8 2}
