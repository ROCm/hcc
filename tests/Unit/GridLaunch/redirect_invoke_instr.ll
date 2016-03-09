; RUN: %opt -load %llvm_libs_dir/LLVMDirectFuncCall.so -S -redirect < %s | tee %t | %FileCheck %s
; ModuleID = 'redirect_invoke_instr.ll'
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
%"class.Kalmar::index" = type { %"struct.Kalmar::index_impl" }
%"struct.Kalmar::index_impl" = type { %"class.Kalmar::__index_leaf" }
%"class.Kalmar::__index_leaf" = type { i32, i32 }
%"class.Kalmar::index.0" = type { %"struct.Kalmar::index_impl.1" }
%"struct.Kalmar::index_impl.1" = type { %"class.Kalmar::__index_leaf", %"class.Kalmar::__index_leaf.2" }
%"class.Kalmar::__index_leaf.2" = type { i32, i32 }
%"class.Kalmar::index.3" = type { %"struct.Kalmar::index_impl.4" }
%"struct.Kalmar::index_impl.4" = type { %"class.Kalmar::__index_leaf", %"class.Kalmar::__index_leaf.2", %"class.Kalmar::__index_leaf.5" }
%"class.Kalmar::__index_leaf.5" = type { i32, i32 }
%"struct.std::__1::less" = type { i8 }
%struct.grid_launch_parm = type { %struct.uint3, %struct.uint3, %struct.uint3, %struct.uint3, i32 }
%struct.uint3 = type { i32, i32, i32 }
%struct.foo = type { %"class.hc::completion_future"* }
%"class.hc::completion_future" = type { %"class.std::__1::shared_future", %"class.std::__1::thread"*, %"class.std::__1::shared_ptr" }
%"class.std::__1::shared_future" = type { %"class.std::__1::__assoc_sub_state"* }
%"class.std::__1::__assoc_sub_state" = type { %"class.std::__1::__shared_count", %"class.std::exception_ptr", %"class.std::__1::mutex", %"class.std::__1::condition_variable", i32 }
%"class.std::__1::__shared_count" = type { i32 (...)**, i64 }
%"class.std::exception_ptr" = type { i8* }
%"class.std::__1::mutex" = type { %union.pthread_mutex_t }
%union.pthread_mutex_t = type { %"struct.(anonymous union)::__pthread_mutex_s" }
%"struct.(anonymous union)::__pthread_mutex_s" = type { i32, i32, i32, i32, i32, i16, i16, %struct.__pthread_internal_list }
%struct.__pthread_internal_list = type { %struct.__pthread_internal_list*, %struct.__pthread_internal_list* }
%"class.std::__1::condition_variable" = type { %union.pthread_cond_t }
%union.pthread_cond_t = type { %struct.anon }
%struct.anon = type { i32, i32, i64, i64, i64, i8*, i32, i32 }
%"class.std::__1::thread" = type { i64 }
%"class.std::__1::shared_ptr" = type { %"class.Kalmar::KalmarAsyncOp"*, %"class.std::__1::__shared_weak_count"* }
%"class.Kalmar::KalmarAsyncOp" = type { i32 (...)** }
%"class.std::__1::__shared_weak_count" = type { %"class.std::__1::__shared_count", i64 }
%struct.__va_list_tag = type { i32, i32, i8*, i8* }
%"class.std::__1::allocator" = type { i8 }
%"struct.std::__1::integral_constant" = type { i8 }
%"struct.std::__1::__has_destroy" = type { i8 }
%"class.std::__1::allocator.10" = type { i8 }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short" = type { %union.anon.12, [23 x i8] }
%union.anon.12 = type { i8 }

@_ZN6KalmarL20__mcw_cxxamp_kernelsE = internal global %"class.std::__1::set" zeroinitializer, align 8
@__dso_handle = external global i8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_directfunction_handle.cpp, i8* null }]

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2Ev(%"class.Kalmar::index"* %this) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %4, align 8
  %5 = load %"class.Kalmar::index"** %4
  %6 = getelementptr inbounds %"class.Kalmar::index"* %5, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %6, %"struct.Kalmar::index_impl"** %3, align 8
  %7 = load %"struct.Kalmar::index_impl"** %3
  %8 = bitcast %"struct.Kalmar::index_impl"* %7 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %8, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 0, i32* %2, align 4
  %9 = load %"class.Kalmar::__index_leaf"** %1
  %10 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %9, i32 0, i32 0
  %11 = load i32* %2, align 4
  store i32 %11, i32* %10, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1Ev(%"class.Kalmar::index"* %this) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca %"class.Kalmar::index"*, align 8
  %5 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %5, align 8
  %6 = load %"class.Kalmar::index"** %5
  store %"class.Kalmar::index"* %6, %"class.Kalmar::index"** %4, align 8
  %7 = load %"class.Kalmar::index"** %4
  %8 = getelementptr inbounds %"class.Kalmar::index"* %7, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %8, %"struct.Kalmar::index_impl"** %3, align 8
  %9 = load %"struct.Kalmar::index_impl"** %3
  %10 = bitcast %"struct.Kalmar::index_impl"* %9 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %10, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 0, i32* %2, align 4
  %11 = load %"class.Kalmar::__index_leaf"** %1
  %12 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %11, i32 0, i32 0
  %13 = load i32* %2, align 4
  store i32 %13, i32* %12, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2ERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %other) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca %"class.Kalmar::index"*, align 8
  %9 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %8, align 8
  store %"class.Kalmar::index"* %other, %"class.Kalmar::index"** %9, align 8
  %10 = load %"class.Kalmar::index"** %8
  %11 = getelementptr inbounds %"class.Kalmar::index"* %10, i32 0, i32 0
  %12 = load %"class.Kalmar::index"** %9, align 8
  %13 = getelementptr inbounds %"class.Kalmar::index"* %12, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %11, %"struct.Kalmar::index_impl"** %6, align 8
  store %"struct.Kalmar::index_impl"* %13, %"struct.Kalmar::index_impl"** %7, align 8
  %14 = load %"struct.Kalmar::index_impl"** %6
  %15 = load %"struct.Kalmar::index_impl"** %7, align 8
  %16 = bitcast %"struct.Kalmar::index_impl"* %15 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %16, %"class.Kalmar::__index_leaf"** %5, align 8
  %17 = load %"class.Kalmar::__index_leaf"** %5
  %18 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %17, i32 0, i32 0
  %19 = load i32* %18
  store %"struct.Kalmar::index_impl"* %14, %"struct.Kalmar::index_impl"** %3, align 8
  store i32 %19, i32* %4, align 4
  %20 = load %"struct.Kalmar::index_impl"** %3
  %21 = bitcast %"struct.Kalmar::index_impl"* %20 to %"class.Kalmar::__index_leaf"*
  %22 = load i32* %4, align 4
  store %"class.Kalmar::__index_leaf"* %21, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %22, i32* %2, align 4
  %23 = load %"class.Kalmar::__index_leaf"** %1
  %24 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %23, i32 0, i32 0
  %25 = load i32* %2, align 4
  store i32 %25, i32* %24, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1ERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %other) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca %"class.Kalmar::index"*, align 8
  %9 = alloca %"class.Kalmar::index"*, align 8
  %10 = alloca %"class.Kalmar::index"*, align 8
  %11 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %10, align 8
  store %"class.Kalmar::index"* %other, %"class.Kalmar::index"** %11, align 8
  %12 = load %"class.Kalmar::index"** %10
  %13 = load %"class.Kalmar::index"** %11
  store %"class.Kalmar::index"* %12, %"class.Kalmar::index"** %8, align 8
  store %"class.Kalmar::index"* %13, %"class.Kalmar::index"** %9, align 8
  %14 = load %"class.Kalmar::index"** %8
  %15 = getelementptr inbounds %"class.Kalmar::index"* %14, i32 0, i32 0
  %16 = load %"class.Kalmar::index"** %9, align 8
  %17 = getelementptr inbounds %"class.Kalmar::index"* %16, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %15, %"struct.Kalmar::index_impl"** %6, align 8
  store %"struct.Kalmar::index_impl"* %17, %"struct.Kalmar::index_impl"** %7, align 8
  %18 = load %"struct.Kalmar::index_impl"** %6
  %19 = load %"struct.Kalmar::index_impl"** %7, align 8
  %20 = bitcast %"struct.Kalmar::index_impl"* %19 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %20, %"class.Kalmar::__index_leaf"** %5, align 8
  %21 = load %"class.Kalmar::__index_leaf"** %5
  %22 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %21, i32 0, i32 0
  %23 = load i32* %22
  store %"struct.Kalmar::index_impl"* %18, %"struct.Kalmar::index_impl"** %3, align 8
  store i32 %23, i32* %4, align 4
  %24 = load %"struct.Kalmar::index_impl"** %3
  %25 = bitcast %"struct.Kalmar::index_impl"* %24 to %"class.Kalmar::__index_leaf"*
  %26 = load i32* %4, align 4
  store %"class.Kalmar::__index_leaf"* %25, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %26, i32* %2, align 4
  %27 = load %"class.Kalmar::__index_leaf"** %1
  %28 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %27, i32 0, i32 0
  %29 = load i32* %2, align 4
  store i32 %29, i32* %28, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2Ei(%"class.Kalmar::index"* %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::index"*, align 8
  %6 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %5, align 8
  store i32 %i0, i32* %6, align 4
  %7 = load %"class.Kalmar::index"** %5
  %8 = getelementptr inbounds %"class.Kalmar::index"* %7, i32 0, i32 0
  %9 = load i32* %6, align 4
  store %"struct.Kalmar::index_impl"* %8, %"struct.Kalmar::index_impl"** %3, align 8
  store i32 %9, i32* %4, align 4
  %10 = load %"struct.Kalmar::index_impl"** %3
  %11 = bitcast %"struct.Kalmar::index_impl"* %10 to %"class.Kalmar::__index_leaf"*
  %12 = load i32* %4, align 4
  store %"class.Kalmar::__index_leaf"* %11, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %12, i32* %2, align 4
  %13 = load %"class.Kalmar::__index_leaf"** %1
  %14 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %13, i32 0, i32 0
  %15 = load i32* %2, align 4
  store i32 %15, i32* %14, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1Ei(%"class.Kalmar::index"* %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::index"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::index"*, align 8
  %8 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %7, align 8
  store i32 %i0, i32* %8, align 4
  %9 = load %"class.Kalmar::index"** %7
  %10 = load i32* %8, align 4
  store %"class.Kalmar::index"* %9, %"class.Kalmar::index"** %5, align 8
  store i32 %10, i32* %6, align 4
  %11 = load %"class.Kalmar::index"** %5
  %12 = getelementptr inbounds %"class.Kalmar::index"* %11, i32 0, i32 0
  %13 = load i32* %6, align 4
  store %"struct.Kalmar::index_impl"* %12, %"struct.Kalmar::index_impl"** %3, align 8
  store i32 %13, i32* %4, align 4
  %14 = load %"struct.Kalmar::index_impl"** %3
  %15 = bitcast %"struct.Kalmar::index_impl"* %14 to %"class.Kalmar::__index_leaf"*
  %16 = load i32* %4, align 4
  store %"class.Kalmar::__index_leaf"* %15, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %16, i32* %2, align 4
  %17 = load %"class.Kalmar::__index_leaf"** %1
  %18 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %17, i32 0, i32 0
  %19 = load i32* %2, align 4
  store i32 %19, i32* %18, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2EPKi(%"class.Kalmar::index"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca %"class.Kalmar::index"*, align 8
  %6 = alloca i32*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %5, align 8
  store i32* %components, i32** %6, align 8
  %7 = load %"class.Kalmar::index"** %5
  %8 = getelementptr inbounds %"class.Kalmar::index"* %7, i32 0, i32 0
  %9 = load i32** %6, align 8
  store %"struct.Kalmar::index_impl"* %8, %"struct.Kalmar::index_impl"** %3, align 8
  store i32* %9, i32** %4, align 8
  %10 = load %"struct.Kalmar::index_impl"** %3
  %11 = bitcast %"struct.Kalmar::index_impl"* %10 to %"class.Kalmar::__index_leaf"*
  %12 = load i32** %4, align 8
  %13 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %11, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %13, i32* %2, align 4
  %14 = load %"class.Kalmar::__index_leaf"** %1
  %15 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  %16 = load i32* %2, align 4
  store i32 %16, i32* %15, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1EPKi(%"class.Kalmar::index"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca %"class.Kalmar::index"*, align 8
  %6 = alloca i32*, align 8
  %7 = alloca %"class.Kalmar::index"*, align 8
  %8 = alloca i32*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %7, align 8
  store i32* %components, i32** %8, align 8
  %9 = load %"class.Kalmar::index"** %7
  %10 = load i32** %8, align 8
  store %"class.Kalmar::index"* %9, %"class.Kalmar::index"** %5, align 8
  store i32* %10, i32** %6, align 8
  %11 = load %"class.Kalmar::index"** %5
  %12 = getelementptr inbounds %"class.Kalmar::index"* %11, i32 0, i32 0
  %13 = load i32** %6, align 8
  store %"struct.Kalmar::index_impl"* %12, %"struct.Kalmar::index_impl"** %3, align 8
  store i32* %13, i32** %4, align 8
  %14 = load %"struct.Kalmar::index_impl"** %3
  %15 = bitcast %"struct.Kalmar::index_impl"* %14 to %"class.Kalmar::__index_leaf"*
  %16 = load i32** %4, align 8
  %17 = load i32* %16, align 4
  store %"class.Kalmar::__index_leaf"* %15, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %17, i32* %2, align 4
  %18 = load %"class.Kalmar::__index_leaf"** %1
  %19 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %18, i32 0, i32 0
  %20 = load i32* %2, align 4
  store i32 %20, i32* %19, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC2EPi(%"class.Kalmar::index"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca %"class.Kalmar::index"*, align 8
  %6 = alloca i32*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %5, align 8
  store i32* %components, i32** %6, align 8
  %7 = load %"class.Kalmar::index"** %5
  %8 = getelementptr inbounds %"class.Kalmar::index"* %7, i32 0, i32 0
  %9 = load i32** %6, align 8
  store %"struct.Kalmar::index_impl"* %8, %"struct.Kalmar::index_impl"** %3, align 8
  store i32* %9, i32** %4, align 8
  %10 = load %"struct.Kalmar::index_impl"** %3
  %11 = bitcast %"struct.Kalmar::index_impl"* %10 to %"class.Kalmar::__index_leaf"*
  %12 = load i32** %4, align 8
  %13 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %11, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %13, i32* %2, align 4
  %14 = load %"class.Kalmar::__index_leaf"** %1
  %15 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  %16 = load i32* %2, align 4
  store i32 %16, i32* %15, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEC1EPi(%"class.Kalmar::index"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca i32*, align 8
  %5 = alloca %"class.Kalmar::index"*, align 8
  %6 = alloca i32*, align 8
  %7 = alloca %"class.Kalmar::index"*, align 8
  %8 = alloca i32*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %7, align 8
  store i32* %components, i32** %8, align 8
  %9 = load %"class.Kalmar::index"** %7
  %10 = load i32** %8, align 8
  store %"class.Kalmar::index"* %9, %"class.Kalmar::index"** %5, align 8
  store i32* %10, i32** %6, align 8
  %11 = load %"class.Kalmar::index"** %5
  %12 = getelementptr inbounds %"class.Kalmar::index"* %11, i32 0, i32 0
  %13 = load i32** %6, align 8
  store %"struct.Kalmar::index_impl"* %12, %"struct.Kalmar::index_impl"** %3, align 8
  store i32* %13, i32** %4, align 8
  %14 = load %"struct.Kalmar::index_impl"** %3
  %15 = bitcast %"struct.Kalmar::index_impl"* %14 to %"class.Kalmar::__index_leaf"*
  %16 = load i32** %4, align 8
  %17 = load i32* %16, align 4
  store %"class.Kalmar::__index_leaf"* %15, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %17, i32* %2, align 4
  %18 = load %"class.Kalmar::__index_leaf"** %1
  %19 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %18, i32 0, i32 0
  %20 = load i32* %2, align 4
  store i32 %20, i32* %19, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEaSERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %other) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"struct.Kalmar::index_impl"*, align 8
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca %"class.Kalmar::__index_leaf", align 4
  %9 = alloca %"class.Kalmar::index"*, align 8
  %10 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %9, align 8
  store %"class.Kalmar::index"* %other, %"class.Kalmar::index"** %10, align 8
  %11 = load %"class.Kalmar::index"** %9
  %12 = getelementptr inbounds %"class.Kalmar::index"* %11, i32 0, i32 0
  %13 = load %"class.Kalmar::index"** %10, align 8
  %14 = getelementptr inbounds %"class.Kalmar::index"* %13, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %12, %"struct.Kalmar::index_impl"** %6, align 8
  store %"struct.Kalmar::index_impl"* %14, %"struct.Kalmar::index_impl"** %7, align 8
  %15 = load %"struct.Kalmar::index_impl"** %6
  %16 = bitcast %"struct.Kalmar::index_impl"* %15 to %"class.Kalmar::__index_leaf"*
  %17 = load %"struct.Kalmar::index_impl"** %7, align 8
  %18 = bitcast %"struct.Kalmar::index_impl"* %17 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %5, align 8
  %19 = load %"class.Kalmar::__index_leaf"** %5
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %19, i32 0, i32 0
  %21 = load i32* %20
  store %"class.Kalmar::__index_leaf"* %16, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %21, i32* %2, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %1
  %23 = load i32* %2, align 4
  %24 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  store i32 %23, i32* %24, align 4
  %25 = bitcast %"class.Kalmar::__index_leaf"* %8 to i8*
  %26 = bitcast %"class.Kalmar::__index_leaf"* %22 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %25, i8* %26, i64 8, i32 4, i1 false) #2
  %27 = bitcast %"class.Kalmar::__index_leaf"* %8 to i64*
  %28 = load i64* %27, align 1
  %29 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %28, i64* %29, align 1
  store %"struct.Kalmar::index_impl"* %15, %"struct.Kalmar::index_impl"** %4, align 8
  %30 = load %"struct.Kalmar::index_impl"** %4
  ret %"class.Kalmar::index"* %11
}

; Function Attrs: alwaysinline uwtable
define weak_odr i32 @_ZNK6Kalmar5indexILi1EEixEj(%"class.Kalmar::index"* %this, i32 %c) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index"*, align 8
  %5 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %4, align 8
  store i32 %c, i32* %5, align 4
  %6 = load %"class.Kalmar::index"** %4
  %7 = getelementptr inbounds %"class.Kalmar::index"* %6, i32 0, i32 0
  %8 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl"* %7, %"struct.Kalmar::index_impl"** %2, align 8
  store i32 %8, i32* %3, align 4
  %9 = load %"struct.Kalmar::index_impl"** %2
  %10 = bitcast %"struct.Kalmar::index_impl"* %9 to %"class.Kalmar::__index_leaf"*
  %11 = load i32* %3, align 4
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %10, i64 %12
  store %"class.Kalmar::__index_leaf"* %13, %"class.Kalmar::__index_leaf"** %1, align 8
  %14 = load %"class.Kalmar::__index_leaf"** %1
  %15 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  %16 = load i32* %15
  ret i32 %16
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(4) i32* @_ZN6Kalmar5indexILi1EEixEj(%"class.Kalmar::index"* %this, i32 %c) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index"*, align 8
  %5 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %4, align 8
  store i32 %c, i32* %5, align 4
  %6 = load %"class.Kalmar::index"** %4
  %7 = getelementptr inbounds %"class.Kalmar::index"* %6, i32 0, i32 0
  %8 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl"* %7, %"struct.Kalmar::index_impl"** %2, align 8
  store i32 %8, i32* %3, align 4
  %9 = load %"struct.Kalmar::index_impl"** %2
  %10 = bitcast %"struct.Kalmar::index_impl"* %9 to %"class.Kalmar::__index_leaf"*
  %11 = load i32* %3, align 4
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %10, i64 %12
  store %"class.Kalmar::__index_leaf"* %13, %"class.Kalmar::__index_leaf"** %1, align 8
  %14 = load %"class.Kalmar::__index_leaf"** %1
  %15 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  ret i32* %15
}

; Function Attrs: alwaysinline uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi1EEeqERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %other) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::index"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"class.Kalmar::index"*, align 8
  %12 = alloca %"class.Kalmar::index"*, align 8
  %13 = alloca %"class.Kalmar::index"*, align 8
  %14 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %13, align 8
  store %"class.Kalmar::index"* %other, %"class.Kalmar::index"** %14, align 8
  %15 = load %"class.Kalmar::index"** %13
  %16 = load %"class.Kalmar::index"** %14, align 8
  store %"class.Kalmar::index"* %15, %"class.Kalmar::index"** %11, align 8
  store %"class.Kalmar::index"* %16, %"class.Kalmar::index"** %12, align 8
  %17 = load %"class.Kalmar::index"** %11, align 8
  store %"class.Kalmar::index"* %17, %"class.Kalmar::index"** %9, align 8
  store i32 0, i32* %10, align 4
  %18 = load %"class.Kalmar::index"** %9
  %19 = getelementptr inbounds %"class.Kalmar::index"* %18, i32 0, i32 0
  %20 = load i32* %10, align 4
  store %"struct.Kalmar::index_impl"* %19, %"struct.Kalmar::index_impl"** %7, align 8
  store i32 %20, i32* %8, align 4
  %21 = load %"struct.Kalmar::index_impl"** %7
  %22 = bitcast %"struct.Kalmar::index_impl"* %21 to %"class.Kalmar::__index_leaf"*
  %23 = load i32* %8, align 4
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i64 %24
  store %"class.Kalmar::__index_leaf"* %25, %"class.Kalmar::__index_leaf"** %6, align 8
  %26 = load %"class.Kalmar::__index_leaf"** %6
  %27 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %26, i32 0, i32 0
  %28 = load i32* %27
  %29 = load %"class.Kalmar::index"** %12, align 8
  store %"class.Kalmar::index"* %29, %"class.Kalmar::index"** %4, align 8
  store i32 0, i32* %5, align 4
  %30 = load %"class.Kalmar::index"** %4
  %31 = getelementptr inbounds %"class.Kalmar::index"* %30, i32 0, i32 0
  %32 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl"* %31, %"struct.Kalmar::index_impl"** %2, align 8
  store i32 %32, i32* %3, align 4
  %33 = load %"struct.Kalmar::index_impl"** %2
  %34 = bitcast %"struct.Kalmar::index_impl"* %33 to %"class.Kalmar::__index_leaf"*
  %35 = load i32* %3, align 4
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %34, i64 %36
  store %"class.Kalmar::__index_leaf"* %37, %"class.Kalmar::__index_leaf"** %1, align 8
  %38 = load %"class.Kalmar::__index_leaf"** %1
  %39 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %38, i32 0, i32 0
  %40 = load i32* %39
  %41 = icmp eq i32 %28, %40
  ret i1 %41
}

; Function Attrs: alwaysinline uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi1EEneERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %other) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::index"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"class.Kalmar::index"*, align 8
  %12 = alloca %"class.Kalmar::index"*, align 8
  %13 = alloca %"class.Kalmar::index"*, align 8
  %14 = alloca %"class.Kalmar::index"*, align 8
  %15 = alloca %"class.Kalmar::index"*, align 8
  %16 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %15, align 8
  store %"class.Kalmar::index"* %other, %"class.Kalmar::index"** %16, align 8
  %17 = load %"class.Kalmar::index"** %15
  %18 = load %"class.Kalmar::index"** %16, align 8
  store %"class.Kalmar::index"* %17, %"class.Kalmar::index"** %13, align 8
  store %"class.Kalmar::index"* %18, %"class.Kalmar::index"** %14, align 8
  %19 = load %"class.Kalmar::index"** %13
  %20 = load %"class.Kalmar::index"** %14, align 8
  store %"class.Kalmar::index"* %19, %"class.Kalmar::index"** %11, align 8
  store %"class.Kalmar::index"* %20, %"class.Kalmar::index"** %12, align 8
  %21 = load %"class.Kalmar::index"** %11, align 8
  store %"class.Kalmar::index"* %21, %"class.Kalmar::index"** %9, align 8
  store i32 0, i32* %10, align 4
  %22 = load %"class.Kalmar::index"** %9
  %23 = getelementptr inbounds %"class.Kalmar::index"* %22, i32 0, i32 0
  %24 = load i32* %10, align 4
  store %"struct.Kalmar::index_impl"* %23, %"struct.Kalmar::index_impl"** %7, align 8
  store i32 %24, i32* %8, align 4
  %25 = load %"struct.Kalmar::index_impl"** %7
  %26 = bitcast %"struct.Kalmar::index_impl"* %25 to %"class.Kalmar::__index_leaf"*
  %27 = load i32* %8, align 4
  %28 = zext i32 %27 to i64
  %29 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %26, i64 %28
  store %"class.Kalmar::__index_leaf"* %29, %"class.Kalmar::__index_leaf"** %6, align 8
  %30 = load %"class.Kalmar::__index_leaf"** %6
  %31 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %30, i32 0, i32 0
  %32 = load i32* %31
  %33 = load %"class.Kalmar::index"** %12, align 8
  store %"class.Kalmar::index"* %33, %"class.Kalmar::index"** %4, align 8
  store i32 0, i32* %5, align 4
  %34 = load %"class.Kalmar::index"** %4
  %35 = getelementptr inbounds %"class.Kalmar::index"* %34, i32 0, i32 0
  %36 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl"* %35, %"struct.Kalmar::index_impl"** %2, align 8
  store i32 %36, i32* %3, align 4
  %37 = load %"struct.Kalmar::index_impl"** %2
  %38 = bitcast %"struct.Kalmar::index_impl"* %37 to %"class.Kalmar::__index_leaf"*
  %39 = load i32* %3, align 4
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %38, i64 %40
  store %"class.Kalmar::__index_leaf"* %41, %"class.Kalmar::__index_leaf"** %1, align 8
  %42 = load %"class.Kalmar::__index_leaf"** %1
  %43 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %42, i32 0, i32 0
  %44 = load i32* %43
  %45 = icmp eq i32 %32, %44
  %46 = xor i1 %45, true
  ret i1 %46
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEpLERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %rhs) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"struct.Kalmar::index_impl"*, align 8
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca %"class.Kalmar::__index_leaf", align 4
  %9 = alloca %"class.Kalmar::index"*, align 8
  %10 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %9, align 8
  store %"class.Kalmar::index"* %rhs, %"class.Kalmar::index"** %10, align 8
  %11 = load %"class.Kalmar::index"** %9
  %12 = getelementptr inbounds %"class.Kalmar::index"* %11, i32 0, i32 0
  %13 = load %"class.Kalmar::index"** %10, align 8
  %14 = getelementptr inbounds %"class.Kalmar::index"* %13, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %12, %"struct.Kalmar::index_impl"** %6, align 8
  store %"struct.Kalmar::index_impl"* %14, %"struct.Kalmar::index_impl"** %7, align 8
  %15 = load %"struct.Kalmar::index_impl"** %6
  %16 = bitcast %"struct.Kalmar::index_impl"* %15 to %"class.Kalmar::__index_leaf"*
  %17 = load %"struct.Kalmar::index_impl"** %7, align 8
  %18 = bitcast %"struct.Kalmar::index_impl"* %17 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %5, align 8
  %19 = load %"class.Kalmar::__index_leaf"** %5
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %19, i32 0, i32 0
  %21 = load i32* %20
  store %"class.Kalmar::__index_leaf"* %16, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %21, i32* %2, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %1
  %23 = load i32* %2, align 4
  %24 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  %25 = load i32* %24, align 4
  %26 = add nsw i32 %25, %23
  store i32 %26, i32* %24, align 4
  %27 = bitcast %"class.Kalmar::__index_leaf"* %8 to i8*
  %28 = bitcast %"class.Kalmar::__index_leaf"* %22 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %27, i8* %28, i64 8, i32 4, i1 false) #2
  %29 = bitcast %"class.Kalmar::__index_leaf"* %8 to i64*
  %30 = load i64* %29, align 1
  %31 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %30, i64* %31, align 1
  store %"struct.Kalmar::index_impl"* %15, %"struct.Kalmar::index_impl"** %4, align 8
  %32 = load %"struct.Kalmar::index_impl"** %4
  ret %"class.Kalmar::index"* %11
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmIERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %rhs) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"struct.Kalmar::index_impl"*, align 8
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca %"class.Kalmar::__index_leaf", align 4
  %9 = alloca %"class.Kalmar::index"*, align 8
  %10 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %9, align 8
  store %"class.Kalmar::index"* %rhs, %"class.Kalmar::index"** %10, align 8
  %11 = load %"class.Kalmar::index"** %9
  %12 = getelementptr inbounds %"class.Kalmar::index"* %11, i32 0, i32 0
  %13 = load %"class.Kalmar::index"** %10, align 8
  %14 = getelementptr inbounds %"class.Kalmar::index"* %13, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %12, %"struct.Kalmar::index_impl"** %6, align 8
  store %"struct.Kalmar::index_impl"* %14, %"struct.Kalmar::index_impl"** %7, align 8
  %15 = load %"struct.Kalmar::index_impl"** %6
  %16 = bitcast %"struct.Kalmar::index_impl"* %15 to %"class.Kalmar::__index_leaf"*
  %17 = load %"struct.Kalmar::index_impl"** %7, align 8
  %18 = bitcast %"struct.Kalmar::index_impl"* %17 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %5, align 8
  %19 = load %"class.Kalmar::__index_leaf"** %5
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %19, i32 0, i32 0
  %21 = load i32* %20
  store %"class.Kalmar::__index_leaf"* %16, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %21, i32* %2, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %1
  %23 = load i32* %2, align 4
  %24 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  %25 = load i32* %24, align 4
  %26 = sub nsw i32 %25, %23
  store i32 %26, i32* %24, align 4
  %27 = bitcast %"class.Kalmar::__index_leaf"* %8 to i8*
  %28 = bitcast %"class.Kalmar::__index_leaf"* %22 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %27, i8* %28, i64 8, i32 4, i1 false) #2
  %29 = bitcast %"class.Kalmar::__index_leaf"* %8 to i64*
  %30 = load i64* %29, align 1
  %31 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %30, i64* %31, align 1
  store %"struct.Kalmar::index_impl"* %15, %"struct.Kalmar::index_impl"** %4, align 8
  %32 = load %"struct.Kalmar::index_impl"** %4
  ret %"class.Kalmar::index"* %11
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmLERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %__r) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"struct.Kalmar::index_impl"*, align 8
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca %"class.Kalmar::__index_leaf", align 4
  %9 = alloca %"class.Kalmar::index"*, align 8
  %10 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %9, align 8
  store %"class.Kalmar::index"* %__r, %"class.Kalmar::index"** %10, align 8
  %11 = load %"class.Kalmar::index"** %9
  %12 = getelementptr inbounds %"class.Kalmar::index"* %11, i32 0, i32 0
  %13 = load %"class.Kalmar::index"** %10, align 8
  %14 = getelementptr inbounds %"class.Kalmar::index"* %13, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %12, %"struct.Kalmar::index_impl"** %6, align 8
  store %"struct.Kalmar::index_impl"* %14, %"struct.Kalmar::index_impl"** %7, align 8
  %15 = load %"struct.Kalmar::index_impl"** %6
  %16 = bitcast %"struct.Kalmar::index_impl"* %15 to %"class.Kalmar::__index_leaf"*
  %17 = load %"struct.Kalmar::index_impl"** %7, align 8
  %18 = bitcast %"struct.Kalmar::index_impl"* %17 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %5, align 8
  %19 = load %"class.Kalmar::__index_leaf"** %5
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %19, i32 0, i32 0
  %21 = load i32* %20
  store %"class.Kalmar::__index_leaf"* %16, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %21, i32* %2, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %1
  %23 = load i32* %2, align 4
  %24 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  %25 = load i32* %24, align 4
  %26 = mul nsw i32 %25, %23
  store i32 %26, i32* %24, align 4
  %27 = bitcast %"class.Kalmar::__index_leaf"* %8 to i8*
  %28 = bitcast %"class.Kalmar::__index_leaf"* %22 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %27, i8* %28, i64 8, i32 4, i1 false) #2
  %29 = bitcast %"class.Kalmar::__index_leaf"* %8 to i64*
  %30 = load i64* %29, align 1
  %31 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %30, i64* %31, align 1
  store %"struct.Kalmar::index_impl"* %15, %"struct.Kalmar::index_impl"** %4, align 8
  %32 = load %"struct.Kalmar::index_impl"** %4
  ret %"class.Kalmar::index"* %11
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEdVERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %__r) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"struct.Kalmar::index_impl"*, align 8
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca %"class.Kalmar::__index_leaf", align 4
  %9 = alloca %"class.Kalmar::index"*, align 8
  %10 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %9, align 8
  store %"class.Kalmar::index"* %__r, %"class.Kalmar::index"** %10, align 8
  %11 = load %"class.Kalmar::index"** %9
  %12 = getelementptr inbounds %"class.Kalmar::index"* %11, i32 0, i32 0
  %13 = load %"class.Kalmar::index"** %10, align 8
  %14 = getelementptr inbounds %"class.Kalmar::index"* %13, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %12, %"struct.Kalmar::index_impl"** %6, align 8
  store %"struct.Kalmar::index_impl"* %14, %"struct.Kalmar::index_impl"** %7, align 8
  %15 = load %"struct.Kalmar::index_impl"** %6
  %16 = bitcast %"struct.Kalmar::index_impl"* %15 to %"class.Kalmar::__index_leaf"*
  %17 = load %"struct.Kalmar::index_impl"** %7, align 8
  %18 = bitcast %"struct.Kalmar::index_impl"* %17 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %5, align 8
  %19 = load %"class.Kalmar::__index_leaf"** %5
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %19, i32 0, i32 0
  %21 = load i32* %20
  store %"class.Kalmar::__index_leaf"* %16, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %21, i32* %2, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %1
  %23 = load i32* %2, align 4
  %24 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  %25 = load i32* %24, align 4
  %26 = sdiv i32 %25, %23
  store i32 %26, i32* %24, align 4
  %27 = bitcast %"class.Kalmar::__index_leaf"* %8 to i8*
  %28 = bitcast %"class.Kalmar::__index_leaf"* %22 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %27, i8* %28, i64 8, i32 4, i1 false) #2
  %29 = bitcast %"class.Kalmar::__index_leaf"* %8 to i64*
  %30 = load i64* %29, align 1
  %31 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %30, i64* %31, align 1
  store %"struct.Kalmar::index_impl"* %15, %"struct.Kalmar::index_impl"** %4, align 8
  %32 = load %"struct.Kalmar::index_impl"** %4
  ret %"class.Kalmar::index"* %11
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EErMERKS1_(%"class.Kalmar::index"* %this, %"class.Kalmar::index"* dereferenceable(8) %__r) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"struct.Kalmar::index_impl"*, align 8
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl"*, align 8
  %8 = alloca %"class.Kalmar::__index_leaf", align 4
  %9 = alloca %"class.Kalmar::index"*, align 8
  %10 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %9, align 8
  store %"class.Kalmar::index"* %__r, %"class.Kalmar::index"** %10, align 8
  %11 = load %"class.Kalmar::index"** %9
  %12 = getelementptr inbounds %"class.Kalmar::index"* %11, i32 0, i32 0
  %13 = load %"class.Kalmar::index"** %10, align 8
  %14 = getelementptr inbounds %"class.Kalmar::index"* %13, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %12, %"struct.Kalmar::index_impl"** %6, align 8
  store %"struct.Kalmar::index_impl"* %14, %"struct.Kalmar::index_impl"** %7, align 8
  %15 = load %"struct.Kalmar::index_impl"** %6
  %16 = bitcast %"struct.Kalmar::index_impl"* %15 to %"class.Kalmar::__index_leaf"*
  %17 = load %"struct.Kalmar::index_impl"** %7, align 8
  %18 = bitcast %"struct.Kalmar::index_impl"* %17 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %5, align 8
  %19 = load %"class.Kalmar::__index_leaf"** %5
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %19, i32 0, i32 0
  %21 = load i32* %20
  store %"class.Kalmar::__index_leaf"* %16, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %21, i32* %2, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %1
  %23 = load i32* %2, align 4
  %24 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  %25 = load i32* %24, align 4
  %26 = srem i32 %25, %23
  store i32 %26, i32* %24, align 4
  %27 = bitcast %"class.Kalmar::__index_leaf"* %8 to i8*
  %28 = bitcast %"class.Kalmar::__index_leaf"* %22 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %27, i8* %28, i64 8, i32 4, i1 false) #2
  %29 = bitcast %"class.Kalmar::__index_leaf"* %8 to i64*
  %30 = load i64* %29, align 1
  %31 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %30, i64* %31, align 1
  store %"struct.Kalmar::index_impl"* %15, %"struct.Kalmar::index_impl"** %4, align 8
  %32 = load %"struct.Kalmar::index_impl"** %4
  ret %"class.Kalmar::index"* %11
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEpLEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf", align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::__index_leaf", align 4
  %8 = alloca %"class.Kalmar::index"*, align 8
  %9 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %8, align 8
  store i32 %value, i32* %9, align 4
  %10 = load %"class.Kalmar::index"** %8
  %11 = getelementptr inbounds %"class.Kalmar::index"* %10, i32 0, i32 0
  %12 = load i32* %9, align 4
  store %"struct.Kalmar::index_impl"* %11, %"struct.Kalmar::index_impl"** %5, align 8
  store i32 %12, i32* %6, align 4
  %13 = load %"struct.Kalmar::index_impl"** %5
  %14 = bitcast %"struct.Kalmar::index_impl"* %13 to %"class.Kalmar::__index_leaf"*
  %15 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf"* %14, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %15, i32* %4, align 4
  %16 = load %"class.Kalmar::__index_leaf"** %3
  %17 = load i32* %4, align 4
  %18 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %16, i32 0, i32 0
  %19 = load i32* %18, align 4
  %20 = add nsw i32 %19, %17
  store i32 %20, i32* %18, align 4
  %21 = bitcast %"class.Kalmar::__index_leaf"* %7 to i8*
  %22 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %21, i8* %22, i64 8, i32 4, i1 false) #2
  %23 = bitcast %"class.Kalmar::__index_leaf"* %7 to i64*
  %24 = load i64* %23, align 1
  %25 = bitcast %"class.Kalmar::__index_leaf"* %1 to i64*
  store i64 %24, i64* %25, align 1
  store %"struct.Kalmar::index_impl"* %13, %"struct.Kalmar::index_impl"** %2, align 8
  %26 = load %"struct.Kalmar::index_impl"** %2
  ret %"class.Kalmar::index"* %10
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmIEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf", align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::__index_leaf", align 4
  %8 = alloca %"class.Kalmar::index"*, align 8
  %9 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %8, align 8
  store i32 %value, i32* %9, align 4
  %10 = load %"class.Kalmar::index"** %8
  %11 = getelementptr inbounds %"class.Kalmar::index"* %10, i32 0, i32 0
  %12 = load i32* %9, align 4
  store %"struct.Kalmar::index_impl"* %11, %"struct.Kalmar::index_impl"** %5, align 8
  store i32 %12, i32* %6, align 4
  %13 = load %"struct.Kalmar::index_impl"** %5
  %14 = bitcast %"struct.Kalmar::index_impl"* %13 to %"class.Kalmar::__index_leaf"*
  %15 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf"* %14, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %15, i32* %4, align 4
  %16 = load %"class.Kalmar::__index_leaf"** %3
  %17 = load i32* %4, align 4
  %18 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %16, i32 0, i32 0
  %19 = load i32* %18, align 4
  %20 = sub nsw i32 %19, %17
  store i32 %20, i32* %18, align 4
  %21 = bitcast %"class.Kalmar::__index_leaf"* %7 to i8*
  %22 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %21, i8* %22, i64 8, i32 4, i1 false) #2
  %23 = bitcast %"class.Kalmar::__index_leaf"* %7 to i64*
  %24 = load i64* %23, align 1
  %25 = bitcast %"class.Kalmar::__index_leaf"* %1 to i64*
  store i64 %24, i64* %25, align 1
  store %"struct.Kalmar::index_impl"* %13, %"struct.Kalmar::index_impl"** %2, align 8
  %26 = load %"struct.Kalmar::index_impl"** %2
  ret %"class.Kalmar::index"* %10
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmLEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf", align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::__index_leaf", align 4
  %8 = alloca %"class.Kalmar::index"*, align 8
  %9 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %8, align 8
  store i32 %value, i32* %9, align 4
  %10 = load %"class.Kalmar::index"** %8
  %11 = getelementptr inbounds %"class.Kalmar::index"* %10, i32 0, i32 0
  %12 = load i32* %9, align 4
  store %"struct.Kalmar::index_impl"* %11, %"struct.Kalmar::index_impl"** %5, align 8
  store i32 %12, i32* %6, align 4
  %13 = load %"struct.Kalmar::index_impl"** %5
  %14 = bitcast %"struct.Kalmar::index_impl"* %13 to %"class.Kalmar::__index_leaf"*
  %15 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf"* %14, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %15, i32* %4, align 4
  %16 = load %"class.Kalmar::__index_leaf"** %3
  %17 = load i32* %4, align 4
  %18 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %16, i32 0, i32 0
  %19 = load i32* %18, align 4
  %20 = mul nsw i32 %19, %17
  store i32 %20, i32* %18, align 4
  %21 = bitcast %"class.Kalmar::__index_leaf"* %7 to i8*
  %22 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %21, i8* %22, i64 8, i32 4, i1 false) #2
  %23 = bitcast %"class.Kalmar::__index_leaf"* %7 to i64*
  %24 = load i64* %23, align 1
  %25 = bitcast %"class.Kalmar::__index_leaf"* %1 to i64*
  store i64 %24, i64* %25, align 1
  store %"struct.Kalmar::index_impl"* %13, %"struct.Kalmar::index_impl"** %2, align 8
  %26 = load %"struct.Kalmar::index_impl"** %2
  ret %"class.Kalmar::index"* %10
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEdVEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf", align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::__index_leaf", align 4
  %8 = alloca %"class.Kalmar::index"*, align 8
  %9 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %8, align 8
  store i32 %value, i32* %9, align 4
  %10 = load %"class.Kalmar::index"** %8
  %11 = getelementptr inbounds %"class.Kalmar::index"* %10, i32 0, i32 0
  %12 = load i32* %9, align 4
  store %"struct.Kalmar::index_impl"* %11, %"struct.Kalmar::index_impl"** %5, align 8
  store i32 %12, i32* %6, align 4
  %13 = load %"struct.Kalmar::index_impl"** %5
  %14 = bitcast %"struct.Kalmar::index_impl"* %13 to %"class.Kalmar::__index_leaf"*
  %15 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf"* %14, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %15, i32* %4, align 4
  %16 = load %"class.Kalmar::__index_leaf"** %3
  %17 = load i32* %4, align 4
  %18 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %16, i32 0, i32 0
  %19 = load i32* %18, align 4
  %20 = sdiv i32 %19, %17
  store i32 %20, i32* %18, align 4
  %21 = bitcast %"class.Kalmar::__index_leaf"* %7 to i8*
  %22 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %21, i8* %22, i64 8, i32 4, i1 false) #2
  %23 = bitcast %"class.Kalmar::__index_leaf"* %7 to i64*
  %24 = load i64* %23, align 1
  %25 = bitcast %"class.Kalmar::__index_leaf"* %1 to i64*
  store i64 %24, i64* %25, align 1
  store %"struct.Kalmar::index_impl"* %13, %"struct.Kalmar::index_impl"** %2, align 8
  %26 = load %"struct.Kalmar::index_impl"** %2
  ret %"class.Kalmar::index"* %10
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EErMEi(%"class.Kalmar::index"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf", align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::__index_leaf", align 4
  %8 = alloca %"class.Kalmar::index"*, align 8
  %9 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %8, align 8
  store i32 %value, i32* %9, align 4
  %10 = load %"class.Kalmar::index"** %8
  %11 = getelementptr inbounds %"class.Kalmar::index"* %10, i32 0, i32 0
  %12 = load i32* %9, align 4
  store %"struct.Kalmar::index_impl"* %11, %"struct.Kalmar::index_impl"** %5, align 8
  store i32 %12, i32* %6, align 4
  %13 = load %"struct.Kalmar::index_impl"** %5
  %14 = bitcast %"struct.Kalmar::index_impl"* %13 to %"class.Kalmar::__index_leaf"*
  %15 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf"* %14, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %15, i32* %4, align 4
  %16 = load %"class.Kalmar::__index_leaf"** %3
  %17 = load i32* %4, align 4
  %18 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %16, i32 0, i32 0
  %19 = load i32* %18, align 4
  %20 = srem i32 %19, %17
  store i32 %20, i32* %18, align 4
  %21 = bitcast %"class.Kalmar::__index_leaf"* %7 to i8*
  %22 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %21, i8* %22, i64 8, i32 4, i1 false)
  %23 = bitcast %"class.Kalmar::__index_leaf"* %7 to i64*
  %24 = load i64* %23, align 1
  %25 = bitcast %"class.Kalmar::__index_leaf"* %1 to i64*
  store i64 %24, i64* %25, align 1
  store %"struct.Kalmar::index_impl"* %13, %"struct.Kalmar::index_impl"** %2, align 8
  %26 = load %"struct.Kalmar::index_impl"** %2
  ret %"class.Kalmar::index"* %10
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEppEv(%"class.Kalmar::index"* %this) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf", align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::__index_leaf", align 4
  %8 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %8, align 8
  %9 = load %"class.Kalmar::index"** %8
  %10 = getelementptr inbounds %"class.Kalmar::index"* %9, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %10, %"struct.Kalmar::index_impl"** %5, align 8
  store i32 1, i32* %6, align 4
  %11 = load %"struct.Kalmar::index_impl"** %5
  %12 = bitcast %"struct.Kalmar::index_impl"* %11 to %"class.Kalmar::__index_leaf"*
  %13 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf"* %12, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %13, i32* %4, align 4
  %14 = load %"class.Kalmar::__index_leaf"** %3
  %15 = load i32* %4, align 4
  %16 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  %17 = load i32* %16, align 4
  %18 = add nsw i32 %17, %15
  store i32 %18, i32* %16, align 4
  %19 = bitcast %"class.Kalmar::__index_leaf"* %7 to i8*
  %20 = bitcast %"class.Kalmar::__index_leaf"* %14 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %19, i8* %20, i64 8, i32 4, i1 false) #2
  %21 = bitcast %"class.Kalmar::__index_leaf"* %7 to i64*
  %22 = load i64* %21, align 1
  %23 = bitcast %"class.Kalmar::__index_leaf"* %1 to i64*
  store i64 %22, i64* %23, align 1
  store %"struct.Kalmar::index_impl"* %11, %"struct.Kalmar::index_impl"** %2, align 8
  %24 = load %"struct.Kalmar::index_impl"** %2
  ret %"class.Kalmar::index"* %9
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEppEi(%"class.Kalmar::index"* noalias sret %agg.result, %"class.Kalmar::index"* %this, i32) #0 align 2 {
  %2 = alloca %"class.Kalmar::__index_leaf", align 8
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"class.Kalmar::__index_leaf", align 4
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"struct.Kalmar::index_impl"*, align 8
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl"*, align 8
  %16 = alloca %"class.Kalmar::index"*, align 8
  %17 = alloca %"class.Kalmar::index"*, align 8
  %18 = alloca %"class.Kalmar::index"*, align 8
  %19 = alloca %"class.Kalmar::index"*, align 8
  %20 = alloca %"class.Kalmar::index"*, align 8
  %21 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %20, align 8
  store i32 %0, i32* %21, align 4
  %22 = load %"class.Kalmar::index"** %20
  store %"class.Kalmar::index"* %agg.result, %"class.Kalmar::index"** %18, align 8
  store %"class.Kalmar::index"* %22, %"class.Kalmar::index"** %19, align 8
  %23 = load %"class.Kalmar::index"** %18
  %24 = load %"class.Kalmar::index"** %19
  store %"class.Kalmar::index"* %23, %"class.Kalmar::index"** %16, align 8
  store %"class.Kalmar::index"* %24, %"class.Kalmar::index"** %17, align 8
  %25 = load %"class.Kalmar::index"** %16
  %26 = getelementptr inbounds %"class.Kalmar::index"* %25, i32 0, i32 0
  %27 = load %"class.Kalmar::index"** %17, align 8
  %28 = getelementptr inbounds %"class.Kalmar::index"* %27, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %26, %"struct.Kalmar::index_impl"** %14, align 8
  store %"struct.Kalmar::index_impl"* %28, %"struct.Kalmar::index_impl"** %15, align 8
  %29 = load %"struct.Kalmar::index_impl"** %14
  %30 = load %"struct.Kalmar::index_impl"** %15, align 8
  %31 = bitcast %"struct.Kalmar::index_impl"* %30 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %31, %"class.Kalmar::__index_leaf"** %13, align 8
  %32 = load %"class.Kalmar::__index_leaf"** %13
  %33 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %32, i32 0, i32 0
  %34 = load i32* %33
  store %"struct.Kalmar::index_impl"* %29, %"struct.Kalmar::index_impl"** %11, align 8
  store i32 %34, i32* %12, align 4
  %35 = load %"struct.Kalmar::index_impl"** %11
  %36 = bitcast %"struct.Kalmar::index_impl"* %35 to %"class.Kalmar::__index_leaf"*
  %37 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %36, %"class.Kalmar::__index_leaf"** %9, align 8
  store i32 %37, i32* %10, align 4
  %38 = load %"class.Kalmar::__index_leaf"** %9
  %39 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %38, i32 0, i32 0
  %40 = load i32* %10, align 4
  store i32 %40, i32* %39, align 4
  %41 = getelementptr inbounds %"class.Kalmar::index"* %22, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %41, %"struct.Kalmar::index_impl"** %6, align 8
  store i32 1, i32* %7, align 4
  %42 = load %"struct.Kalmar::index_impl"** %6
  %43 = bitcast %"struct.Kalmar::index_impl"* %42 to %"class.Kalmar::__index_leaf"*
  %44 = load i32* %7, align 4
  store %"class.Kalmar::__index_leaf"* %43, %"class.Kalmar::__index_leaf"** %4, align 8
  store i32 %44, i32* %5, align 4
  %45 = load %"class.Kalmar::__index_leaf"** %4
  %46 = load i32* %5, align 4
  %47 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %45, i32 0, i32 0
  %48 = load i32* %47, align 4
  %49 = add nsw i32 %48, %46
  store i32 %49, i32* %47, align 4
  %50 = bitcast %"class.Kalmar::__index_leaf"* %8 to i8*
  %51 = bitcast %"class.Kalmar::__index_leaf"* %45 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %50, i8* %51, i64 8, i32 4, i1 false) #2
  %52 = bitcast %"class.Kalmar::__index_leaf"* %8 to i64*
  %53 = load i64* %52, align 1
  %54 = bitcast %"class.Kalmar::__index_leaf"* %2 to i64*
  store i64 %53, i64* %54, align 1
  store %"struct.Kalmar::index_impl"* %42, %"struct.Kalmar::index_impl"** %3, align 8
  %55 = load %"struct.Kalmar::index_impl"** %3
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(8) %"class.Kalmar::index"* @_ZN6Kalmar5indexILi1EEmmEv(%"class.Kalmar::index"* %this) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf", align 8
  %2 = alloca %"struct.Kalmar::index_impl"*, align 8
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::__index_leaf", align 4
  %8 = alloca %"class.Kalmar::index"*, align 8
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %8, align 8
  %9 = load %"class.Kalmar::index"** %8
  %10 = getelementptr inbounds %"class.Kalmar::index"* %9, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %10, %"struct.Kalmar::index_impl"** %5, align 8
  store i32 1, i32* %6, align 4
  %11 = load %"struct.Kalmar::index_impl"** %5
  %12 = bitcast %"struct.Kalmar::index_impl"* %11 to %"class.Kalmar::__index_leaf"*
  %13 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf"* %12, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %13, i32* %4, align 4
  %14 = load %"class.Kalmar::__index_leaf"** %3
  %15 = load i32* %4, align 4
  %16 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  %17 = load i32* %16, align 4
  %18 = sub nsw i32 %17, %15
  store i32 %18, i32* %16, align 4
  %19 = bitcast %"class.Kalmar::__index_leaf"* %7 to i8*
  %20 = bitcast %"class.Kalmar::__index_leaf"* %14 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %19, i8* %20, i64 8, i32 4, i1 false) #2
  %21 = bitcast %"class.Kalmar::__index_leaf"* %7 to i64*
  %22 = load i64* %21, align 1
  %23 = bitcast %"class.Kalmar::__index_leaf"* %1 to i64*
  store i64 %22, i64* %23, align 1
  store %"struct.Kalmar::index_impl"* %11, %"struct.Kalmar::index_impl"** %2, align 8
  %24 = load %"struct.Kalmar::index_impl"** %2
  ret %"class.Kalmar::index"* %9
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi1EEmmEi(%"class.Kalmar::index"* noalias sret %agg.result, %"class.Kalmar::index"* %this, i32) #0 align 2 {
  %2 = alloca %"class.Kalmar::__index_leaf", align 8
  %3 = alloca %"struct.Kalmar::index_impl"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"struct.Kalmar::index_impl"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"class.Kalmar::__index_leaf", align 4
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"struct.Kalmar::index_impl"*, align 8
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl"*, align 8
  %16 = alloca %"class.Kalmar::index"*, align 8
  %17 = alloca %"class.Kalmar::index"*, align 8
  %18 = alloca %"class.Kalmar::index"*, align 8
  %19 = alloca %"class.Kalmar::index"*, align 8
  %20 = alloca %"class.Kalmar::index"*, align 8
  %21 = alloca i32, align 4
  store %"class.Kalmar::index"* %this, %"class.Kalmar::index"** %20, align 8
  store i32 %0, i32* %21, align 4
  %22 = load %"class.Kalmar::index"** %20
  store %"class.Kalmar::index"* %agg.result, %"class.Kalmar::index"** %18, align 8
  store %"class.Kalmar::index"* %22, %"class.Kalmar::index"** %19, align 8
  %23 = load %"class.Kalmar::index"** %18
  %24 = load %"class.Kalmar::index"** %19
  store %"class.Kalmar::index"* %23, %"class.Kalmar::index"** %16, align 8
  store %"class.Kalmar::index"* %24, %"class.Kalmar::index"** %17, align 8
  %25 = load %"class.Kalmar::index"** %16
  %26 = getelementptr inbounds %"class.Kalmar::index"* %25, i32 0, i32 0
  %27 = load %"class.Kalmar::index"** %17, align 8
  %28 = getelementptr inbounds %"class.Kalmar::index"* %27, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %26, %"struct.Kalmar::index_impl"** %14, align 8
  store %"struct.Kalmar::index_impl"* %28, %"struct.Kalmar::index_impl"** %15, align 8
  %29 = load %"struct.Kalmar::index_impl"** %14
  %30 = load %"struct.Kalmar::index_impl"** %15, align 8
  %31 = bitcast %"struct.Kalmar::index_impl"* %30 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %31, %"class.Kalmar::__index_leaf"** %13, align 8
  %32 = load %"class.Kalmar::__index_leaf"** %13
  %33 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %32, i32 0, i32 0
  %34 = load i32* %33
  store %"struct.Kalmar::index_impl"* %29, %"struct.Kalmar::index_impl"** %11, align 8
  store i32 %34, i32* %12, align 4
  %35 = load %"struct.Kalmar::index_impl"** %11
  %36 = bitcast %"struct.Kalmar::index_impl"* %35 to %"class.Kalmar::__index_leaf"*
  %37 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %36, %"class.Kalmar::__index_leaf"** %9, align 8
  store i32 %37, i32* %10, align 4
  %38 = load %"class.Kalmar::__index_leaf"** %9
  %39 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %38, i32 0, i32 0
  %40 = load i32* %10, align 4
  store i32 %40, i32* %39, align 4
  %41 = getelementptr inbounds %"class.Kalmar::index"* %22, i32 0, i32 0
  store %"struct.Kalmar::index_impl"* %41, %"struct.Kalmar::index_impl"** %6, align 8
  store i32 1, i32* %7, align 4
  %42 = load %"struct.Kalmar::index_impl"** %6
  %43 = bitcast %"struct.Kalmar::index_impl"* %42 to %"class.Kalmar::__index_leaf"*
  %44 = load i32* %7, align 4
  store %"class.Kalmar::__index_leaf"* %43, %"class.Kalmar::__index_leaf"** %4, align 8
  store i32 %44, i32* %5, align 4
  %45 = load %"class.Kalmar::__index_leaf"** %4
  %46 = load i32* %5, align 4
  %47 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %45, i32 0, i32 0
  %48 = load i32* %47, align 4
  %49 = sub nsw i32 %48, %46
  store i32 %49, i32* %47, align 4
  %50 = bitcast %"class.Kalmar::__index_leaf"* %8 to i8*
  %51 = bitcast %"class.Kalmar::__index_leaf"* %45 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %50, i8* %51, i64 8, i32 4, i1 false) #2
  %52 = bitcast %"class.Kalmar::__index_leaf"* %8 to i64*
  %53 = load i64* %52, align 1
  %54 = bitcast %"class.Kalmar::__index_leaf"* %2 to i64*
  store i64 %53, i64* %54, align 1
  store %"struct.Kalmar::index_impl"* %42, %"struct.Kalmar::index_impl"** %3, align 8
  %55 = load %"struct.Kalmar::index_impl"** %3
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2Ev(%"class.Kalmar::index.0"* %this) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %6, align 8
  %7 = load %"class.Kalmar::index.0"** %6
  %8 = getelementptr inbounds %"class.Kalmar::index.0"* %7, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %8, %"struct.Kalmar::index_impl.1"** %5, align 8
  %9 = load %"struct.Kalmar::index_impl.1"** %5
  %10 = bitcast %"struct.Kalmar::index_impl.1"* %9 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %10, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 0, i32* %4, align 4
  %11 = load %"class.Kalmar::__index_leaf"** %3
  %12 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %11, i32 0, i32 0
  %13 = load i32* %4, align 4
  store i32 %13, i32* %12, align 4
  %14 = bitcast %"struct.Kalmar::index_impl.1"* %9 to i8*
  %15 = getelementptr inbounds i8* %14, i64 8
  %16 = bitcast i8* %15 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %16, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 0, i32* %2, align 4
  %17 = load %"class.Kalmar::__index_leaf.2"** %1
  %18 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %17, i32 0, i32 0
  %19 = load i32* %2, align 4
  store i32 %19, i32* %18, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1Ev(%"class.Kalmar::index.0"* %this) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca %"class.Kalmar::index.0"*, align 8
  %7 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %7, align 8
  %8 = load %"class.Kalmar::index.0"** %7
  store %"class.Kalmar::index.0"* %8, %"class.Kalmar::index.0"** %6, align 8
  %9 = load %"class.Kalmar::index.0"** %6
  %10 = getelementptr inbounds %"class.Kalmar::index.0"* %9, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %10, %"struct.Kalmar::index_impl.1"** %5, align 8
  %11 = load %"struct.Kalmar::index_impl.1"** %5
  %12 = bitcast %"struct.Kalmar::index_impl.1"* %11 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %12, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 0, i32* %4, align 4
  %13 = load %"class.Kalmar::__index_leaf"** %3
  %14 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %13, i32 0, i32 0
  %15 = load i32* %4, align 4
  store i32 %15, i32* %14, align 4
  %16 = bitcast %"struct.Kalmar::index_impl.1"* %11 to i8*
  %17 = getelementptr inbounds i8* %16, i64 8
  %18 = bitcast i8* %17 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %18, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 0, i32* %2, align 4
  %19 = load %"class.Kalmar::__index_leaf.2"** %1
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %19, i32 0, i32 0
  %21 = load i32* %2, align 4
  store i32 %21, i32* %20, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2ERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %other) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %11 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %12 = alloca %"class.Kalmar::index.0"*, align 8
  %13 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %12, align 8
  store %"class.Kalmar::index.0"* %other, %"class.Kalmar::index.0"** %13, align 8
  %14 = load %"class.Kalmar::index.0"** %12
  %15 = getelementptr inbounds %"class.Kalmar::index.0"* %14, i32 0, i32 0
  %16 = load %"class.Kalmar::index.0"** %13, align 8
  %17 = getelementptr inbounds %"class.Kalmar::index.0"* %16, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %15, %"struct.Kalmar::index_impl.1"** %10, align 8
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %11, align 8
  %18 = load %"struct.Kalmar::index_impl.1"** %10
  %19 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %20 = bitcast %"struct.Kalmar::index_impl.1"* %19 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %20, %"class.Kalmar::__index_leaf"** %9, align 8
  %21 = load %"class.Kalmar::__index_leaf"** %9
  %22 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %21, i32 0, i32 0
  %23 = load i32* %22
  %24 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %25 = bitcast %"struct.Kalmar::index_impl.1"* %24 to i8*
  %26 = getelementptr inbounds i8* %25, i64 8
  %27 = bitcast i8* %26 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %27, %"class.Kalmar::__index_leaf.2"** %1, align 8
  %28 = load %"class.Kalmar::__index_leaf.2"** %1
  %29 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %28, i32 0, i32 0
  %30 = load i32* %29
  store %"struct.Kalmar::index_impl.1"* %18, %"struct.Kalmar::index_impl.1"** %6, align 8
  store i32 %23, i32* %7, align 4
  store i32 %30, i32* %8, align 4
  %31 = load %"struct.Kalmar::index_impl.1"** %6
  %32 = bitcast %"struct.Kalmar::index_impl.1"* %31 to %"class.Kalmar::__index_leaf"*
  %33 = load i32* %7, align 4
  store %"class.Kalmar::__index_leaf"* %32, %"class.Kalmar::__index_leaf"** %4, align 8
  store i32 %33, i32* %5, align 4
  %34 = load %"class.Kalmar::__index_leaf"** %4
  %35 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %34, i32 0, i32 0
  %36 = load i32* %5, align 4
  store i32 %36, i32* %35, align 4
  %37 = bitcast %"struct.Kalmar::index_impl.1"* %31 to i8*
  %38 = getelementptr inbounds i8* %37, i64 8
  %39 = bitcast i8* %38 to %"class.Kalmar::__index_leaf.2"*
  %40 = load i32* %8, align 4
  store %"class.Kalmar::__index_leaf.2"* %39, %"class.Kalmar::__index_leaf.2"** %2, align 8
  store i32 %40, i32* %3, align 4
  %41 = load %"class.Kalmar::__index_leaf.2"** %2
  %42 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %41, i32 0, i32 0
  %43 = load i32* %3, align 4
  store i32 %43, i32* %42, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1ERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %other) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %11 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %12 = alloca %"class.Kalmar::index.0"*, align 8
  %13 = alloca %"class.Kalmar::index.0"*, align 8
  %14 = alloca %"class.Kalmar::index.0"*, align 8
  %15 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %14, align 8
  store %"class.Kalmar::index.0"* %other, %"class.Kalmar::index.0"** %15, align 8
  %16 = load %"class.Kalmar::index.0"** %14
  %17 = load %"class.Kalmar::index.0"** %15
  store %"class.Kalmar::index.0"* %16, %"class.Kalmar::index.0"** %12, align 8
  store %"class.Kalmar::index.0"* %17, %"class.Kalmar::index.0"** %13, align 8
  %18 = load %"class.Kalmar::index.0"** %12
  %19 = getelementptr inbounds %"class.Kalmar::index.0"* %18, i32 0, i32 0
  %20 = load %"class.Kalmar::index.0"** %13, align 8
  %21 = getelementptr inbounds %"class.Kalmar::index.0"* %20, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %19, %"struct.Kalmar::index_impl.1"** %10, align 8
  store %"struct.Kalmar::index_impl.1"* %21, %"struct.Kalmar::index_impl.1"** %11, align 8
  %22 = load %"struct.Kalmar::index_impl.1"** %10
  %23 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %24 = bitcast %"struct.Kalmar::index_impl.1"* %23 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %24, %"class.Kalmar::__index_leaf"** %9, align 8
  %25 = load %"class.Kalmar::__index_leaf"** %9
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %25, i32 0, i32 0
  %27 = load i32* %26
  %28 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %29 = bitcast %"struct.Kalmar::index_impl.1"* %28 to i8*
  %30 = getelementptr inbounds i8* %29, i64 8
  %31 = bitcast i8* %30 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %31, %"class.Kalmar::__index_leaf.2"** %1, align 8
  %32 = load %"class.Kalmar::__index_leaf.2"** %1
  %33 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %32, i32 0, i32 0
  %34 = load i32* %33
  store %"struct.Kalmar::index_impl.1"* %22, %"struct.Kalmar::index_impl.1"** %6, align 8
  store i32 %27, i32* %7, align 4
  store i32 %34, i32* %8, align 4
  %35 = load %"struct.Kalmar::index_impl.1"** %6
  %36 = bitcast %"struct.Kalmar::index_impl.1"* %35 to %"class.Kalmar::__index_leaf"*
  %37 = load i32* %7, align 4
  store %"class.Kalmar::__index_leaf"* %36, %"class.Kalmar::__index_leaf"** %4, align 8
  store i32 %37, i32* %5, align 4
  %38 = load %"class.Kalmar::__index_leaf"** %4
  %39 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %38, i32 0, i32 0
  %40 = load i32* %5, align 4
  store i32 %40, i32* %39, align 4
  %41 = bitcast %"struct.Kalmar::index_impl.1"* %35 to i8*
  %42 = getelementptr inbounds i8* %41, i64 8
  %43 = bitcast i8* %42 to %"class.Kalmar::__index_leaf.2"*
  %44 = load i32* %8, align 4
  store %"class.Kalmar::__index_leaf.2"* %43, %"class.Kalmar::__index_leaf.2"** %2, align 8
  store i32 %44, i32* %3, align 4
  %45 = load %"class.Kalmar::__index_leaf.2"** %2
  %46 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %45, i32 0, i32 0
  %47 = load i32* %3, align 4
  store i32 %47, i32* %46, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2Ei(%"class.Kalmar::index.0"* %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::index.0"*, align 8
  %8 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %7, align 8
  store i32 %i0, i32* %8, align 4
  %9 = load %"class.Kalmar::index.0"** %7
  %10 = getelementptr inbounds %"class.Kalmar::index.0"* %9, i32 0, i32 0
  %11 = load i32* %8, align 4
  store %"struct.Kalmar::index_impl.1"* %10, %"struct.Kalmar::index_impl.1"** %5, align 8
  store i32 %11, i32* %6, align 4
  %12 = load %"struct.Kalmar::index_impl.1"** %5
  %13 = bitcast %"struct.Kalmar::index_impl.1"* %12 to %"class.Kalmar::__index_leaf"*
  %14 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf"* %13, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %14, i32* %4, align 4
  %15 = load %"class.Kalmar::__index_leaf"** %3
  %16 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %15, i32 0, i32 0
  %17 = load i32* %4, align 4
  store i32 %17, i32* %16, align 4
  %18 = bitcast %"struct.Kalmar::index_impl.1"* %12 to i8*
  %19 = getelementptr inbounds i8* %18, i64 8
  %20 = bitcast i8* %19 to %"class.Kalmar::__index_leaf.2"*
  %21 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf.2"* %20, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %21, i32* %2, align 4
  %22 = load %"class.Kalmar::__index_leaf.2"** %1
  %23 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %22, i32 0, i32 0
  %24 = load i32* %2, align 4
  store i32 %24, i32* %23, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1Ei(%"class.Kalmar::index.0"* %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::index.0"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::index.0"*, align 8
  %10 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %9, align 8
  store i32 %i0, i32* %10, align 4
  %11 = load %"class.Kalmar::index.0"** %9
  %12 = load i32* %10, align 4
  store %"class.Kalmar::index.0"* %11, %"class.Kalmar::index.0"** %7, align 8
  store i32 %12, i32* %8, align 4
  %13 = load %"class.Kalmar::index.0"** %7
  %14 = getelementptr inbounds %"class.Kalmar::index.0"* %13, i32 0, i32 0
  %15 = load i32* %8, align 4
  store %"struct.Kalmar::index_impl.1"* %14, %"struct.Kalmar::index_impl.1"** %5, align 8
  store i32 %15, i32* %6, align 4
  %16 = load %"struct.Kalmar::index_impl.1"** %5
  %17 = bitcast %"struct.Kalmar::index_impl.1"* %16 to %"class.Kalmar::__index_leaf"*
  %18 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf"* %17, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %18, i32* %4, align 4
  %19 = load %"class.Kalmar::__index_leaf"** %3
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %19, i32 0, i32 0
  %21 = load i32* %4, align 4
  store i32 %21, i32* %20, align 4
  %22 = bitcast %"struct.Kalmar::index_impl.1"* %16 to i8*
  %23 = getelementptr inbounds i8* %22, i64 8
  %24 = bitcast i8* %23 to %"class.Kalmar::__index_leaf.2"*
  %25 = load i32* %6, align 4
  store %"class.Kalmar::__index_leaf.2"* %24, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %25, i32* %2, align 4
  %26 = load %"class.Kalmar::__index_leaf.2"** %1
  %27 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %26, i32 0, i32 0
  %28 = load i32* %2, align 4
  store i32 %28, i32* %27, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2EPKi(%"class.Kalmar::index.0"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca i32*, align 8
  %7 = alloca %"class.Kalmar::index.0"*, align 8
  %8 = alloca i32*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %7, align 8
  store i32* %components, i32** %8, align 8
  %9 = load %"class.Kalmar::index.0"** %7
  %10 = getelementptr inbounds %"class.Kalmar::index.0"* %9, i32 0, i32 0
  %11 = load i32** %8, align 8
  store %"struct.Kalmar::index_impl.1"* %10, %"struct.Kalmar::index_impl.1"** %5, align 8
  store i32* %11, i32** %6, align 8
  %12 = load %"struct.Kalmar::index_impl.1"** %5
  %13 = bitcast %"struct.Kalmar::index_impl.1"* %12 to %"class.Kalmar::__index_leaf"*
  %14 = load i32** %6, align 8
  %15 = load i32* %14, align 4
  store %"class.Kalmar::__index_leaf"* %13, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %15, i32* %4, align 4
  %16 = load %"class.Kalmar::__index_leaf"** %3
  %17 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %16, i32 0, i32 0
  %18 = load i32* %4, align 4
  store i32 %18, i32* %17, align 4
  %19 = bitcast %"struct.Kalmar::index_impl.1"* %12 to i8*
  %20 = getelementptr inbounds i8* %19, i64 8
  %21 = bitcast i8* %20 to %"class.Kalmar::__index_leaf.2"*
  %22 = load i32** %6, align 8
  %23 = getelementptr inbounds i32* %22, i64 1
  %24 = load i32* %23, align 4
  store %"class.Kalmar::__index_leaf.2"* %21, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %24, i32* %2, align 4
  %25 = load %"class.Kalmar::__index_leaf.2"** %1
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %25, i32 0, i32 0
  %27 = load i32* %2, align 4
  store i32 %27, i32* %26, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1EPKi(%"class.Kalmar::index.0"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca i32*, align 8
  %7 = alloca %"class.Kalmar::index.0"*, align 8
  %8 = alloca i32*, align 8
  %9 = alloca %"class.Kalmar::index.0"*, align 8
  %10 = alloca i32*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %9, align 8
  store i32* %components, i32** %10, align 8
  %11 = load %"class.Kalmar::index.0"** %9
  %12 = load i32** %10, align 8
  store %"class.Kalmar::index.0"* %11, %"class.Kalmar::index.0"** %7, align 8
  store i32* %12, i32** %8, align 8
  %13 = load %"class.Kalmar::index.0"** %7
  %14 = getelementptr inbounds %"class.Kalmar::index.0"* %13, i32 0, i32 0
  %15 = load i32** %8, align 8
  store %"struct.Kalmar::index_impl.1"* %14, %"struct.Kalmar::index_impl.1"** %5, align 8
  store i32* %15, i32** %6, align 8
  %16 = load %"struct.Kalmar::index_impl.1"** %5
  %17 = bitcast %"struct.Kalmar::index_impl.1"* %16 to %"class.Kalmar::__index_leaf"*
  %18 = load i32** %6, align 8
  %19 = load i32* %18, align 4
  store %"class.Kalmar::__index_leaf"* %17, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %19, i32* %4, align 4
  %20 = load %"class.Kalmar::__index_leaf"** %3
  %21 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %20, i32 0, i32 0
  %22 = load i32* %4, align 4
  store i32 %22, i32* %21, align 4
  %23 = bitcast %"struct.Kalmar::index_impl.1"* %16 to i8*
  %24 = getelementptr inbounds i8* %23, i64 8
  %25 = bitcast i8* %24 to %"class.Kalmar::__index_leaf.2"*
  %26 = load i32** %6, align 8
  %27 = getelementptr inbounds i32* %26, i64 1
  %28 = load i32* %27, align 4
  store %"class.Kalmar::__index_leaf.2"* %25, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %28, i32* %2, align 4
  %29 = load %"class.Kalmar::__index_leaf.2"** %1
  %30 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %29, i32 0, i32 0
  %31 = load i32* %2, align 4
  store i32 %31, i32* %30, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC2EPi(%"class.Kalmar::index.0"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca i32*, align 8
  %7 = alloca %"class.Kalmar::index.0"*, align 8
  %8 = alloca i32*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %7, align 8
  store i32* %components, i32** %8, align 8
  %9 = load %"class.Kalmar::index.0"** %7
  %10 = getelementptr inbounds %"class.Kalmar::index.0"* %9, i32 0, i32 0
  %11 = load i32** %8, align 8
  store %"struct.Kalmar::index_impl.1"* %10, %"struct.Kalmar::index_impl.1"** %5, align 8
  store i32* %11, i32** %6, align 8
  %12 = load %"struct.Kalmar::index_impl.1"** %5
  %13 = bitcast %"struct.Kalmar::index_impl.1"* %12 to %"class.Kalmar::__index_leaf"*
  %14 = load i32** %6, align 8
  %15 = load i32* %14, align 4
  store %"class.Kalmar::__index_leaf"* %13, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %15, i32* %4, align 4
  %16 = load %"class.Kalmar::__index_leaf"** %3
  %17 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %16, i32 0, i32 0
  %18 = load i32* %4, align 4
  store i32 %18, i32* %17, align 4
  %19 = bitcast %"struct.Kalmar::index_impl.1"* %12 to i8*
  %20 = getelementptr inbounds i8* %19, i64 8
  %21 = bitcast i8* %20 to %"class.Kalmar::__index_leaf.2"*
  %22 = load i32** %6, align 8
  %23 = getelementptr inbounds i32* %22, i64 1
  %24 = load i32* %23, align 4
  store %"class.Kalmar::__index_leaf.2"* %21, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %24, i32* %2, align 4
  %25 = load %"class.Kalmar::__index_leaf.2"** %1
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %25, i32 0, i32 0
  %27 = load i32* %2, align 4
  store i32 %27, i32* %26, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEC1EPi(%"class.Kalmar::index.0"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca i32*, align 8
  %7 = alloca %"class.Kalmar::index.0"*, align 8
  %8 = alloca i32*, align 8
  %9 = alloca %"class.Kalmar::index.0"*, align 8
  %10 = alloca i32*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %9, align 8
  store i32* %components, i32** %10, align 8
  %11 = load %"class.Kalmar::index.0"** %9
  %12 = load i32** %10, align 8
  store %"class.Kalmar::index.0"* %11, %"class.Kalmar::index.0"** %7, align 8
  store i32* %12, i32** %8, align 8
  %13 = load %"class.Kalmar::index.0"** %7
  %14 = getelementptr inbounds %"class.Kalmar::index.0"* %13, i32 0, i32 0
  %15 = load i32** %8, align 8
  store %"struct.Kalmar::index_impl.1"* %14, %"struct.Kalmar::index_impl.1"** %5, align 8
  store i32* %15, i32** %6, align 8
  %16 = load %"struct.Kalmar::index_impl.1"** %5
  %17 = bitcast %"struct.Kalmar::index_impl.1"* %16 to %"class.Kalmar::__index_leaf"*
  %18 = load i32** %6, align 8
  %19 = load i32* %18, align 4
  store %"class.Kalmar::__index_leaf"* %17, %"class.Kalmar::__index_leaf"** %3, align 8
  store i32 %19, i32* %4, align 4
  %20 = load %"class.Kalmar::__index_leaf"** %3
  %21 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %20, i32 0, i32 0
  %22 = load i32* %4, align 4
  store i32 %22, i32* %21, align 4
  %23 = bitcast %"struct.Kalmar::index_impl.1"* %16 to i8*
  %24 = getelementptr inbounds i8* %23, i64 8
  %25 = bitcast i8* %24 to %"class.Kalmar::__index_leaf.2"*
  %26 = load i32** %6, align 8
  %27 = getelementptr inbounds i32* %26, i64 1
  %28 = load i32* %27, align 4
  store %"class.Kalmar::__index_leaf.2"* %25, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %28, i32* %2, align 4
  %29 = load %"class.Kalmar::__index_leaf.2"** %1
  %30 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %29, i32 0, i32 0
  %31 = load i32* %2, align 4
  store i32 %31, i32* %30, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEaSERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %other) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %11 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %12 = alloca %"class.Kalmar::__index_leaf", align 4
  %13 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %14 = alloca %"class.Kalmar::index.0"*, align 8
  %15 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %14, align 8
  store %"class.Kalmar::index.0"* %other, %"class.Kalmar::index.0"** %15, align 8
  %16 = load %"class.Kalmar::index.0"** %14
  %17 = getelementptr inbounds %"class.Kalmar::index.0"* %16, i32 0, i32 0
  %18 = load %"class.Kalmar::index.0"** %15, align 8
  %19 = getelementptr inbounds %"class.Kalmar::index.0"* %18, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %10, align 8
  store %"struct.Kalmar::index_impl.1"* %19, %"struct.Kalmar::index_impl.1"** %11, align 8
  %20 = load %"struct.Kalmar::index_impl.1"** %10
  %21 = bitcast %"struct.Kalmar::index_impl.1"* %20 to %"class.Kalmar::__index_leaf"*
  %22 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %23 = bitcast %"struct.Kalmar::index_impl.1"* %22 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %23, %"class.Kalmar::__index_leaf"** %9, align 8
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %26 = load i32* %25
  store %"class.Kalmar::__index_leaf"* %21, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %26, i32* %2, align 4
  %27 = load %"class.Kalmar::__index_leaf"** %1
  %28 = load i32* %2, align 4
  %29 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %27, i32 0, i32 0
  store i32 %28, i32* %29, align 4
  %30 = bitcast %"class.Kalmar::__index_leaf"* %12 to i8*
  %31 = bitcast %"class.Kalmar::__index_leaf"* %27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %30, i8* %31, i64 8, i32 4, i1 false) #2
  %32 = bitcast %"struct.Kalmar::index_impl.1"* %20 to i8*
  %33 = getelementptr inbounds i8* %32, i64 8
  %34 = bitcast i8* %33 to %"class.Kalmar::__index_leaf.2"*
  %35 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %36 = bitcast %"struct.Kalmar::index_impl.1"* %35 to i8*
  %37 = getelementptr inbounds i8* %36, i64 8
  %38 = bitcast i8* %37 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %38, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %39 = load %"class.Kalmar::__index_leaf.2"** %3
  %40 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %39, i32 0, i32 0
  %41 = load i32* %40
  store %"class.Kalmar::__index_leaf.2"* %34, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %41, i32* %5, align 4
  %42 = load %"class.Kalmar::__index_leaf.2"** %4
  %43 = load i32* %5, align 4
  %44 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %42, i32 0, i32 0
  store i32 %43, i32* %44, align 4
  %45 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i8*
  %46 = bitcast %"class.Kalmar::__index_leaf.2"* %42 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %45, i8* %46, i64 8, i32 4, i1 false) #2
  %47 = bitcast %"class.Kalmar::__index_leaf"* %12 to i64*
  %48 = load i64* %47, align 1
  %49 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i64*
  %50 = load i64* %49, align 1
  %51 = bitcast %"class.Kalmar::__index_leaf"* %6 to i64*
  store i64 %48, i64* %51, align 1
  %52 = bitcast %"class.Kalmar::__index_leaf.2"* %7 to i64*
  store i64 %50, i64* %52, align 1
  store %"struct.Kalmar::index_impl.1"* %20, %"struct.Kalmar::index_impl.1"** %8, align 8
  %53 = load %"struct.Kalmar::index_impl.1"** %8
  ret %"class.Kalmar::index.0"* %16
}

; Function Attrs: alwaysinline uwtable
define weak_odr i32 @_ZNK6Kalmar5indexILi2EEixEj(%"class.Kalmar::index.0"* %this, i32 %c) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index.0"*, align 8
  %5 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %4, align 8
  store i32 %c, i32* %5, align 4
  %6 = load %"class.Kalmar::index.0"** %4
  %7 = getelementptr inbounds %"class.Kalmar::index.0"* %6, i32 0, i32 0
  %8 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl.1"* %7, %"struct.Kalmar::index_impl.1"** %2, align 8
  store i32 %8, i32* %3, align 4
  %9 = load %"struct.Kalmar::index_impl.1"** %2
  %10 = bitcast %"struct.Kalmar::index_impl.1"* %9 to %"class.Kalmar::__index_leaf"*
  %11 = load i32* %3, align 4
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %10, i64 %12
  store %"class.Kalmar::__index_leaf"* %13, %"class.Kalmar::__index_leaf"** %1, align 8
  %14 = load %"class.Kalmar::__index_leaf"** %1
  %15 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  %16 = load i32* %15
  ret i32 %16
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(4) i32* @_ZN6Kalmar5indexILi2EEixEj(%"class.Kalmar::index.0"* %this, i32 %c) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index.0"*, align 8
  %5 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %4, align 8
  store i32 %c, i32* %5, align 4
  %6 = load %"class.Kalmar::index.0"** %4
  %7 = getelementptr inbounds %"class.Kalmar::index.0"* %6, i32 0, i32 0
  %8 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl.1"* %7, %"struct.Kalmar::index_impl.1"** %2, align 8
  store i32 %8, i32* %3, align 4
  %9 = load %"struct.Kalmar::index_impl.1"** %2
  %10 = bitcast %"struct.Kalmar::index_impl.1"* %9 to %"class.Kalmar::__index_leaf"*
  %11 = load i32* %3, align 4
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %10, i64 %12
  store %"class.Kalmar::__index_leaf"* %13, %"class.Kalmar::__index_leaf"** %1, align 8
  %14 = load %"class.Kalmar::__index_leaf"** %1
  %15 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  ret i32* %15
}

; Function Attrs: alwaysinline uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi2EEeqERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %other) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index.0"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::index.0"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %12 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %13 = alloca i32, align 4
  %14 = alloca %"class.Kalmar::index.0"*, align 8
  %15 = alloca i32, align 4
  %16 = alloca %"class.Kalmar::index.0"*, align 8
  %17 = alloca %"class.Kalmar::index.0"*, align 8
  %18 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %19 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %20 = alloca i32, align 4
  %21 = alloca %"class.Kalmar::index.0"*, align 8
  %22 = alloca i32, align 4
  %23 = alloca %"class.Kalmar::index.0"*, align 8
  %24 = alloca %"class.Kalmar::index.0"*, align 8
  %25 = alloca %"class.Kalmar::index.0"*, align 8
  %26 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %25, align 8
  store %"class.Kalmar::index.0"* %other, %"class.Kalmar::index.0"** %26, align 8
  %27 = load %"class.Kalmar::index.0"** %25
  %28 = load %"class.Kalmar::index.0"** %26, align 8
  store %"class.Kalmar::index.0"* %27, %"class.Kalmar::index.0"** %23, align 8
  store %"class.Kalmar::index.0"* %28, %"class.Kalmar::index.0"** %24, align 8
  %29 = load %"class.Kalmar::index.0"** %23, align 8
  store %"class.Kalmar::index.0"* %29, %"class.Kalmar::index.0"** %21, align 8
  store i32 1, i32* %22, align 4
  %30 = load %"class.Kalmar::index.0"** %21
  %31 = getelementptr inbounds %"class.Kalmar::index.0"* %30, i32 0, i32 0
  %32 = load i32* %22, align 4
  store %"struct.Kalmar::index_impl.1"* %31, %"struct.Kalmar::index_impl.1"** %19, align 8
  store i32 %32, i32* %20, align 4
  %33 = load %"struct.Kalmar::index_impl.1"** %19
  %34 = bitcast %"struct.Kalmar::index_impl.1"* %33 to %"class.Kalmar::__index_leaf"*
  %35 = load i32* %20, align 4
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %34, i64 %36
  store %"class.Kalmar::__index_leaf"* %37, %"class.Kalmar::__index_leaf"** %18, align 8
  %38 = load %"class.Kalmar::__index_leaf"** %18
  %39 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %38, i32 0, i32 0
  %40 = load i32* %39
  %41 = load %"class.Kalmar::index.0"** %24, align 8
  store %"class.Kalmar::index.0"* %41, %"class.Kalmar::index.0"** %4, align 8
  store i32 1, i32* %5, align 4
  %42 = load %"class.Kalmar::index.0"** %4
  %43 = getelementptr inbounds %"class.Kalmar::index.0"* %42, i32 0, i32 0
  %44 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl.1"* %43, %"struct.Kalmar::index_impl.1"** %2, align 8
  store i32 %44, i32* %3, align 4
  %45 = load %"struct.Kalmar::index_impl.1"** %2
  %46 = bitcast %"struct.Kalmar::index_impl.1"* %45 to %"class.Kalmar::__index_leaf"*
  %47 = load i32* %3, align 4
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %46, i64 %48
  store %"class.Kalmar::__index_leaf"* %49, %"class.Kalmar::__index_leaf"** %1, align 8
  %50 = load %"class.Kalmar::__index_leaf"** %1
  %51 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %50, i32 0, i32 0
  %52 = load i32* %51
  %53 = icmp eq i32 %40, %52
  br i1 %53, label %54, label %_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_.exit

; <label>:54                                      ; preds = %0
  %55 = load %"class.Kalmar::index.0"** %23, align 8
  %56 = load %"class.Kalmar::index.0"** %24, align 8
  store %"class.Kalmar::index.0"* %55, %"class.Kalmar::index.0"** %16, align 8
  store %"class.Kalmar::index.0"* %56, %"class.Kalmar::index.0"** %17, align 8
  %57 = load %"class.Kalmar::index.0"** %16, align 8
  store %"class.Kalmar::index.0"* %57, %"class.Kalmar::index.0"** %14, align 8
  store i32 0, i32* %15, align 4
  %58 = load %"class.Kalmar::index.0"** %14
  %59 = getelementptr inbounds %"class.Kalmar::index.0"* %58, i32 0, i32 0
  %60 = load i32* %15, align 4
  store %"struct.Kalmar::index_impl.1"* %59, %"struct.Kalmar::index_impl.1"** %12, align 8
  store i32 %60, i32* %13, align 4
  %61 = load %"struct.Kalmar::index_impl.1"** %12
  %62 = bitcast %"struct.Kalmar::index_impl.1"* %61 to %"class.Kalmar::__index_leaf"*
  %63 = load i32* %13, align 4
  %64 = zext i32 %63 to i64
  %65 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %62, i64 %64
  store %"class.Kalmar::__index_leaf"* %65, %"class.Kalmar::__index_leaf"** %11, align 8
  %66 = load %"class.Kalmar::__index_leaf"** %11
  %67 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %66, i32 0, i32 0
  %68 = load i32* %67
  %69 = load %"class.Kalmar::index.0"** %17, align 8
  store %"class.Kalmar::index.0"* %69, %"class.Kalmar::index.0"** %9, align 8
  store i32 0, i32* %10, align 4
  %70 = load %"class.Kalmar::index.0"** %9
  %71 = getelementptr inbounds %"class.Kalmar::index.0"* %70, i32 0, i32 0
  %72 = load i32* %10, align 4
  store %"struct.Kalmar::index_impl.1"* %71, %"struct.Kalmar::index_impl.1"** %7, align 8
  store i32 %72, i32* %8, align 4
  %73 = load %"struct.Kalmar::index_impl.1"** %7
  %74 = bitcast %"struct.Kalmar::index_impl.1"* %73 to %"class.Kalmar::__index_leaf"*
  %75 = load i32* %8, align 4
  %76 = zext i32 %75 to i64
  %77 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %74, i64 %76
  store %"class.Kalmar::__index_leaf"* %77, %"class.Kalmar::__index_leaf"** %6, align 8
  %78 = load %"class.Kalmar::__index_leaf"** %6
  %79 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %78, i32 0, i32 0
  %80 = load i32* %79
  %81 = icmp eq i32 %68, %80
  br label %_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_.exit

_ZN6Kalmar12index_helperILi2ENS_5indexILi2EEEE5equalERKS2_S5_.exit: ; preds = %54, %0
  %82 = phi i1 [ false, %0 ], [ %81, %54 ]
  ret i1 %82
}

; Function Attrs: alwaysinline uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi2EEneERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %other) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index.0"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::index.0"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %12 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %13 = alloca i32, align 4
  %14 = alloca %"class.Kalmar::index.0"*, align 8
  %15 = alloca i32, align 4
  %16 = alloca %"class.Kalmar::index.0"*, align 8
  %17 = alloca %"class.Kalmar::index.0"*, align 8
  %18 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %19 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %20 = alloca i32, align 4
  %21 = alloca %"class.Kalmar::index.0"*, align 8
  %22 = alloca i32, align 4
  %23 = alloca %"class.Kalmar::index.0"*, align 8
  %24 = alloca %"class.Kalmar::index.0"*, align 8
  %25 = alloca %"class.Kalmar::index.0"*, align 8
  %26 = alloca %"class.Kalmar::index.0"*, align 8
  %27 = alloca %"class.Kalmar::index.0"*, align 8
  %28 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %27, align 8
  store %"class.Kalmar::index.0"* %other, %"class.Kalmar::index.0"** %28, align 8
  %29 = load %"class.Kalmar::index.0"** %27
  %30 = load %"class.Kalmar::index.0"** %28, align 8
  store %"class.Kalmar::index.0"* %29, %"class.Kalmar::index.0"** %25, align 8
  store %"class.Kalmar::index.0"* %30, %"class.Kalmar::index.0"** %26, align 8
  %31 = load %"class.Kalmar::index.0"** %25
  %32 = load %"class.Kalmar::index.0"** %26, align 8
  store %"class.Kalmar::index.0"* %31, %"class.Kalmar::index.0"** %23, align 8
  store %"class.Kalmar::index.0"* %32, %"class.Kalmar::index.0"** %24, align 8
  %33 = load %"class.Kalmar::index.0"** %23, align 8
  store %"class.Kalmar::index.0"* %33, %"class.Kalmar::index.0"** %21, align 8
  store i32 1, i32* %22, align 4
  %34 = load %"class.Kalmar::index.0"** %21
  %35 = getelementptr inbounds %"class.Kalmar::index.0"* %34, i32 0, i32 0
  %36 = load i32* %22, align 4
  store %"struct.Kalmar::index_impl.1"* %35, %"struct.Kalmar::index_impl.1"** %19, align 8
  store i32 %36, i32* %20, align 4
  %37 = load %"struct.Kalmar::index_impl.1"** %19
  %38 = bitcast %"struct.Kalmar::index_impl.1"* %37 to %"class.Kalmar::__index_leaf"*
  %39 = load i32* %20, align 4
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %38, i64 %40
  store %"class.Kalmar::__index_leaf"* %41, %"class.Kalmar::__index_leaf"** %18, align 8
  %42 = load %"class.Kalmar::__index_leaf"** %18
  %43 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %42, i32 0, i32 0
  %44 = load i32* %43
  %45 = load %"class.Kalmar::index.0"** %24, align 8
  store %"class.Kalmar::index.0"* %45, %"class.Kalmar::index.0"** %4, align 8
  store i32 1, i32* %5, align 4
  %46 = load %"class.Kalmar::index.0"** %4
  %47 = getelementptr inbounds %"class.Kalmar::index.0"* %46, i32 0, i32 0
  %48 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl.1"* %47, %"struct.Kalmar::index_impl.1"** %2, align 8
  store i32 %48, i32* %3, align 4
  %49 = load %"struct.Kalmar::index_impl.1"** %2
  %50 = bitcast %"struct.Kalmar::index_impl.1"* %49 to %"class.Kalmar::__index_leaf"*
  %51 = load i32* %3, align 4
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %50, i64 %52
  store %"class.Kalmar::__index_leaf"* %53, %"class.Kalmar::__index_leaf"** %1, align 8
  %54 = load %"class.Kalmar::__index_leaf"** %1
  %55 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %54, i32 0, i32 0
  %56 = load i32* %55
  %57 = icmp eq i32 %44, %56
  br i1 %57, label %58, label %_ZNK6Kalmar5indexILi2EEeqERKS1_.exit

; <label>:58                                      ; preds = %0
  %59 = load %"class.Kalmar::index.0"** %23, align 8
  %60 = load %"class.Kalmar::index.0"** %24, align 8
  store %"class.Kalmar::index.0"* %59, %"class.Kalmar::index.0"** %16, align 8
  store %"class.Kalmar::index.0"* %60, %"class.Kalmar::index.0"** %17, align 8
  %61 = load %"class.Kalmar::index.0"** %16, align 8
  store %"class.Kalmar::index.0"* %61, %"class.Kalmar::index.0"** %14, align 8
  store i32 0, i32* %15, align 4
  %62 = load %"class.Kalmar::index.0"** %14
  %63 = getelementptr inbounds %"class.Kalmar::index.0"* %62, i32 0, i32 0
  %64 = load i32* %15, align 4
  store %"struct.Kalmar::index_impl.1"* %63, %"struct.Kalmar::index_impl.1"** %12, align 8
  store i32 %64, i32* %13, align 4
  %65 = load %"struct.Kalmar::index_impl.1"** %12
  %66 = bitcast %"struct.Kalmar::index_impl.1"* %65 to %"class.Kalmar::__index_leaf"*
  %67 = load i32* %13, align 4
  %68 = zext i32 %67 to i64
  %69 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %66, i64 %68
  store %"class.Kalmar::__index_leaf"* %69, %"class.Kalmar::__index_leaf"** %11, align 8
  %70 = load %"class.Kalmar::__index_leaf"** %11
  %71 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %70, i32 0, i32 0
  %72 = load i32* %71
  %73 = load %"class.Kalmar::index.0"** %17, align 8
  store %"class.Kalmar::index.0"* %73, %"class.Kalmar::index.0"** %9, align 8
  store i32 0, i32* %10, align 4
  %74 = load %"class.Kalmar::index.0"** %9
  %75 = getelementptr inbounds %"class.Kalmar::index.0"* %74, i32 0, i32 0
  %76 = load i32* %10, align 4
  store %"struct.Kalmar::index_impl.1"* %75, %"struct.Kalmar::index_impl.1"** %7, align 8
  store i32 %76, i32* %8, align 4
  %77 = load %"struct.Kalmar::index_impl.1"** %7
  %78 = bitcast %"struct.Kalmar::index_impl.1"* %77 to %"class.Kalmar::__index_leaf"*
  %79 = load i32* %8, align 4
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %78, i64 %80
  store %"class.Kalmar::__index_leaf"* %81, %"class.Kalmar::__index_leaf"** %6, align 8
  %82 = load %"class.Kalmar::__index_leaf"** %6
  %83 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %82, i32 0, i32 0
  %84 = load i32* %83
  %85 = icmp eq i32 %72, %84
  br label %_ZNK6Kalmar5indexILi2EEeqERKS1_.exit

_ZNK6Kalmar5indexILi2EEeqERKS1_.exit:             ; preds = %58, %0
  %86 = phi i1 [ false, %0 ], [ %85, %58 ]
  %87 = xor i1 %86, true
  ret i1 %87
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEpLERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %rhs) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %11 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %12 = alloca %"class.Kalmar::__index_leaf", align 4
  %13 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %14 = alloca %"class.Kalmar::index.0"*, align 8
  %15 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %14, align 8
  store %"class.Kalmar::index.0"* %rhs, %"class.Kalmar::index.0"** %15, align 8
  %16 = load %"class.Kalmar::index.0"** %14
  %17 = getelementptr inbounds %"class.Kalmar::index.0"* %16, i32 0, i32 0
  %18 = load %"class.Kalmar::index.0"** %15, align 8
  %19 = getelementptr inbounds %"class.Kalmar::index.0"* %18, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %10, align 8
  store %"struct.Kalmar::index_impl.1"* %19, %"struct.Kalmar::index_impl.1"** %11, align 8
  %20 = load %"struct.Kalmar::index_impl.1"** %10
  %21 = bitcast %"struct.Kalmar::index_impl.1"* %20 to %"class.Kalmar::__index_leaf"*
  %22 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %23 = bitcast %"struct.Kalmar::index_impl.1"* %22 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %23, %"class.Kalmar::__index_leaf"** %9, align 8
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %26 = load i32* %25
  store %"class.Kalmar::__index_leaf"* %21, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %26, i32* %2, align 4
  %27 = load %"class.Kalmar::__index_leaf"** %1
  %28 = load i32* %2, align 4
  %29 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %27, i32 0, i32 0
  %30 = load i32* %29, align 4
  %31 = add nsw i32 %30, %28
  store i32 %31, i32* %29, align 4
  %32 = bitcast %"class.Kalmar::__index_leaf"* %12 to i8*
  %33 = bitcast %"class.Kalmar::__index_leaf"* %27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %32, i8* %33, i64 8, i32 4, i1 false) #2
  %34 = bitcast %"struct.Kalmar::index_impl.1"* %20 to i8*
  %35 = getelementptr inbounds i8* %34, i64 8
  %36 = bitcast i8* %35 to %"class.Kalmar::__index_leaf.2"*
  %37 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %38 = bitcast %"struct.Kalmar::index_impl.1"* %37 to i8*
  %39 = getelementptr inbounds i8* %38, i64 8
  %40 = bitcast i8* %39 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %40, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %41 = load %"class.Kalmar::__index_leaf.2"** %3
  %42 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %41, i32 0, i32 0
  %43 = load i32* %42
  store %"class.Kalmar::__index_leaf.2"* %36, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %43, i32* %5, align 4
  %44 = load %"class.Kalmar::__index_leaf.2"** %4
  %45 = load i32* %5, align 4
  %46 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %44, i32 0, i32 0
  %47 = load i32* %46, align 4
  %48 = add nsw i32 %47, %45
  store i32 %48, i32* %46, align 4
  %49 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i8*
  %50 = bitcast %"class.Kalmar::__index_leaf.2"* %44 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %49, i8* %50, i64 8, i32 4, i1 false) #2
  %51 = bitcast %"class.Kalmar::__index_leaf"* %12 to i64*
  %52 = load i64* %51, align 1
  %53 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf"* %6 to i64*
  store i64 %52, i64* %55, align 1
  %56 = bitcast %"class.Kalmar::__index_leaf.2"* %7 to i64*
  store i64 %54, i64* %56, align 1
  store %"struct.Kalmar::index_impl.1"* %20, %"struct.Kalmar::index_impl.1"** %8, align 8
  %57 = load %"struct.Kalmar::index_impl.1"** %8
  ret %"class.Kalmar::index.0"* %16
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmIERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %rhs) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %11 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %12 = alloca %"class.Kalmar::__index_leaf", align 4
  %13 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %14 = alloca %"class.Kalmar::index.0"*, align 8
  %15 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %14, align 8
  store %"class.Kalmar::index.0"* %rhs, %"class.Kalmar::index.0"** %15, align 8
  %16 = load %"class.Kalmar::index.0"** %14
  %17 = getelementptr inbounds %"class.Kalmar::index.0"* %16, i32 0, i32 0
  %18 = load %"class.Kalmar::index.0"** %15, align 8
  %19 = getelementptr inbounds %"class.Kalmar::index.0"* %18, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %10, align 8
  store %"struct.Kalmar::index_impl.1"* %19, %"struct.Kalmar::index_impl.1"** %11, align 8
  %20 = load %"struct.Kalmar::index_impl.1"** %10
  %21 = bitcast %"struct.Kalmar::index_impl.1"* %20 to %"class.Kalmar::__index_leaf"*
  %22 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %23 = bitcast %"struct.Kalmar::index_impl.1"* %22 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %23, %"class.Kalmar::__index_leaf"** %9, align 8
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %26 = load i32* %25
  store %"class.Kalmar::__index_leaf"* %21, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %26, i32* %2, align 4
  %27 = load %"class.Kalmar::__index_leaf"** %1
  %28 = load i32* %2, align 4
  %29 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %27, i32 0, i32 0
  %30 = load i32* %29, align 4
  %31 = sub nsw i32 %30, %28
  store i32 %31, i32* %29, align 4
  %32 = bitcast %"class.Kalmar::__index_leaf"* %12 to i8*
  %33 = bitcast %"class.Kalmar::__index_leaf"* %27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %32, i8* %33, i64 8, i32 4, i1 false) #2
  %34 = bitcast %"struct.Kalmar::index_impl.1"* %20 to i8*
  %35 = getelementptr inbounds i8* %34, i64 8
  %36 = bitcast i8* %35 to %"class.Kalmar::__index_leaf.2"*
  %37 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %38 = bitcast %"struct.Kalmar::index_impl.1"* %37 to i8*
  %39 = getelementptr inbounds i8* %38, i64 8
  %40 = bitcast i8* %39 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %40, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %41 = load %"class.Kalmar::__index_leaf.2"** %3
  %42 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %41, i32 0, i32 0
  %43 = load i32* %42
  store %"class.Kalmar::__index_leaf.2"* %36, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %43, i32* %5, align 4
  %44 = load %"class.Kalmar::__index_leaf.2"** %4
  %45 = load i32* %5, align 4
  %46 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %44, i32 0, i32 0
  %47 = load i32* %46, align 4
  %48 = sub nsw i32 %47, %45
  store i32 %48, i32* %46, align 4
  %49 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i8*
  %50 = bitcast %"class.Kalmar::__index_leaf.2"* %44 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %49, i8* %50, i64 8, i32 4, i1 false) #2
  %51 = bitcast %"class.Kalmar::__index_leaf"* %12 to i64*
  %52 = load i64* %51, align 1
  %53 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf"* %6 to i64*
  store i64 %52, i64* %55, align 1
  %56 = bitcast %"class.Kalmar::__index_leaf.2"* %7 to i64*
  store i64 %54, i64* %56, align 1
  store %"struct.Kalmar::index_impl.1"* %20, %"struct.Kalmar::index_impl.1"** %8, align 8
  %57 = load %"struct.Kalmar::index_impl.1"** %8
  ret %"class.Kalmar::index.0"* %16
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmLERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %__r) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %11 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %12 = alloca %"class.Kalmar::__index_leaf", align 4
  %13 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %14 = alloca %"class.Kalmar::index.0"*, align 8
  %15 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %14, align 8
  store %"class.Kalmar::index.0"* %__r, %"class.Kalmar::index.0"** %15, align 8
  %16 = load %"class.Kalmar::index.0"** %14
  %17 = getelementptr inbounds %"class.Kalmar::index.0"* %16, i32 0, i32 0
  %18 = load %"class.Kalmar::index.0"** %15, align 8
  %19 = getelementptr inbounds %"class.Kalmar::index.0"* %18, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %10, align 8
  store %"struct.Kalmar::index_impl.1"* %19, %"struct.Kalmar::index_impl.1"** %11, align 8
  %20 = load %"struct.Kalmar::index_impl.1"** %10
  %21 = bitcast %"struct.Kalmar::index_impl.1"* %20 to %"class.Kalmar::__index_leaf"*
  %22 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %23 = bitcast %"struct.Kalmar::index_impl.1"* %22 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %23, %"class.Kalmar::__index_leaf"** %9, align 8
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %26 = load i32* %25
  store %"class.Kalmar::__index_leaf"* %21, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %26, i32* %2, align 4
  %27 = load %"class.Kalmar::__index_leaf"** %1
  %28 = load i32* %2, align 4
  %29 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %27, i32 0, i32 0
  %30 = load i32* %29, align 4
  %31 = mul nsw i32 %30, %28
  store i32 %31, i32* %29, align 4
  %32 = bitcast %"class.Kalmar::__index_leaf"* %12 to i8*
  %33 = bitcast %"class.Kalmar::__index_leaf"* %27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %32, i8* %33, i64 8, i32 4, i1 false) #2
  %34 = bitcast %"struct.Kalmar::index_impl.1"* %20 to i8*
  %35 = getelementptr inbounds i8* %34, i64 8
  %36 = bitcast i8* %35 to %"class.Kalmar::__index_leaf.2"*
  %37 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %38 = bitcast %"struct.Kalmar::index_impl.1"* %37 to i8*
  %39 = getelementptr inbounds i8* %38, i64 8
  %40 = bitcast i8* %39 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %40, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %41 = load %"class.Kalmar::__index_leaf.2"** %3
  %42 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %41, i32 0, i32 0
  %43 = load i32* %42
  store %"class.Kalmar::__index_leaf.2"* %36, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %43, i32* %5, align 4
  %44 = load %"class.Kalmar::__index_leaf.2"** %4
  %45 = load i32* %5, align 4
  %46 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %44, i32 0, i32 0
  %47 = load i32* %46, align 4
  %48 = mul nsw i32 %47, %45
  store i32 %48, i32* %46, align 4
  %49 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i8*
  %50 = bitcast %"class.Kalmar::__index_leaf.2"* %44 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %49, i8* %50, i64 8, i32 4, i1 false) #2
  %51 = bitcast %"class.Kalmar::__index_leaf"* %12 to i64*
  %52 = load i64* %51, align 1
  %53 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf"* %6 to i64*
  store i64 %52, i64* %55, align 1
  %56 = bitcast %"class.Kalmar::__index_leaf.2"* %7 to i64*
  store i64 %54, i64* %56, align 1
  store %"struct.Kalmar::index_impl.1"* %20, %"struct.Kalmar::index_impl.1"** %8, align 8
  %57 = load %"struct.Kalmar::index_impl.1"** %8
  ret %"class.Kalmar::index.0"* %16
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEdVERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %__r) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %11 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %12 = alloca %"class.Kalmar::__index_leaf", align 4
  %13 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %14 = alloca %"class.Kalmar::index.0"*, align 8
  %15 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %14, align 8
  store %"class.Kalmar::index.0"* %__r, %"class.Kalmar::index.0"** %15, align 8
  %16 = load %"class.Kalmar::index.0"** %14
  %17 = getelementptr inbounds %"class.Kalmar::index.0"* %16, i32 0, i32 0
  %18 = load %"class.Kalmar::index.0"** %15, align 8
  %19 = getelementptr inbounds %"class.Kalmar::index.0"* %18, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %10, align 8
  store %"struct.Kalmar::index_impl.1"* %19, %"struct.Kalmar::index_impl.1"** %11, align 8
  %20 = load %"struct.Kalmar::index_impl.1"** %10
  %21 = bitcast %"struct.Kalmar::index_impl.1"* %20 to %"class.Kalmar::__index_leaf"*
  %22 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %23 = bitcast %"struct.Kalmar::index_impl.1"* %22 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %23, %"class.Kalmar::__index_leaf"** %9, align 8
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %26 = load i32* %25
  store %"class.Kalmar::__index_leaf"* %21, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %26, i32* %2, align 4
  %27 = load %"class.Kalmar::__index_leaf"** %1
  %28 = load i32* %2, align 4
  %29 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %27, i32 0, i32 0
  %30 = load i32* %29, align 4
  %31 = sdiv i32 %30, %28
  store i32 %31, i32* %29, align 4
  %32 = bitcast %"class.Kalmar::__index_leaf"* %12 to i8*
  %33 = bitcast %"class.Kalmar::__index_leaf"* %27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %32, i8* %33, i64 8, i32 4, i1 false) #2
  %34 = bitcast %"struct.Kalmar::index_impl.1"* %20 to i8*
  %35 = getelementptr inbounds i8* %34, i64 8
  %36 = bitcast i8* %35 to %"class.Kalmar::__index_leaf.2"*
  %37 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %38 = bitcast %"struct.Kalmar::index_impl.1"* %37 to i8*
  %39 = getelementptr inbounds i8* %38, i64 8
  %40 = bitcast i8* %39 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %40, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %41 = load %"class.Kalmar::__index_leaf.2"** %3
  %42 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %41, i32 0, i32 0
  %43 = load i32* %42
  store %"class.Kalmar::__index_leaf.2"* %36, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %43, i32* %5, align 4
  %44 = load %"class.Kalmar::__index_leaf.2"** %4
  %45 = load i32* %5, align 4
  %46 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %44, i32 0, i32 0
  %47 = load i32* %46, align 4
  %48 = sdiv i32 %47, %45
  store i32 %48, i32* %46, align 4
  %49 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i8*
  %50 = bitcast %"class.Kalmar::__index_leaf.2"* %44 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %49, i8* %50, i64 8, i32 4, i1 false) #2
  %51 = bitcast %"class.Kalmar::__index_leaf"* %12 to i64*
  %52 = load i64* %51, align 1
  %53 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf"* %6 to i64*
  store i64 %52, i64* %55, align 1
  %56 = bitcast %"class.Kalmar::__index_leaf.2"* %7 to i64*
  store i64 %54, i64* %56, align 1
  store %"struct.Kalmar::index_impl.1"* %20, %"struct.Kalmar::index_impl.1"** %8, align 8
  %57 = load %"struct.Kalmar::index_impl.1"** %8
  ret %"class.Kalmar::index.0"* %16
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EErMERKS1_(%"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"* dereferenceable(16) %__r) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %11 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %12 = alloca %"class.Kalmar::__index_leaf", align 4
  %13 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %14 = alloca %"class.Kalmar::index.0"*, align 8
  %15 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %14, align 8
  store %"class.Kalmar::index.0"* %__r, %"class.Kalmar::index.0"** %15, align 8
  %16 = load %"class.Kalmar::index.0"** %14
  %17 = getelementptr inbounds %"class.Kalmar::index.0"* %16, i32 0, i32 0
  %18 = load %"class.Kalmar::index.0"** %15, align 8
  %19 = getelementptr inbounds %"class.Kalmar::index.0"* %18, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %10, align 8
  store %"struct.Kalmar::index_impl.1"* %19, %"struct.Kalmar::index_impl.1"** %11, align 8
  %20 = load %"struct.Kalmar::index_impl.1"** %10
  %21 = bitcast %"struct.Kalmar::index_impl.1"* %20 to %"class.Kalmar::__index_leaf"*
  %22 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %23 = bitcast %"struct.Kalmar::index_impl.1"* %22 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %23, %"class.Kalmar::__index_leaf"** %9, align 8
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %26 = load i32* %25
  store %"class.Kalmar::__index_leaf"* %21, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %26, i32* %2, align 4
  %27 = load %"class.Kalmar::__index_leaf"** %1
  %28 = load i32* %2, align 4
  %29 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %27, i32 0, i32 0
  %30 = load i32* %29, align 4
  %31 = srem i32 %30, %28
  store i32 %31, i32* %29, align 4
  %32 = bitcast %"class.Kalmar::__index_leaf"* %12 to i8*
  %33 = bitcast %"class.Kalmar::__index_leaf"* %27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %32, i8* %33, i64 8, i32 4, i1 false) #2
  %34 = bitcast %"struct.Kalmar::index_impl.1"* %20 to i8*
  %35 = getelementptr inbounds i8* %34, i64 8
  %36 = bitcast i8* %35 to %"class.Kalmar::__index_leaf.2"*
  %37 = load %"struct.Kalmar::index_impl.1"** %11, align 8
  %38 = bitcast %"struct.Kalmar::index_impl.1"* %37 to i8*
  %39 = getelementptr inbounds i8* %38, i64 8
  %40 = bitcast i8* %39 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %40, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %41 = load %"class.Kalmar::__index_leaf.2"** %3
  %42 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %41, i32 0, i32 0
  %43 = load i32* %42
  store %"class.Kalmar::__index_leaf.2"* %36, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %43, i32* %5, align 4
  %44 = load %"class.Kalmar::__index_leaf.2"** %4
  %45 = load i32* %5, align 4
  %46 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %44, i32 0, i32 0
  %47 = load i32* %46, align 4
  %48 = srem i32 %47, %45
  store i32 %48, i32* %46, align 4
  %49 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i8*
  %50 = bitcast %"class.Kalmar::__index_leaf.2"* %44 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %49, i8* %50, i64 8, i32 4, i1 false) #2
  %51 = bitcast %"class.Kalmar::__index_leaf"* %12 to i64*
  %52 = load i64* %51, align 1
  %53 = bitcast %"class.Kalmar::__index_leaf.2"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf"* %6 to i64*
  store i64 %52, i64* %55, align 1
  %56 = bitcast %"class.Kalmar::__index_leaf.2"* %7 to i64*
  store i64 %54, i64* %56, align 1
  store %"struct.Kalmar::index_impl.1"* %20, %"struct.Kalmar::index_impl.1"** %8, align 8
  %57 = load %"struct.Kalmar::index_impl.1"** %8
  ret %"class.Kalmar::index.0"* %16
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEpLEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.Kalmar::__index_leaf", align 4
  %11 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %12 = alloca %"class.Kalmar::index.0"*, align 8
  %13 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %12, align 8
  store i32 %value, i32* %13, align 4
  %14 = load %"class.Kalmar::index.0"** %12
  %15 = getelementptr inbounds %"class.Kalmar::index.0"* %14, i32 0, i32 0
  %16 = load i32* %13, align 4
  store %"struct.Kalmar::index_impl.1"* %15, %"struct.Kalmar::index_impl.1"** %8, align 8
  store i32 %16, i32* %9, align 4
  %17 = load %"struct.Kalmar::index_impl.1"** %8
  %18 = bitcast %"struct.Kalmar::index_impl.1"* %17 to %"class.Kalmar::__index_leaf"*
  %19 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %6, align 8
  store i32 %19, i32* %7, align 4
  %20 = load %"class.Kalmar::__index_leaf"** %6
  %21 = load i32* %7, align 4
  %22 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %20, i32 0, i32 0
  %23 = load i32* %22, align 4
  %24 = add nsw i32 %23, %21
  store i32 %24, i32* %22, align 4
  %25 = bitcast %"class.Kalmar::__index_leaf"* %10 to i8*
  %26 = bitcast %"class.Kalmar::__index_leaf"* %20 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %25, i8* %26, i64 8, i32 4, i1 false) #2
  %27 = bitcast %"struct.Kalmar::index_impl.1"* %17 to i8*
  %28 = getelementptr inbounds i8* %27, i64 8
  %29 = bitcast i8* %28 to %"class.Kalmar::__index_leaf.2"*
  %30 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf.2"* %29, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %30, i32* %2, align 4
  %31 = load %"class.Kalmar::__index_leaf.2"** %1
  %32 = load i32* %2, align 4
  %33 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %31, i32 0, i32 0
  %34 = load i32* %33, align 4
  %35 = add nsw i32 %34, %32
  store i32 %35, i32* %33, align 4
  %36 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i8*
  %37 = bitcast %"class.Kalmar::__index_leaf.2"* %31 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %36, i8* %37, i64 8, i32 4, i1 false) #2
  %38 = bitcast %"class.Kalmar::__index_leaf"* %10 to i64*
  %39 = load i64* %38, align 1
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i64*
  %41 = load i64* %40, align 1
  %42 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %39, i64* %42, align 1
  %43 = bitcast %"class.Kalmar::__index_leaf.2"* %4 to i64*
  store i64 %41, i64* %43, align 1
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %5, align 8
  %44 = load %"struct.Kalmar::index_impl.1"** %5
  ret %"class.Kalmar::index.0"* %14
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmIEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.Kalmar::__index_leaf", align 4
  %11 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %12 = alloca %"class.Kalmar::index.0"*, align 8
  %13 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %12, align 8
  store i32 %value, i32* %13, align 4
  %14 = load %"class.Kalmar::index.0"** %12
  %15 = getelementptr inbounds %"class.Kalmar::index.0"* %14, i32 0, i32 0
  %16 = load i32* %13, align 4
  store %"struct.Kalmar::index_impl.1"* %15, %"struct.Kalmar::index_impl.1"** %8, align 8
  store i32 %16, i32* %9, align 4
  %17 = load %"struct.Kalmar::index_impl.1"** %8
  %18 = bitcast %"struct.Kalmar::index_impl.1"* %17 to %"class.Kalmar::__index_leaf"*
  %19 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %6, align 8
  store i32 %19, i32* %7, align 4
  %20 = load %"class.Kalmar::__index_leaf"** %6
  %21 = load i32* %7, align 4
  %22 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %20, i32 0, i32 0
  %23 = load i32* %22, align 4
  %24 = sub nsw i32 %23, %21
  store i32 %24, i32* %22, align 4
  %25 = bitcast %"class.Kalmar::__index_leaf"* %10 to i8*
  %26 = bitcast %"class.Kalmar::__index_leaf"* %20 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %25, i8* %26, i64 8, i32 4, i1 false) #2
  %27 = bitcast %"struct.Kalmar::index_impl.1"* %17 to i8*
  %28 = getelementptr inbounds i8* %27, i64 8
  %29 = bitcast i8* %28 to %"class.Kalmar::__index_leaf.2"*
  %30 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf.2"* %29, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %30, i32* %2, align 4
  %31 = load %"class.Kalmar::__index_leaf.2"** %1
  %32 = load i32* %2, align 4
  %33 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %31, i32 0, i32 0
  %34 = load i32* %33, align 4
  %35 = sub nsw i32 %34, %32
  store i32 %35, i32* %33, align 4
  %36 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i8*
  %37 = bitcast %"class.Kalmar::__index_leaf.2"* %31 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %36, i8* %37, i64 8, i32 4, i1 false) #2
  %38 = bitcast %"class.Kalmar::__index_leaf"* %10 to i64*
  %39 = load i64* %38, align 1
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i64*
  %41 = load i64* %40, align 1
  %42 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %39, i64* %42, align 1
  %43 = bitcast %"class.Kalmar::__index_leaf.2"* %4 to i64*
  store i64 %41, i64* %43, align 1
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %5, align 8
  %44 = load %"struct.Kalmar::index_impl.1"** %5
  ret %"class.Kalmar::index.0"* %14
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmLEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.Kalmar::__index_leaf", align 4
  %11 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %12 = alloca %"class.Kalmar::index.0"*, align 8
  %13 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %12, align 8
  store i32 %value, i32* %13, align 4
  %14 = load %"class.Kalmar::index.0"** %12
  %15 = getelementptr inbounds %"class.Kalmar::index.0"* %14, i32 0, i32 0
  %16 = load i32* %13, align 4
  store %"struct.Kalmar::index_impl.1"* %15, %"struct.Kalmar::index_impl.1"** %8, align 8
  store i32 %16, i32* %9, align 4
  %17 = load %"struct.Kalmar::index_impl.1"** %8
  %18 = bitcast %"struct.Kalmar::index_impl.1"* %17 to %"class.Kalmar::__index_leaf"*
  %19 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %6, align 8
  store i32 %19, i32* %7, align 4
  %20 = load %"class.Kalmar::__index_leaf"** %6
  %21 = load i32* %7, align 4
  %22 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %20, i32 0, i32 0
  %23 = load i32* %22, align 4
  %24 = mul nsw i32 %23, %21
  store i32 %24, i32* %22, align 4
  %25 = bitcast %"class.Kalmar::__index_leaf"* %10 to i8*
  %26 = bitcast %"class.Kalmar::__index_leaf"* %20 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %25, i8* %26, i64 8, i32 4, i1 false) #2
  %27 = bitcast %"struct.Kalmar::index_impl.1"* %17 to i8*
  %28 = getelementptr inbounds i8* %27, i64 8
  %29 = bitcast i8* %28 to %"class.Kalmar::__index_leaf.2"*
  %30 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf.2"* %29, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %30, i32* %2, align 4
  %31 = load %"class.Kalmar::__index_leaf.2"** %1
  %32 = load i32* %2, align 4
  %33 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %31, i32 0, i32 0
  %34 = load i32* %33, align 4
  %35 = mul nsw i32 %34, %32
  store i32 %35, i32* %33, align 4
  %36 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i8*
  %37 = bitcast %"class.Kalmar::__index_leaf.2"* %31 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %36, i8* %37, i64 8, i32 4, i1 false) #2
  %38 = bitcast %"class.Kalmar::__index_leaf"* %10 to i64*
  %39 = load i64* %38, align 1
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i64*
  %41 = load i64* %40, align 1
  %42 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %39, i64* %42, align 1
  %43 = bitcast %"class.Kalmar::__index_leaf.2"* %4 to i64*
  store i64 %41, i64* %43, align 1
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %5, align 8
  %44 = load %"struct.Kalmar::index_impl.1"** %5
  ret %"class.Kalmar::index.0"* %14
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEdVEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.Kalmar::__index_leaf", align 4
  %11 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %12 = alloca %"class.Kalmar::index.0"*, align 8
  %13 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %12, align 8
  store i32 %value, i32* %13, align 4
  %14 = load %"class.Kalmar::index.0"** %12
  %15 = getelementptr inbounds %"class.Kalmar::index.0"* %14, i32 0, i32 0
  %16 = load i32* %13, align 4
  store %"struct.Kalmar::index_impl.1"* %15, %"struct.Kalmar::index_impl.1"** %8, align 8
  store i32 %16, i32* %9, align 4
  %17 = load %"struct.Kalmar::index_impl.1"** %8
  %18 = bitcast %"struct.Kalmar::index_impl.1"* %17 to %"class.Kalmar::__index_leaf"*
  %19 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %6, align 8
  store i32 %19, i32* %7, align 4
  %20 = load %"class.Kalmar::__index_leaf"** %6
  %21 = load i32* %7, align 4
  %22 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %20, i32 0, i32 0
  %23 = load i32* %22, align 4
  %24 = sdiv i32 %23, %21
  store i32 %24, i32* %22, align 4
  %25 = bitcast %"class.Kalmar::__index_leaf"* %10 to i8*
  %26 = bitcast %"class.Kalmar::__index_leaf"* %20 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %25, i8* %26, i64 8, i32 4, i1 false) #2
  %27 = bitcast %"struct.Kalmar::index_impl.1"* %17 to i8*
  %28 = getelementptr inbounds i8* %27, i64 8
  %29 = bitcast i8* %28 to %"class.Kalmar::__index_leaf.2"*
  %30 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf.2"* %29, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %30, i32* %2, align 4
  %31 = load %"class.Kalmar::__index_leaf.2"** %1
  %32 = load i32* %2, align 4
  %33 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %31, i32 0, i32 0
  %34 = load i32* %33, align 4
  %35 = sdiv i32 %34, %32
  store i32 %35, i32* %33, align 4
  %36 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i8*
  %37 = bitcast %"class.Kalmar::__index_leaf.2"* %31 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %36, i8* %37, i64 8, i32 4, i1 false) #2
  %38 = bitcast %"class.Kalmar::__index_leaf"* %10 to i64*
  %39 = load i64* %38, align 1
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i64*
  %41 = load i64* %40, align 1
  %42 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %39, i64* %42, align 1
  %43 = bitcast %"class.Kalmar::__index_leaf.2"* %4 to i64*
  store i64 %41, i64* %43, align 1
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %5, align 8
  %44 = load %"struct.Kalmar::index_impl.1"** %5
  ret %"class.Kalmar::index.0"* %14
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EErMEi(%"class.Kalmar::index.0"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.Kalmar::__index_leaf", align 4
  %11 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %12 = alloca %"class.Kalmar::index.0"*, align 8
  %13 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %12, align 8
  store i32 %value, i32* %13, align 4
  %14 = load %"class.Kalmar::index.0"** %12
  %15 = getelementptr inbounds %"class.Kalmar::index.0"* %14, i32 0, i32 0
  %16 = load i32* %13, align 4
  store %"struct.Kalmar::index_impl.1"* %15, %"struct.Kalmar::index_impl.1"** %8, align 8
  store i32 %16, i32* %9, align 4
  %17 = load %"struct.Kalmar::index_impl.1"** %8
  %18 = bitcast %"struct.Kalmar::index_impl.1"* %17 to %"class.Kalmar::__index_leaf"*
  %19 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf"* %18, %"class.Kalmar::__index_leaf"** %6, align 8
  store i32 %19, i32* %7, align 4
  %20 = load %"class.Kalmar::__index_leaf"** %6
  %21 = load i32* %7, align 4
  %22 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %20, i32 0, i32 0
  %23 = load i32* %22, align 4
  %24 = srem i32 %23, %21
  store i32 %24, i32* %22, align 4
  %25 = bitcast %"class.Kalmar::__index_leaf"* %10 to i8*
  %26 = bitcast %"class.Kalmar::__index_leaf"* %20 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %25, i8* %26, i64 8, i32 4, i1 false)
  %27 = bitcast %"struct.Kalmar::index_impl.1"* %17 to i8*
  %28 = getelementptr inbounds i8* %27, i64 8
  %29 = bitcast i8* %28 to %"class.Kalmar::__index_leaf.2"*
  %30 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf.2"* %29, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %30, i32* %2, align 4
  %31 = load %"class.Kalmar::__index_leaf.2"** %1
  %32 = load i32* %2, align 4
  %33 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %31, i32 0, i32 0
  %34 = load i32* %33, align 4
  %35 = srem i32 %34, %32
  store i32 %35, i32* %33, align 4
  %36 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i8*
  %37 = bitcast %"class.Kalmar::__index_leaf.2"* %31 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %36, i8* %37, i64 8, i32 4, i1 false)
  %38 = bitcast %"class.Kalmar::__index_leaf"* %10 to i64*
  %39 = load i64* %38, align 1
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i64*
  %41 = load i64* %40, align 1
  %42 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %39, i64* %42, align 1
  %43 = bitcast %"class.Kalmar::__index_leaf.2"* %4 to i64*
  store i64 %41, i64* %43, align 1
  store %"struct.Kalmar::index_impl.1"* %17, %"struct.Kalmar::index_impl.1"** %5, align 8
  %44 = load %"struct.Kalmar::index_impl.1"** %5
  ret %"class.Kalmar::index.0"* %14
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEppEv(%"class.Kalmar::index.0"* %this) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.Kalmar::__index_leaf", align 4
  %11 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %12 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %12, align 8
  %13 = load %"class.Kalmar::index.0"** %12
  %14 = getelementptr inbounds %"class.Kalmar::index.0"* %13, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %14, %"struct.Kalmar::index_impl.1"** %8, align 8
  store i32 1, i32* %9, align 4
  %15 = load %"struct.Kalmar::index_impl.1"** %8
  %16 = bitcast %"struct.Kalmar::index_impl.1"* %15 to %"class.Kalmar::__index_leaf"*
  %17 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf"* %16, %"class.Kalmar::__index_leaf"** %6, align 8
  store i32 %17, i32* %7, align 4
  %18 = load %"class.Kalmar::__index_leaf"** %6
  %19 = load i32* %7, align 4
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %18, i32 0, i32 0
  %21 = load i32* %20, align 4
  %22 = add nsw i32 %21, %19
  store i32 %22, i32* %20, align 4
  %23 = bitcast %"class.Kalmar::__index_leaf"* %10 to i8*
  %24 = bitcast %"class.Kalmar::__index_leaf"* %18 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %23, i8* %24, i64 8, i32 4, i1 false) #2
  %25 = bitcast %"struct.Kalmar::index_impl.1"* %15 to i8*
  %26 = getelementptr inbounds i8* %25, i64 8
  %27 = bitcast i8* %26 to %"class.Kalmar::__index_leaf.2"*
  %28 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf.2"* %27, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %28, i32* %2, align 4
  %29 = load %"class.Kalmar::__index_leaf.2"** %1
  %30 = load i32* %2, align 4
  %31 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %29, i32 0, i32 0
  %32 = load i32* %31, align 4
  %33 = add nsw i32 %32, %30
  store i32 %33, i32* %31, align 4
  %34 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i8*
  %35 = bitcast %"class.Kalmar::__index_leaf.2"* %29 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %34, i8* %35, i64 8, i32 4, i1 false) #2
  %36 = bitcast %"class.Kalmar::__index_leaf"* %10 to i64*
  %37 = load i64* %36, align 1
  %38 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i64*
  %39 = load i64* %38, align 1
  %40 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %37, i64* %40, align 1
  %41 = bitcast %"class.Kalmar::__index_leaf.2"* %4 to i64*
  store i64 %39, i64* %41, align 1
  store %"struct.Kalmar::index_impl.1"* %15, %"struct.Kalmar::index_impl.1"** %5, align 8
  %42 = load %"struct.Kalmar::index_impl.1"** %5
  ret %"class.Kalmar::index.0"* %13
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEppEi(%"class.Kalmar::index.0"* noalias sret %agg.result, %"class.Kalmar::index.0"* %this, i32) #0 align 2 {
  %2 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::__index_leaf", align 8
  %5 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %6 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %7 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"class.Kalmar::__index_leaf", align 4
  %12 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %13 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %14 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %15 = alloca i32, align 4
  %16 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %17 = alloca i32, align 4
  %18 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %22 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %23 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %24 = alloca %"class.Kalmar::index.0"*, align 8
  %25 = alloca %"class.Kalmar::index.0"*, align 8
  %26 = alloca %"class.Kalmar::index.0"*, align 8
  %27 = alloca %"class.Kalmar::index.0"*, align 8
  %28 = alloca %"class.Kalmar::index.0"*, align 8
  %29 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %28, align 8
  store i32 %0, i32* %29, align 4
  %30 = load %"class.Kalmar::index.0"** %28
  store %"class.Kalmar::index.0"* %agg.result, %"class.Kalmar::index.0"** %26, align 8
  store %"class.Kalmar::index.0"* %30, %"class.Kalmar::index.0"** %27, align 8
  %31 = load %"class.Kalmar::index.0"** %26
  %32 = load %"class.Kalmar::index.0"** %27
  store %"class.Kalmar::index.0"* %31, %"class.Kalmar::index.0"** %24, align 8
  store %"class.Kalmar::index.0"* %32, %"class.Kalmar::index.0"** %25, align 8
  %33 = load %"class.Kalmar::index.0"** %24
  %34 = getelementptr inbounds %"class.Kalmar::index.0"* %33, i32 0, i32 0
  %35 = load %"class.Kalmar::index.0"** %25, align 8
  %36 = getelementptr inbounds %"class.Kalmar::index.0"* %35, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %34, %"struct.Kalmar::index_impl.1"** %22, align 8
  store %"struct.Kalmar::index_impl.1"* %36, %"struct.Kalmar::index_impl.1"** %23, align 8
  %37 = load %"struct.Kalmar::index_impl.1"** %22
  %38 = load %"struct.Kalmar::index_impl.1"** %23, align 8
  %39 = bitcast %"struct.Kalmar::index_impl.1"* %38 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %39, %"class.Kalmar::__index_leaf"** %21, align 8
  %40 = load %"class.Kalmar::__index_leaf"** %21
  %41 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %40, i32 0, i32 0
  %42 = load i32* %41
  %43 = load %"struct.Kalmar::index_impl.1"** %23, align 8
  %44 = bitcast %"struct.Kalmar::index_impl.1"* %43 to i8*
  %45 = getelementptr inbounds i8* %44, i64 8
  %46 = bitcast i8* %45 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %46, %"class.Kalmar::__index_leaf.2"** %13, align 8
  %47 = load %"class.Kalmar::__index_leaf.2"** %13
  %48 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %47, i32 0, i32 0
  %49 = load i32* %48
  store %"struct.Kalmar::index_impl.1"* %37, %"struct.Kalmar::index_impl.1"** %18, align 8
  store i32 %42, i32* %19, align 4
  store i32 %49, i32* %20, align 4
  %50 = load %"struct.Kalmar::index_impl.1"** %18
  %51 = bitcast %"struct.Kalmar::index_impl.1"* %50 to %"class.Kalmar::__index_leaf"*
  %52 = load i32* %19, align 4
  store %"class.Kalmar::__index_leaf"* %51, %"class.Kalmar::__index_leaf"** %16, align 8
  store i32 %52, i32* %17, align 4
  %53 = load %"class.Kalmar::__index_leaf"** %16
  %54 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %53, i32 0, i32 0
  %55 = load i32* %17, align 4
  store i32 %55, i32* %54, align 4
  %56 = bitcast %"struct.Kalmar::index_impl.1"* %50 to i8*
  %57 = getelementptr inbounds i8* %56, i64 8
  %58 = bitcast i8* %57 to %"class.Kalmar::__index_leaf.2"*
  %59 = load i32* %20, align 4
  store %"class.Kalmar::__index_leaf.2"* %58, %"class.Kalmar::__index_leaf.2"** %14, align 8
  store i32 %59, i32* %15, align 4
  %60 = load %"class.Kalmar::__index_leaf.2"** %14
  %61 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %60, i32 0, i32 0
  %62 = load i32* %15, align 4
  store i32 %62, i32* %61, align 4
  %63 = getelementptr inbounds %"class.Kalmar::index.0"* %30, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %63, %"struct.Kalmar::index_impl.1"** %9, align 8
  store i32 1, i32* %10, align 4
  %64 = load %"struct.Kalmar::index_impl.1"** %9
  %65 = bitcast %"struct.Kalmar::index_impl.1"* %64 to %"class.Kalmar::__index_leaf"*
  %66 = load i32* %10, align 4
  store %"class.Kalmar::__index_leaf"* %65, %"class.Kalmar::__index_leaf"** %7, align 8
  store i32 %66, i32* %8, align 4
  %67 = load %"class.Kalmar::__index_leaf"** %7
  %68 = load i32* %8, align 4
  %69 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %67, i32 0, i32 0
  %70 = load i32* %69, align 4
  %71 = add nsw i32 %70, %68
  store i32 %71, i32* %69, align 4
  %72 = bitcast %"class.Kalmar::__index_leaf"* %11 to i8*
  %73 = bitcast %"class.Kalmar::__index_leaf"* %67 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %72, i8* %73, i64 8, i32 4, i1 false) #2
  %74 = bitcast %"struct.Kalmar::index_impl.1"* %64 to i8*
  %75 = getelementptr inbounds i8* %74, i64 8
  %76 = bitcast i8* %75 to %"class.Kalmar::__index_leaf.2"*
  %77 = load i32* %10, align 4
  store %"class.Kalmar::__index_leaf.2"* %76, %"class.Kalmar::__index_leaf.2"** %2, align 8
  store i32 %77, i32* %3, align 4
  %78 = load %"class.Kalmar::__index_leaf.2"** %2
  %79 = load i32* %3, align 4
  %80 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %78, i32 0, i32 0
  %81 = load i32* %80, align 4
  %82 = add nsw i32 %81, %79
  store i32 %82, i32* %80, align 4
  %83 = bitcast %"class.Kalmar::__index_leaf.2"* %12 to i8*
  %84 = bitcast %"class.Kalmar::__index_leaf.2"* %78 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %83, i8* %84, i64 8, i32 4, i1 false) #2
  %85 = bitcast %"class.Kalmar::__index_leaf"* %11 to i64*
  %86 = load i64* %85, align 1
  %87 = bitcast %"class.Kalmar::__index_leaf.2"* %12 to i64*
  %88 = load i64* %87, align 1
  %89 = bitcast %"class.Kalmar::__index_leaf"* %4 to i64*
  store i64 %86, i64* %89, align 1
  %90 = bitcast %"class.Kalmar::__index_leaf.2"* %5 to i64*
  store i64 %88, i64* %90, align 1
  store %"struct.Kalmar::index_impl.1"* %64, %"struct.Kalmar::index_impl.1"** %6, align 8
  %91 = load %"struct.Kalmar::index_impl.1"** %6
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(16) %"class.Kalmar::index.0"* @_ZN6Kalmar5indexILi2EEmmEv(%"class.Kalmar::index.0"* %this) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf", align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %5 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %"class.Kalmar::__index_leaf", align 4
  %11 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %12 = alloca %"class.Kalmar::index.0"*, align 8
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %12, align 8
  %13 = load %"class.Kalmar::index.0"** %12
  %14 = getelementptr inbounds %"class.Kalmar::index.0"* %13, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %14, %"struct.Kalmar::index_impl.1"** %8, align 8
  store i32 1, i32* %9, align 4
  %15 = load %"struct.Kalmar::index_impl.1"** %8
  %16 = bitcast %"struct.Kalmar::index_impl.1"* %15 to %"class.Kalmar::__index_leaf"*
  %17 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf"* %16, %"class.Kalmar::__index_leaf"** %6, align 8
  store i32 %17, i32* %7, align 4
  %18 = load %"class.Kalmar::__index_leaf"** %6
  %19 = load i32* %7, align 4
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %18, i32 0, i32 0
  %21 = load i32* %20, align 4
  %22 = sub nsw i32 %21, %19
  store i32 %22, i32* %20, align 4
  %23 = bitcast %"class.Kalmar::__index_leaf"* %10 to i8*
  %24 = bitcast %"class.Kalmar::__index_leaf"* %18 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %23, i8* %24, i64 8, i32 4, i1 false) #2
  %25 = bitcast %"struct.Kalmar::index_impl.1"* %15 to i8*
  %26 = getelementptr inbounds i8* %25, i64 8
  %27 = bitcast i8* %26 to %"class.Kalmar::__index_leaf.2"*
  %28 = load i32* %9, align 4
  store %"class.Kalmar::__index_leaf.2"* %27, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %28, i32* %2, align 4
  %29 = load %"class.Kalmar::__index_leaf.2"** %1
  %30 = load i32* %2, align 4
  %31 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %29, i32 0, i32 0
  %32 = load i32* %31, align 4
  %33 = sub nsw i32 %32, %30
  store i32 %33, i32* %31, align 4
  %34 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i8*
  %35 = bitcast %"class.Kalmar::__index_leaf.2"* %29 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %34, i8* %35, i64 8, i32 4, i1 false) #2
  %36 = bitcast %"class.Kalmar::__index_leaf"* %10 to i64*
  %37 = load i64* %36, align 1
  %38 = bitcast %"class.Kalmar::__index_leaf.2"* %11 to i64*
  %39 = load i64* %38, align 1
  %40 = bitcast %"class.Kalmar::__index_leaf"* %3 to i64*
  store i64 %37, i64* %40, align 1
  %41 = bitcast %"class.Kalmar::__index_leaf.2"* %4 to i64*
  store i64 %39, i64* %41, align 1
  store %"struct.Kalmar::index_impl.1"* %15, %"struct.Kalmar::index_impl.1"** %5, align 8
  %42 = load %"struct.Kalmar::index_impl.1"** %5
  ret %"class.Kalmar::index.0"* %13
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi2EEmmEi(%"class.Kalmar::index.0"* noalias sret %agg.result, %"class.Kalmar::index.0"* %this, i32) #0 align 2 {
  %2 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::__index_leaf", align 8
  %5 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %6 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %7 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"class.Kalmar::__index_leaf", align 4
  %12 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %13 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %14 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %15 = alloca i32, align 4
  %16 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %17 = alloca i32, align 4
  %18 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %22 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %23 = alloca %"struct.Kalmar::index_impl.1"*, align 8
  %24 = alloca %"class.Kalmar::index.0"*, align 8
  %25 = alloca %"class.Kalmar::index.0"*, align 8
  %26 = alloca %"class.Kalmar::index.0"*, align 8
  %27 = alloca %"class.Kalmar::index.0"*, align 8
  %28 = alloca %"class.Kalmar::index.0"*, align 8
  %29 = alloca i32, align 4
  store %"class.Kalmar::index.0"* %this, %"class.Kalmar::index.0"** %28, align 8
  store i32 %0, i32* %29, align 4
  %30 = load %"class.Kalmar::index.0"** %28
  store %"class.Kalmar::index.0"* %agg.result, %"class.Kalmar::index.0"** %26, align 8
  store %"class.Kalmar::index.0"* %30, %"class.Kalmar::index.0"** %27, align 8
  %31 = load %"class.Kalmar::index.0"** %26
  %32 = load %"class.Kalmar::index.0"** %27
  store %"class.Kalmar::index.0"* %31, %"class.Kalmar::index.0"** %24, align 8
  store %"class.Kalmar::index.0"* %32, %"class.Kalmar::index.0"** %25, align 8
  %33 = load %"class.Kalmar::index.0"** %24
  %34 = getelementptr inbounds %"class.Kalmar::index.0"* %33, i32 0, i32 0
  %35 = load %"class.Kalmar::index.0"** %25, align 8
  %36 = getelementptr inbounds %"class.Kalmar::index.0"* %35, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %34, %"struct.Kalmar::index_impl.1"** %22, align 8
  store %"struct.Kalmar::index_impl.1"* %36, %"struct.Kalmar::index_impl.1"** %23, align 8
  %37 = load %"struct.Kalmar::index_impl.1"** %22
  %38 = load %"struct.Kalmar::index_impl.1"** %23, align 8
  %39 = bitcast %"struct.Kalmar::index_impl.1"* %38 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %39, %"class.Kalmar::__index_leaf"** %21, align 8
  %40 = load %"class.Kalmar::__index_leaf"** %21
  %41 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %40, i32 0, i32 0
  %42 = load i32* %41
  %43 = load %"struct.Kalmar::index_impl.1"** %23, align 8
  %44 = bitcast %"struct.Kalmar::index_impl.1"* %43 to i8*
  %45 = getelementptr inbounds i8* %44, i64 8
  %46 = bitcast i8* %45 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %46, %"class.Kalmar::__index_leaf.2"** %13, align 8
  %47 = load %"class.Kalmar::__index_leaf.2"** %13
  %48 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %47, i32 0, i32 0
  %49 = load i32* %48
  store %"struct.Kalmar::index_impl.1"* %37, %"struct.Kalmar::index_impl.1"** %18, align 8
  store i32 %42, i32* %19, align 4
  store i32 %49, i32* %20, align 4
  %50 = load %"struct.Kalmar::index_impl.1"** %18
  %51 = bitcast %"struct.Kalmar::index_impl.1"* %50 to %"class.Kalmar::__index_leaf"*
  %52 = load i32* %19, align 4
  store %"class.Kalmar::__index_leaf"* %51, %"class.Kalmar::__index_leaf"** %16, align 8
  store i32 %52, i32* %17, align 4
  %53 = load %"class.Kalmar::__index_leaf"** %16
  %54 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %53, i32 0, i32 0
  %55 = load i32* %17, align 4
  store i32 %55, i32* %54, align 4
  %56 = bitcast %"struct.Kalmar::index_impl.1"* %50 to i8*
  %57 = getelementptr inbounds i8* %56, i64 8
  %58 = bitcast i8* %57 to %"class.Kalmar::__index_leaf.2"*
  %59 = load i32* %20, align 4
  store %"class.Kalmar::__index_leaf.2"* %58, %"class.Kalmar::__index_leaf.2"** %14, align 8
  store i32 %59, i32* %15, align 4
  %60 = load %"class.Kalmar::__index_leaf.2"** %14
  %61 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %60, i32 0, i32 0
  %62 = load i32* %15, align 4
  store i32 %62, i32* %61, align 4
  %63 = getelementptr inbounds %"class.Kalmar::index.0"* %30, i32 0, i32 0
  store %"struct.Kalmar::index_impl.1"* %63, %"struct.Kalmar::index_impl.1"** %9, align 8
  store i32 1, i32* %10, align 4
  %64 = load %"struct.Kalmar::index_impl.1"** %9
  %65 = bitcast %"struct.Kalmar::index_impl.1"* %64 to %"class.Kalmar::__index_leaf"*
  %66 = load i32* %10, align 4
  store %"class.Kalmar::__index_leaf"* %65, %"class.Kalmar::__index_leaf"** %7, align 8
  store i32 %66, i32* %8, align 4
  %67 = load %"class.Kalmar::__index_leaf"** %7
  %68 = load i32* %8, align 4
  %69 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %67, i32 0, i32 0
  %70 = load i32* %69, align 4
  %71 = sub nsw i32 %70, %68
  store i32 %71, i32* %69, align 4
  %72 = bitcast %"class.Kalmar::__index_leaf"* %11 to i8*
  %73 = bitcast %"class.Kalmar::__index_leaf"* %67 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %72, i8* %73, i64 8, i32 4, i1 false) #2
  %74 = bitcast %"struct.Kalmar::index_impl.1"* %64 to i8*
  %75 = getelementptr inbounds i8* %74, i64 8
  %76 = bitcast i8* %75 to %"class.Kalmar::__index_leaf.2"*
  %77 = load i32* %10, align 4
  store %"class.Kalmar::__index_leaf.2"* %76, %"class.Kalmar::__index_leaf.2"** %2, align 8
  store i32 %77, i32* %3, align 4
  %78 = load %"class.Kalmar::__index_leaf.2"** %2
  %79 = load i32* %3, align 4
  %80 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %78, i32 0, i32 0
  %81 = load i32* %80, align 4
  %82 = sub nsw i32 %81, %79
  store i32 %82, i32* %80, align 4
  %83 = bitcast %"class.Kalmar::__index_leaf.2"* %12 to i8*
  %84 = bitcast %"class.Kalmar::__index_leaf.2"* %78 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %83, i8* %84, i64 8, i32 4, i1 false) #2
  %85 = bitcast %"class.Kalmar::__index_leaf"* %11 to i64*
  %86 = load i64* %85, align 1
  %87 = bitcast %"class.Kalmar::__index_leaf.2"* %12 to i64*
  %88 = load i64* %87, align 1
  %89 = bitcast %"class.Kalmar::__index_leaf"* %4 to i64*
  store i64 %86, i64* %89, align 1
  %90 = bitcast %"class.Kalmar::__index_leaf.2"* %5 to i64*
  store i64 %88, i64* %90, align 1
  store %"struct.Kalmar::index_impl.1"* %64, %"struct.Kalmar::index_impl.1"** %6, align 8
  %91 = load %"struct.Kalmar::index_impl.1"** %6
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2Ev(%"class.Kalmar::index.3"* %this) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %8, align 8
  %9 = load %"class.Kalmar::index.3"** %8
  %10 = getelementptr inbounds %"class.Kalmar::index.3"* %9, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %10, %"struct.Kalmar::index_impl.4"** %7, align 8
  %11 = load %"struct.Kalmar::index_impl.4"** %7
  %12 = bitcast %"struct.Kalmar::index_impl.4"* %11 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %12, %"class.Kalmar::__index_leaf"** %5, align 8
  store i32 0, i32* %6, align 4
  %13 = load %"class.Kalmar::__index_leaf"** %5
  %14 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %13, i32 0, i32 0
  %15 = load i32* %6, align 4
  store i32 %15, i32* %14, align 4
  %16 = bitcast %"struct.Kalmar::index_impl.4"* %11 to i8*
  %17 = getelementptr inbounds i8* %16, i64 8
  %18 = bitcast i8* %17 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %18, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 0, i32* %2, align 4
  %19 = load %"class.Kalmar::__index_leaf.2"** %1
  %20 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %19, i32 0, i32 0
  %21 = load i32* %2, align 4
  store i32 %21, i32* %20, align 4
  %22 = bitcast %"struct.Kalmar::index_impl.4"* %11 to i8*
  %23 = getelementptr inbounds i8* %22, i64 16
  %24 = bitcast i8* %23 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %24, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 0, i32* %4, align 4
  %25 = load %"class.Kalmar::__index_leaf.5"** %3
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %25, i32 0, i32 0
  %27 = load i32* %4, align 4
  store i32 %27, i32* %26, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1Ev(%"class.Kalmar::index.3"* %this) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca %"class.Kalmar::index.3"*, align 8
  %9 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %9, align 8
  %10 = load %"class.Kalmar::index.3"** %9
  store %"class.Kalmar::index.3"* %10, %"class.Kalmar::index.3"** %8, align 8
  %11 = load %"class.Kalmar::index.3"** %8
  %12 = getelementptr inbounds %"class.Kalmar::index.3"* %11, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %12, %"struct.Kalmar::index_impl.4"** %7, align 8
  %13 = load %"struct.Kalmar::index_impl.4"** %7
  %14 = bitcast %"struct.Kalmar::index_impl.4"* %13 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %14, %"class.Kalmar::__index_leaf"** %5, align 8
  store i32 0, i32* %6, align 4
  %15 = load %"class.Kalmar::__index_leaf"** %5
  %16 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %15, i32 0, i32 0
  %17 = load i32* %6, align 4
  store i32 %17, i32* %16, align 4
  %18 = bitcast %"struct.Kalmar::index_impl.4"* %13 to i8*
  %19 = getelementptr inbounds i8* %18, i64 8
  %20 = bitcast i8* %19 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %20, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 0, i32* %2, align 4
  %21 = load %"class.Kalmar::__index_leaf.2"** %1
  %22 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %21, i32 0, i32 0
  %23 = load i32* %2, align 4
  store i32 %23, i32* %22, align 4
  %24 = bitcast %"struct.Kalmar::index_impl.4"* %13 to i8*
  %25 = getelementptr inbounds i8* %24, i64 16
  %26 = bitcast i8* %25 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %26, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 0, i32* %4, align 4
  %27 = load %"class.Kalmar::__index_leaf.5"** %3
  %28 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %27, i32 0, i32 0
  %29 = load i32* %4, align 4
  store i32 %29, i32* %28, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2ERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %other) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %16 = alloca %"class.Kalmar::index.3"*, align 8
  %17 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %16, align 8
  store %"class.Kalmar::index.3"* %other, %"class.Kalmar::index.3"** %17, align 8
  %18 = load %"class.Kalmar::index.3"** %16
  %19 = getelementptr inbounds %"class.Kalmar::index.3"* %18, i32 0, i32 0
  %20 = load %"class.Kalmar::index.3"** %17, align 8
  %21 = getelementptr inbounds %"class.Kalmar::index.3"* %20, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %19, %"struct.Kalmar::index_impl.4"** %14, align 8
  store %"struct.Kalmar::index_impl.4"* %21, %"struct.Kalmar::index_impl.4"** %15, align 8
  %22 = load %"struct.Kalmar::index_impl.4"** %14
  %23 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %24 = bitcast %"struct.Kalmar::index_impl.4"* %23 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %24, %"class.Kalmar::__index_leaf"** %13, align 8
  %25 = load %"class.Kalmar::__index_leaf"** %13
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %25, i32 0, i32 0
  %27 = load i32* %26
  %28 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %29 = bitcast %"struct.Kalmar::index_impl.4"* %28 to i8*
  %30 = getelementptr inbounds i8* %29, i64 8
  %31 = bitcast i8* %30 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %31, %"class.Kalmar::__index_leaf.2"** %1, align 8
  %32 = load %"class.Kalmar::__index_leaf.2"** %1
  %33 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %32, i32 0, i32 0
  %34 = load i32* %33
  %35 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %36 = bitcast %"struct.Kalmar::index_impl.4"* %35 to i8*
  %37 = getelementptr inbounds i8* %36, i64 16
  %38 = bitcast i8* %37 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %38, %"class.Kalmar::__index_leaf.5"** %2, align 8
  %39 = load %"class.Kalmar::__index_leaf.5"** %2
  %40 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %39, i32 0, i32 0
  %41 = load i32* %40
  store %"struct.Kalmar::index_impl.4"* %22, %"struct.Kalmar::index_impl.4"** %9, align 8
  store i32 %27, i32* %10, align 4
  store i32 %34, i32* %11, align 4
  store i32 %41, i32* %12, align 4
  %42 = load %"struct.Kalmar::index_impl.4"** %9
  %43 = bitcast %"struct.Kalmar::index_impl.4"* %42 to %"class.Kalmar::__index_leaf"*
  %44 = load i32* %10, align 4
  store %"class.Kalmar::__index_leaf"* %43, %"class.Kalmar::__index_leaf"** %7, align 8
  store i32 %44, i32* %8, align 4
  %45 = load %"class.Kalmar::__index_leaf"** %7
  %46 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %45, i32 0, i32 0
  %47 = load i32* %8, align 4
  store i32 %47, i32* %46, align 4
  %48 = bitcast %"struct.Kalmar::index_impl.4"* %42 to i8*
  %49 = getelementptr inbounds i8* %48, i64 8
  %50 = bitcast i8* %49 to %"class.Kalmar::__index_leaf.2"*
  %51 = load i32* %11, align 4
  store %"class.Kalmar::__index_leaf.2"* %50, %"class.Kalmar::__index_leaf.2"** %3, align 8
  store i32 %51, i32* %4, align 4
  %52 = load %"class.Kalmar::__index_leaf.2"** %3
  %53 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %52, i32 0, i32 0
  %54 = load i32* %4, align 4
  store i32 %54, i32* %53, align 4
  %55 = bitcast %"struct.Kalmar::index_impl.4"* %42 to i8*
  %56 = getelementptr inbounds i8* %55, i64 16
  %57 = bitcast i8* %56 to %"class.Kalmar::__index_leaf.5"*
  %58 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.5"* %57, %"class.Kalmar::__index_leaf.5"** %5, align 8
  store i32 %58, i32* %6, align 4
  %59 = load %"class.Kalmar::__index_leaf.5"** %5
  %60 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %59, i32 0, i32 0
  %61 = load i32* %6, align 4
  store i32 %61, i32* %60, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1ERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %other) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %16 = alloca %"class.Kalmar::index.3"*, align 8
  %17 = alloca %"class.Kalmar::index.3"*, align 8
  %18 = alloca %"class.Kalmar::index.3"*, align 8
  %19 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %18, align 8
  store %"class.Kalmar::index.3"* %other, %"class.Kalmar::index.3"** %19, align 8
  %20 = load %"class.Kalmar::index.3"** %18
  %21 = load %"class.Kalmar::index.3"** %19
  store %"class.Kalmar::index.3"* %20, %"class.Kalmar::index.3"** %16, align 8
  store %"class.Kalmar::index.3"* %21, %"class.Kalmar::index.3"** %17, align 8
  %22 = load %"class.Kalmar::index.3"** %16
  %23 = getelementptr inbounds %"class.Kalmar::index.3"* %22, i32 0, i32 0
  %24 = load %"class.Kalmar::index.3"** %17, align 8
  %25 = getelementptr inbounds %"class.Kalmar::index.3"* %24, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %23, %"struct.Kalmar::index_impl.4"** %14, align 8
  store %"struct.Kalmar::index_impl.4"* %25, %"struct.Kalmar::index_impl.4"** %15, align 8
  %26 = load %"struct.Kalmar::index_impl.4"** %14
  %27 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %28 = bitcast %"struct.Kalmar::index_impl.4"* %27 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %28, %"class.Kalmar::__index_leaf"** %13, align 8
  %29 = load %"class.Kalmar::__index_leaf"** %13
  %30 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %29, i32 0, i32 0
  %31 = load i32* %30
  %32 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %33 = bitcast %"struct.Kalmar::index_impl.4"* %32 to i8*
  %34 = getelementptr inbounds i8* %33, i64 8
  %35 = bitcast i8* %34 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %35, %"class.Kalmar::__index_leaf.2"** %1, align 8
  %36 = load %"class.Kalmar::__index_leaf.2"** %1
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %36, i32 0, i32 0
  %38 = load i32* %37
  %39 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %40 = bitcast %"struct.Kalmar::index_impl.4"* %39 to i8*
  %41 = getelementptr inbounds i8* %40, i64 16
  %42 = bitcast i8* %41 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %42, %"class.Kalmar::__index_leaf.5"** %2, align 8
  %43 = load %"class.Kalmar::__index_leaf.5"** %2
  %44 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %43, i32 0, i32 0
  %45 = load i32* %44
  store %"struct.Kalmar::index_impl.4"* %26, %"struct.Kalmar::index_impl.4"** %9, align 8
  store i32 %31, i32* %10, align 4
  store i32 %38, i32* %11, align 4
  store i32 %45, i32* %12, align 4
  %46 = load %"struct.Kalmar::index_impl.4"** %9
  %47 = bitcast %"struct.Kalmar::index_impl.4"* %46 to %"class.Kalmar::__index_leaf"*
  %48 = load i32* %10, align 4
  store %"class.Kalmar::__index_leaf"* %47, %"class.Kalmar::__index_leaf"** %7, align 8
  store i32 %48, i32* %8, align 4
  %49 = load %"class.Kalmar::__index_leaf"** %7
  %50 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %49, i32 0, i32 0
  %51 = load i32* %8, align 4
  store i32 %51, i32* %50, align 4
  %52 = bitcast %"struct.Kalmar::index_impl.4"* %46 to i8*
  %53 = getelementptr inbounds i8* %52, i64 8
  %54 = bitcast i8* %53 to %"class.Kalmar::__index_leaf.2"*
  %55 = load i32* %11, align 4
  store %"class.Kalmar::__index_leaf.2"* %54, %"class.Kalmar::__index_leaf.2"** %3, align 8
  store i32 %55, i32* %4, align 4
  %56 = load %"class.Kalmar::__index_leaf.2"** %3
  %57 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %56, i32 0, i32 0
  %58 = load i32* %4, align 4
  store i32 %58, i32* %57, align 4
  %59 = bitcast %"struct.Kalmar::index_impl.4"* %46 to i8*
  %60 = getelementptr inbounds i8* %59, i64 16
  %61 = bitcast i8* %60 to %"class.Kalmar::__index_leaf.5"*
  %62 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.5"* %61, %"class.Kalmar::__index_leaf.5"** %5, align 8
  store i32 %62, i32* %6, align 4
  %63 = load %"class.Kalmar::__index_leaf.5"** %5
  %64 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %63, i32 0, i32 0
  %65 = load i32* %6, align 4
  store i32 %65, i32* %64, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2Ei(%"class.Kalmar::index.3"* %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::index.3"*, align 8
  %10 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %9, align 8
  store i32 %i0, i32* %10, align 4
  %11 = load %"class.Kalmar::index.3"** %9
  %12 = getelementptr inbounds %"class.Kalmar::index.3"* %11, i32 0, i32 0
  %13 = load i32* %10, align 4
  store %"struct.Kalmar::index_impl.4"* %12, %"struct.Kalmar::index_impl.4"** %7, align 8
  store i32 %13, i32* %8, align 4
  %14 = load %"struct.Kalmar::index_impl.4"** %7
  %15 = bitcast %"struct.Kalmar::index_impl.4"* %14 to %"class.Kalmar::__index_leaf"*
  %16 = load i32* %8, align 4
  store %"class.Kalmar::__index_leaf"* %15, %"class.Kalmar::__index_leaf"** %5, align 8
  store i32 %16, i32* %6, align 4
  %17 = load %"class.Kalmar::__index_leaf"** %5
  %18 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %17, i32 0, i32 0
  %19 = load i32* %6, align 4
  store i32 %19, i32* %18, align 4
  %20 = bitcast %"struct.Kalmar::index_impl.4"* %14 to i8*
  %21 = getelementptr inbounds i8* %20, i64 8
  %22 = bitcast i8* %21 to %"class.Kalmar::__index_leaf.2"*
  %23 = load i32* %8, align 4
  store %"class.Kalmar::__index_leaf.2"* %22, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %23, i32* %2, align 4
  %24 = load %"class.Kalmar::__index_leaf.2"** %1
  %25 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %24, i32 0, i32 0
  %26 = load i32* %2, align 4
  store i32 %26, i32* %25, align 4
  %27 = bitcast %"struct.Kalmar::index_impl.4"* %14 to i8*
  %28 = getelementptr inbounds i8* %27, i64 16
  %29 = bitcast i8* %28 to %"class.Kalmar::__index_leaf.5"*
  %30 = load i32* %8, align 4
  store %"class.Kalmar::__index_leaf.5"* %29, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %30, i32* %4, align 4
  %31 = load %"class.Kalmar::__index_leaf.5"** %3
  %32 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %31, i32 0, i32 0
  %33 = load i32* %4, align 4
  store i32 %33, i32* %32, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1Ei(%"class.Kalmar::index.3"* %this, i32 %i0) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::index.3"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"class.Kalmar::index.3"*, align 8
  %12 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %11, align 8
  store i32 %i0, i32* %12, align 4
  %13 = load %"class.Kalmar::index.3"** %11
  %14 = load i32* %12, align 4
  store %"class.Kalmar::index.3"* %13, %"class.Kalmar::index.3"** %9, align 8
  store i32 %14, i32* %10, align 4
  %15 = load %"class.Kalmar::index.3"** %9
  %16 = getelementptr inbounds %"class.Kalmar::index.3"* %15, i32 0, i32 0
  %17 = load i32* %10, align 4
  store %"struct.Kalmar::index_impl.4"* %16, %"struct.Kalmar::index_impl.4"** %7, align 8
  store i32 %17, i32* %8, align 4
  %18 = load %"struct.Kalmar::index_impl.4"** %7
  %19 = bitcast %"struct.Kalmar::index_impl.4"* %18 to %"class.Kalmar::__index_leaf"*
  %20 = load i32* %8, align 4
  store %"class.Kalmar::__index_leaf"* %19, %"class.Kalmar::__index_leaf"** %5, align 8
  store i32 %20, i32* %6, align 4
  %21 = load %"class.Kalmar::__index_leaf"** %5
  %22 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %21, i32 0, i32 0
  %23 = load i32* %6, align 4
  store i32 %23, i32* %22, align 4
  %24 = bitcast %"struct.Kalmar::index_impl.4"* %18 to i8*
  %25 = getelementptr inbounds i8* %24, i64 8
  %26 = bitcast i8* %25 to %"class.Kalmar::__index_leaf.2"*
  %27 = load i32* %8, align 4
  store %"class.Kalmar::__index_leaf.2"* %26, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %27, i32* %2, align 4
  %28 = load %"class.Kalmar::__index_leaf.2"** %1
  %29 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %28, i32 0, i32 0
  %30 = load i32* %2, align 4
  store i32 %30, i32* %29, align 4
  %31 = bitcast %"struct.Kalmar::index_impl.4"* %18 to i8*
  %32 = getelementptr inbounds i8* %31, i64 16
  %33 = bitcast i8* %32 to %"class.Kalmar::__index_leaf.5"*
  %34 = load i32* %8, align 4
  store %"class.Kalmar::__index_leaf.5"* %33, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %34, i32* %4, align 4
  %35 = load %"class.Kalmar::__index_leaf.5"** %3
  %36 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %35, i32 0, i32 0
  %37 = load i32* %4, align 4
  store i32 %37, i32* %36, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2EPKi(%"class.Kalmar::index.3"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca i32*, align 8
  %9 = alloca %"class.Kalmar::index.3"*, align 8
  %10 = alloca i32*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %9, align 8
  store i32* %components, i32** %10, align 8
  %11 = load %"class.Kalmar::index.3"** %9
  %12 = getelementptr inbounds %"class.Kalmar::index.3"* %11, i32 0, i32 0
  %13 = load i32** %10, align 8
  store %"struct.Kalmar::index_impl.4"* %12, %"struct.Kalmar::index_impl.4"** %7, align 8
  store i32* %13, i32** %8, align 8
  %14 = load %"struct.Kalmar::index_impl.4"** %7
  %15 = bitcast %"struct.Kalmar::index_impl.4"* %14 to %"class.Kalmar::__index_leaf"*
  %16 = load i32** %8, align 8
  %17 = load i32* %16, align 4
  store %"class.Kalmar::__index_leaf"* %15, %"class.Kalmar::__index_leaf"** %5, align 8
  store i32 %17, i32* %6, align 4
  %18 = load %"class.Kalmar::__index_leaf"** %5
  %19 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %18, i32 0, i32 0
  %20 = load i32* %6, align 4
  store i32 %20, i32* %19, align 4
  %21 = bitcast %"struct.Kalmar::index_impl.4"* %14 to i8*
  %22 = getelementptr inbounds i8* %21, i64 8
  %23 = bitcast i8* %22 to %"class.Kalmar::__index_leaf.2"*
  %24 = load i32** %8, align 8
  %25 = getelementptr inbounds i32* %24, i64 1
  %26 = load i32* %25, align 4
  store %"class.Kalmar::__index_leaf.2"* %23, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %26, i32* %2, align 4
  %27 = load %"class.Kalmar::__index_leaf.2"** %1
  %28 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %27, i32 0, i32 0
  %29 = load i32* %2, align 4
  store i32 %29, i32* %28, align 4
  %30 = bitcast %"struct.Kalmar::index_impl.4"* %14 to i8*
  %31 = getelementptr inbounds i8* %30, i64 16
  %32 = bitcast i8* %31 to %"class.Kalmar::__index_leaf.5"*
  %33 = load i32** %8, align 8
  %34 = getelementptr inbounds i32* %33, i64 2
  %35 = load i32* %34, align 4
  store %"class.Kalmar::__index_leaf.5"* %32, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %35, i32* %4, align 4
  %36 = load %"class.Kalmar::__index_leaf.5"** %3
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %36, i32 0, i32 0
  %38 = load i32* %4, align 4
  store i32 %38, i32* %37, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1EPKi(%"class.Kalmar::index.3"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca i32*, align 8
  %9 = alloca %"class.Kalmar::index.3"*, align 8
  %10 = alloca i32*, align 8
  %11 = alloca %"class.Kalmar::index.3"*, align 8
  %12 = alloca i32*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %11, align 8
  store i32* %components, i32** %12, align 8
  %13 = load %"class.Kalmar::index.3"** %11
  %14 = load i32** %12, align 8
  store %"class.Kalmar::index.3"* %13, %"class.Kalmar::index.3"** %9, align 8
  store i32* %14, i32** %10, align 8
  %15 = load %"class.Kalmar::index.3"** %9
  %16 = getelementptr inbounds %"class.Kalmar::index.3"* %15, i32 0, i32 0
  %17 = load i32** %10, align 8
  store %"struct.Kalmar::index_impl.4"* %16, %"struct.Kalmar::index_impl.4"** %7, align 8
  store i32* %17, i32** %8, align 8
  %18 = load %"struct.Kalmar::index_impl.4"** %7
  %19 = bitcast %"struct.Kalmar::index_impl.4"* %18 to %"class.Kalmar::__index_leaf"*
  %20 = load i32** %8, align 8
  %21 = load i32* %20, align 4
  store %"class.Kalmar::__index_leaf"* %19, %"class.Kalmar::__index_leaf"** %5, align 8
  store i32 %21, i32* %6, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %5
  %23 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  %24 = load i32* %6, align 4
  store i32 %24, i32* %23, align 4
  %25 = bitcast %"struct.Kalmar::index_impl.4"* %18 to i8*
  %26 = getelementptr inbounds i8* %25, i64 8
  %27 = bitcast i8* %26 to %"class.Kalmar::__index_leaf.2"*
  %28 = load i32** %8, align 8
  %29 = getelementptr inbounds i32* %28, i64 1
  %30 = load i32* %29, align 4
  store %"class.Kalmar::__index_leaf.2"* %27, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %30, i32* %2, align 4
  %31 = load %"class.Kalmar::__index_leaf.2"** %1
  %32 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %31, i32 0, i32 0
  %33 = load i32* %2, align 4
  store i32 %33, i32* %32, align 4
  %34 = bitcast %"struct.Kalmar::index_impl.4"* %18 to i8*
  %35 = getelementptr inbounds i8* %34, i64 16
  %36 = bitcast i8* %35 to %"class.Kalmar::__index_leaf.5"*
  %37 = load i32** %8, align 8
  %38 = getelementptr inbounds i32* %37, i64 2
  %39 = load i32* %38, align 4
  store %"class.Kalmar::__index_leaf.5"* %36, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %39, i32* %4, align 4
  %40 = load %"class.Kalmar::__index_leaf.5"** %3
  %41 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %40, i32 0, i32 0
  %42 = load i32* %4, align 4
  store i32 %42, i32* %41, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC2EPi(%"class.Kalmar::index.3"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca i32*, align 8
  %9 = alloca %"class.Kalmar::index.3"*, align 8
  %10 = alloca i32*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %9, align 8
  store i32* %components, i32** %10, align 8
  %11 = load %"class.Kalmar::index.3"** %9
  %12 = getelementptr inbounds %"class.Kalmar::index.3"* %11, i32 0, i32 0
  %13 = load i32** %10, align 8
  store %"struct.Kalmar::index_impl.4"* %12, %"struct.Kalmar::index_impl.4"** %7, align 8
  store i32* %13, i32** %8, align 8
  %14 = load %"struct.Kalmar::index_impl.4"** %7
  %15 = bitcast %"struct.Kalmar::index_impl.4"* %14 to %"class.Kalmar::__index_leaf"*
  %16 = load i32** %8, align 8
  %17 = load i32* %16, align 4
  store %"class.Kalmar::__index_leaf"* %15, %"class.Kalmar::__index_leaf"** %5, align 8
  store i32 %17, i32* %6, align 4
  %18 = load %"class.Kalmar::__index_leaf"** %5
  %19 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %18, i32 0, i32 0
  %20 = load i32* %6, align 4
  store i32 %20, i32* %19, align 4
  %21 = bitcast %"struct.Kalmar::index_impl.4"* %14 to i8*
  %22 = getelementptr inbounds i8* %21, i64 8
  %23 = bitcast i8* %22 to %"class.Kalmar::__index_leaf.2"*
  %24 = load i32** %8, align 8
  %25 = getelementptr inbounds i32* %24, i64 1
  %26 = load i32* %25, align 4
  store %"class.Kalmar::__index_leaf.2"* %23, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %26, i32* %2, align 4
  %27 = load %"class.Kalmar::__index_leaf.2"** %1
  %28 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %27, i32 0, i32 0
  %29 = load i32* %2, align 4
  store i32 %29, i32* %28, align 4
  %30 = bitcast %"struct.Kalmar::index_impl.4"* %14 to i8*
  %31 = getelementptr inbounds i8* %30, i64 16
  %32 = bitcast i8* %31 to %"class.Kalmar::__index_leaf.5"*
  %33 = load i32** %8, align 8
  %34 = getelementptr inbounds i32* %33, i64 2
  %35 = load i32* %34, align 4
  store %"class.Kalmar::__index_leaf.5"* %32, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %35, i32* %4, align 4
  %36 = load %"class.Kalmar::__index_leaf.5"** %3
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %36, i32 0, i32 0
  %38 = load i32* %4, align 4
  store i32 %38, i32* %37, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEC1EPi(%"class.Kalmar::index.3"* %this, i32* %components) unnamed_addr #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca i32*, align 8
  %9 = alloca %"class.Kalmar::index.3"*, align 8
  %10 = alloca i32*, align 8
  %11 = alloca %"class.Kalmar::index.3"*, align 8
  %12 = alloca i32*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %11, align 8
  store i32* %components, i32** %12, align 8
  %13 = load %"class.Kalmar::index.3"** %11
  %14 = load i32** %12, align 8
  store %"class.Kalmar::index.3"* %13, %"class.Kalmar::index.3"** %9, align 8
  store i32* %14, i32** %10, align 8
  %15 = load %"class.Kalmar::index.3"** %9
  %16 = getelementptr inbounds %"class.Kalmar::index.3"* %15, i32 0, i32 0
  %17 = load i32** %10, align 8
  store %"struct.Kalmar::index_impl.4"* %16, %"struct.Kalmar::index_impl.4"** %7, align 8
  store i32* %17, i32** %8, align 8
  %18 = load %"struct.Kalmar::index_impl.4"** %7
  %19 = bitcast %"struct.Kalmar::index_impl.4"* %18 to %"class.Kalmar::__index_leaf"*
  %20 = load i32** %8, align 8
  %21 = load i32* %20, align 4
  store %"class.Kalmar::__index_leaf"* %19, %"class.Kalmar::__index_leaf"** %5, align 8
  store i32 %21, i32* %6, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %5
  %23 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  %24 = load i32* %6, align 4
  store i32 %24, i32* %23, align 4
  %25 = bitcast %"struct.Kalmar::index_impl.4"* %18 to i8*
  %26 = getelementptr inbounds i8* %25, i64 8
  %27 = bitcast i8* %26 to %"class.Kalmar::__index_leaf.2"*
  %28 = load i32** %8, align 8
  %29 = getelementptr inbounds i32* %28, i64 1
  %30 = load i32* %29, align 4
  store %"class.Kalmar::__index_leaf.2"* %27, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %30, i32* %2, align 4
  %31 = load %"class.Kalmar::__index_leaf.2"** %1
  %32 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %31, i32 0, i32 0
  %33 = load i32* %2, align 4
  store i32 %33, i32* %32, align 4
  %34 = bitcast %"struct.Kalmar::index_impl.4"* %18 to i8*
  %35 = getelementptr inbounds i8* %34, i64 16
  %36 = bitcast i8* %35 to %"class.Kalmar::__index_leaf.5"*
  %37 = load i32** %8, align 8
  %38 = getelementptr inbounds i32* %37, i64 2
  %39 = load i32* %38, align 4
  store %"class.Kalmar::__index_leaf.5"* %36, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %39, i32* %4, align 4
  %40 = load %"class.Kalmar::__index_leaf.5"** %3
  %41 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %40, i32 0, i32 0
  %42 = load i32* %4, align 4
  store i32 %42, i32* %41, align 4
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEaSERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %other) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::__index_leaf", align 8
  %10 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %11 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %16 = alloca %"class.Kalmar::__index_leaf", align 4
  %17 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %18 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %19 = alloca %"class.Kalmar::index.3"*, align 8
  %20 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %19, align 8
  store %"class.Kalmar::index.3"* %other, %"class.Kalmar::index.3"** %20, align 8
  %21 = load %"class.Kalmar::index.3"** %19
  %22 = getelementptr inbounds %"class.Kalmar::index.3"* %21, i32 0, i32 0
  %23 = load %"class.Kalmar::index.3"** %20, align 8
  %24 = getelementptr inbounds %"class.Kalmar::index.3"* %23, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %22, %"struct.Kalmar::index_impl.4"** %14, align 8
  store %"struct.Kalmar::index_impl.4"* %24, %"struct.Kalmar::index_impl.4"** %15, align 8
  %25 = load %"struct.Kalmar::index_impl.4"** %14
  %26 = bitcast %"struct.Kalmar::index_impl.4"* %25 to %"class.Kalmar::__index_leaf"*
  %27 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %28 = bitcast %"struct.Kalmar::index_impl.4"* %27 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %28, %"class.Kalmar::__index_leaf"** %13, align 8
  %29 = load %"class.Kalmar::__index_leaf"** %13
  %30 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %29, i32 0, i32 0
  %31 = load i32* %30
  store %"class.Kalmar::__index_leaf"* %26, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %31, i32* %2, align 4
  %32 = load %"class.Kalmar::__index_leaf"** %1
  %33 = load i32* %2, align 4
  %34 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %32, i32 0, i32 0
  store i32 %33, i32* %34, align 4
  %35 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  %36 = bitcast %"class.Kalmar::__index_leaf"* %32 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %35, i8* %36, i64 8, i32 4, i1 false)
  %37 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %38 = getelementptr inbounds i8* %37, i64 8
  %39 = bitcast i8* %38 to %"class.Kalmar::__index_leaf.2"*
  %40 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %41 = bitcast %"struct.Kalmar::index_impl.4"* %40 to i8*
  %42 = getelementptr inbounds i8* %41, i64 8
  %43 = bitcast i8* %42 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %43, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %44 = load %"class.Kalmar::__index_leaf.2"** %3
  %45 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %44, i32 0, i32 0
  %46 = load i32* %45
  store %"class.Kalmar::__index_leaf.2"* %39, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %46, i32* %5, align 4
  %47 = load %"class.Kalmar::__index_leaf.2"** %4
  %48 = load i32* %5, align 4
  %49 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %47, i32 0, i32 0
  store i32 %48, i32* %49, align 4
  %50 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i8*
  %51 = bitcast %"class.Kalmar::__index_leaf.2"* %47 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %50, i8* %51, i64 8, i32 4, i1 false)
  %52 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %53 = getelementptr inbounds i8* %52, i64 16
  %54 = bitcast i8* %53 to %"class.Kalmar::__index_leaf.5"*
  %55 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %56 = bitcast %"struct.Kalmar::index_impl.4"* %55 to i8*
  %57 = getelementptr inbounds i8* %56, i64 16
  %58 = bitcast i8* %57 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %58, %"class.Kalmar::__index_leaf.5"** %6, align 8
  %59 = load %"class.Kalmar::__index_leaf.5"** %6
  %60 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %59, i32 0, i32 0
  %61 = load i32* %60
  store %"class.Kalmar::__index_leaf.5"* %54, %"class.Kalmar::__index_leaf.5"** %7, align 8
  store i32 %61, i32* %8, align 4
  %62 = load %"class.Kalmar::__index_leaf.5"** %7
  %63 = load i32* %8, align 4
  %64 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %62, i32 0, i32 0
  store i32 %63, i32* %64, align 4
  %65 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i8*
  %66 = bitcast %"class.Kalmar::__index_leaf.5"* %62 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %65, i8* %66, i64 8, i32 4, i1 false)
  %67 = bitcast %"class.Kalmar::__index_leaf"* %16 to i64*
  %68 = load i64* %67, align 1
  %69 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i64*
  %70 = load i64* %69, align 1
  %71 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i64*
  %72 = load i64* %71, align 1
  %73 = bitcast %"class.Kalmar::__index_leaf"* %9 to i64*
  store i64 %68, i64* %73, align 1
  %74 = bitcast %"class.Kalmar::__index_leaf.2"* %10 to i64*
  store i64 %70, i64* %74, align 1
  %75 = bitcast %"class.Kalmar::__index_leaf.5"* %11 to i64*
  store i64 %72, i64* %75, align 1
  store %"struct.Kalmar::index_impl.4"* %25, %"struct.Kalmar::index_impl.4"** %12, align 8
  %76 = load %"struct.Kalmar::index_impl.4"** %12
  ret %"class.Kalmar::index.3"* %21
}

; Function Attrs: alwaysinline uwtable
define weak_odr i32 @_ZNK6Kalmar5indexILi3EEixEj(%"class.Kalmar::index.3"* %this, i32 %c) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index.3"*, align 8
  %5 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %4, align 8
  store i32 %c, i32* %5, align 4
  %6 = load %"class.Kalmar::index.3"** %4
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %6, i32 0, i32 0
  %8 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl.4"* %7, %"struct.Kalmar::index_impl.4"** %2, align 8
  store i32 %8, i32* %3, align 4
  %9 = load %"struct.Kalmar::index_impl.4"** %2
  %10 = bitcast %"struct.Kalmar::index_impl.4"* %9 to %"class.Kalmar::__index_leaf"*
  %11 = load i32* %3, align 4
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %10, i64 %12
  store %"class.Kalmar::__index_leaf"* %13, %"class.Kalmar::__index_leaf"** %1, align 8
  %14 = load %"class.Kalmar::__index_leaf"** %1
  %15 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  %16 = load i32* %15
  ret i32 %16
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(4) i32* @_ZN6Kalmar5indexILi3EEixEj(%"class.Kalmar::index.3"* %this, i32 %c) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index.3"*, align 8
  %5 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %4, align 8
  store i32 %c, i32* %5, align 4
  %6 = load %"class.Kalmar::index.3"** %4
  %7 = getelementptr inbounds %"class.Kalmar::index.3"* %6, i32 0, i32 0
  %8 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl.4"* %7, %"struct.Kalmar::index_impl.4"** %2, align 8
  store i32 %8, i32* %3, align 4
  %9 = load %"struct.Kalmar::index_impl.4"** %2
  %10 = bitcast %"struct.Kalmar::index_impl.4"* %9 to %"class.Kalmar::__index_leaf"*
  %11 = load i32* %3, align 4
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %10, i64 %12
  store %"class.Kalmar::__index_leaf"* %13, %"class.Kalmar::__index_leaf"** %1, align 8
  %14 = load %"class.Kalmar::__index_leaf"** %1
  %15 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %14, i32 0, i32 0
  ret i32* %15
}

; Function Attrs: alwaysinline uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi3EEeqERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %other) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index.3"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::index.3"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca i32, align 4
  %14 = alloca %"class.Kalmar::index.3"*, align 8
  %15 = alloca i32, align 4
  %16 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %17 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %18 = alloca i32, align 4
  %19 = alloca %"class.Kalmar::index.3"*, align 8
  %20 = alloca i32, align 4
  %21 = alloca %"class.Kalmar::index.3"*, align 8
  %22 = alloca %"class.Kalmar::index.3"*, align 8
  %23 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %24 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %25 = alloca i32, align 4
  %26 = alloca %"class.Kalmar::index.3"*, align 8
  %27 = alloca i32, align 4
  %28 = alloca %"class.Kalmar::index.3"*, align 8
  %29 = alloca %"class.Kalmar::index.3"*, align 8
  %30 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %31 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %32 = alloca i32, align 4
  %33 = alloca %"class.Kalmar::index.3"*, align 8
  %34 = alloca i32, align 4
  %35 = alloca %"class.Kalmar::index.3"*, align 8
  %36 = alloca %"class.Kalmar::index.3"*, align 8
  %37 = alloca %"class.Kalmar::index.3"*, align 8
  %38 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %37, align 8
  store %"class.Kalmar::index.3"* %other, %"class.Kalmar::index.3"** %38, align 8
  %39 = load %"class.Kalmar::index.3"** %37
  %40 = load %"class.Kalmar::index.3"** %38, align 8
  store %"class.Kalmar::index.3"* %39, %"class.Kalmar::index.3"** %35, align 8
  store %"class.Kalmar::index.3"* %40, %"class.Kalmar::index.3"** %36, align 8
  %41 = load %"class.Kalmar::index.3"** %35, align 8
  store %"class.Kalmar::index.3"* %41, %"class.Kalmar::index.3"** %33, align 8
  store i32 2, i32* %34, align 4
  %42 = load %"class.Kalmar::index.3"** %33
  %43 = getelementptr inbounds %"class.Kalmar::index.3"* %42, i32 0, i32 0
  %44 = load i32* %34, align 4
  store %"struct.Kalmar::index_impl.4"* %43, %"struct.Kalmar::index_impl.4"** %31, align 8
  store i32 %44, i32* %32, align 4
  %45 = load %"struct.Kalmar::index_impl.4"** %31
  %46 = bitcast %"struct.Kalmar::index_impl.4"* %45 to %"class.Kalmar::__index_leaf"*
  %47 = load i32* %32, align 4
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %46, i64 %48
  store %"class.Kalmar::__index_leaf"* %49, %"class.Kalmar::__index_leaf"** %30, align 8
  %50 = load %"class.Kalmar::__index_leaf"** %30
  %51 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %50, i32 0, i32 0
  %52 = load i32* %51
  %53 = load %"class.Kalmar::index.3"** %36, align 8
  store %"class.Kalmar::index.3"* %53, %"class.Kalmar::index.3"** %4, align 8
  store i32 2, i32* %5, align 4
  %54 = load %"class.Kalmar::index.3"** %4
  %55 = getelementptr inbounds %"class.Kalmar::index.3"* %54, i32 0, i32 0
  %56 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl.4"* %55, %"struct.Kalmar::index_impl.4"** %2, align 8
  store i32 %56, i32* %3, align 4
  %57 = load %"struct.Kalmar::index_impl.4"** %2
  %58 = bitcast %"struct.Kalmar::index_impl.4"* %57 to %"class.Kalmar::__index_leaf"*
  %59 = load i32* %3, align 4
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %58, i64 %60
  store %"class.Kalmar::__index_leaf"* %61, %"class.Kalmar::__index_leaf"** %1, align 8
  %62 = load %"class.Kalmar::__index_leaf"** %1
  %63 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %62, i32 0, i32 0
  %64 = load i32* %63
  %65 = icmp eq i32 %52, %64
  br i1 %65, label %66, label %_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_.exit

; <label>:66                                      ; preds = %0
  %67 = load %"class.Kalmar::index.3"** %35, align 8
  %68 = load %"class.Kalmar::index.3"** %36, align 8
  store %"class.Kalmar::index.3"* %67, %"class.Kalmar::index.3"** %28, align 8
  store %"class.Kalmar::index.3"* %68, %"class.Kalmar::index.3"** %29, align 8
  %69 = load %"class.Kalmar::index.3"** %28, align 8
  store %"class.Kalmar::index.3"* %69, %"class.Kalmar::index.3"** %26, align 8
  store i32 1, i32* %27, align 4
  %70 = load %"class.Kalmar::index.3"** %26
  %71 = getelementptr inbounds %"class.Kalmar::index.3"* %70, i32 0, i32 0
  %72 = load i32* %27, align 4
  store %"struct.Kalmar::index_impl.4"* %71, %"struct.Kalmar::index_impl.4"** %24, align 8
  store i32 %72, i32* %25, align 4
  %73 = load %"struct.Kalmar::index_impl.4"** %24
  %74 = bitcast %"struct.Kalmar::index_impl.4"* %73 to %"class.Kalmar::__index_leaf"*
  %75 = load i32* %25, align 4
  %76 = zext i32 %75 to i64
  %77 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %74, i64 %76
  store %"class.Kalmar::__index_leaf"* %77, %"class.Kalmar::__index_leaf"** %23, align 8
  %78 = load %"class.Kalmar::__index_leaf"** %23
  %79 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %78, i32 0, i32 0
  %80 = load i32* %79
  %81 = load %"class.Kalmar::index.3"** %29, align 8
  store %"class.Kalmar::index.3"* %81, %"class.Kalmar::index.3"** %9, align 8
  store i32 1, i32* %10, align 4
  %82 = load %"class.Kalmar::index.3"** %9
  %83 = getelementptr inbounds %"class.Kalmar::index.3"* %82, i32 0, i32 0
  %84 = load i32* %10, align 4
  store %"struct.Kalmar::index_impl.4"* %83, %"struct.Kalmar::index_impl.4"** %7, align 8
  store i32 %84, i32* %8, align 4
  %85 = load %"struct.Kalmar::index_impl.4"** %7
  %86 = bitcast %"struct.Kalmar::index_impl.4"* %85 to %"class.Kalmar::__index_leaf"*
  %87 = load i32* %8, align 4
  %88 = zext i32 %87 to i64
  %89 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %86, i64 %88
  store %"class.Kalmar::__index_leaf"* %89, %"class.Kalmar::__index_leaf"** %6, align 8
  %90 = load %"class.Kalmar::__index_leaf"** %6
  %91 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %90, i32 0, i32 0
  %92 = load i32* %91
  %93 = icmp eq i32 %80, %92
  br i1 %93, label %94, label %_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i

; <label>:94                                      ; preds = %66
  %95 = load %"class.Kalmar::index.3"** %28, align 8
  %96 = load %"class.Kalmar::index.3"** %29, align 8
  store %"class.Kalmar::index.3"* %95, %"class.Kalmar::index.3"** %21, align 8
  store %"class.Kalmar::index.3"* %96, %"class.Kalmar::index.3"** %22, align 8
  %97 = load %"class.Kalmar::index.3"** %21, align 8
  store %"class.Kalmar::index.3"* %97, %"class.Kalmar::index.3"** %19, align 8
  store i32 0, i32* %20, align 4
  %98 = load %"class.Kalmar::index.3"** %19
  %99 = getelementptr inbounds %"class.Kalmar::index.3"* %98, i32 0, i32 0
  %100 = load i32* %20, align 4
  store %"struct.Kalmar::index_impl.4"* %99, %"struct.Kalmar::index_impl.4"** %17, align 8
  store i32 %100, i32* %18, align 4
  %101 = load %"struct.Kalmar::index_impl.4"** %17
  %102 = bitcast %"struct.Kalmar::index_impl.4"* %101 to %"class.Kalmar::__index_leaf"*
  %103 = load i32* %18, align 4
  %104 = zext i32 %103 to i64
  %105 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %102, i64 %104
  store %"class.Kalmar::__index_leaf"* %105, %"class.Kalmar::__index_leaf"** %16, align 8
  %106 = load %"class.Kalmar::__index_leaf"** %16
  %107 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %106, i32 0, i32 0
  %108 = load i32* %107
  %109 = load %"class.Kalmar::index.3"** %22, align 8
  store %"class.Kalmar::index.3"* %109, %"class.Kalmar::index.3"** %14, align 8
  store i32 0, i32* %15, align 4
  %110 = load %"class.Kalmar::index.3"** %14
  %111 = getelementptr inbounds %"class.Kalmar::index.3"* %110, i32 0, i32 0
  %112 = load i32* %15, align 4
  store %"struct.Kalmar::index_impl.4"* %111, %"struct.Kalmar::index_impl.4"** %12, align 8
  store i32 %112, i32* %13, align 4
  %113 = load %"struct.Kalmar::index_impl.4"** %12
  %114 = bitcast %"struct.Kalmar::index_impl.4"* %113 to %"class.Kalmar::__index_leaf"*
  %115 = load i32* %13, align 4
  %116 = zext i32 %115 to i64
  %117 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %114, i64 %116
  store %"class.Kalmar::__index_leaf"* %117, %"class.Kalmar::__index_leaf"** %11, align 8
  %118 = load %"class.Kalmar::__index_leaf"** %11
  %119 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %118, i32 0, i32 0
  %120 = load i32* %119
  %121 = icmp eq i32 %108, %120
  br label %_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i

_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i: ; preds = %94, %66
  %122 = phi i1 [ false, %66 ], [ %121, %94 ]
  br label %_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_.exit

_ZN6Kalmar12index_helperILi3ENS_5indexILi3EEEE5equalERKS2_S5_.exit: ; preds = %_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i, %0
  %123 = phi i1 [ false, %0 ], [ %122, %_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i ]
  ret i1 %123
}

; Function Attrs: alwaysinline uwtable
define weak_odr zeroext i1 @_ZNK6Kalmar5indexILi3EEneERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %other) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::index.3"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %7 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::index.3"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca i32, align 4
  %14 = alloca %"class.Kalmar::index.3"*, align 8
  %15 = alloca i32, align 4
  %16 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %17 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %18 = alloca i32, align 4
  %19 = alloca %"class.Kalmar::index.3"*, align 8
  %20 = alloca i32, align 4
  %21 = alloca %"class.Kalmar::index.3"*, align 8
  %22 = alloca %"class.Kalmar::index.3"*, align 8
  %23 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %24 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %25 = alloca i32, align 4
  %26 = alloca %"class.Kalmar::index.3"*, align 8
  %27 = alloca i32, align 4
  %28 = alloca %"class.Kalmar::index.3"*, align 8
  %29 = alloca %"class.Kalmar::index.3"*, align 8
  %30 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %31 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %32 = alloca i32, align 4
  %33 = alloca %"class.Kalmar::index.3"*, align 8
  %34 = alloca i32, align 4
  %35 = alloca %"class.Kalmar::index.3"*, align 8
  %36 = alloca %"class.Kalmar::index.3"*, align 8
  %37 = alloca %"class.Kalmar::index.3"*, align 8
  %38 = alloca %"class.Kalmar::index.3"*, align 8
  %39 = alloca %"class.Kalmar::index.3"*, align 8
  %40 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %39, align 8
  store %"class.Kalmar::index.3"* %other, %"class.Kalmar::index.3"** %40, align 8
  %41 = load %"class.Kalmar::index.3"** %39
  %42 = load %"class.Kalmar::index.3"** %40, align 8
  store %"class.Kalmar::index.3"* %41, %"class.Kalmar::index.3"** %37, align 8
  store %"class.Kalmar::index.3"* %42, %"class.Kalmar::index.3"** %38, align 8
  %43 = load %"class.Kalmar::index.3"** %37
  %44 = load %"class.Kalmar::index.3"** %38, align 8
  store %"class.Kalmar::index.3"* %43, %"class.Kalmar::index.3"** %35, align 8
  store %"class.Kalmar::index.3"* %44, %"class.Kalmar::index.3"** %36, align 8
  %45 = load %"class.Kalmar::index.3"** %35, align 8
  store %"class.Kalmar::index.3"* %45, %"class.Kalmar::index.3"** %33, align 8
  store i32 2, i32* %34, align 4
  %46 = load %"class.Kalmar::index.3"** %33
  %47 = getelementptr inbounds %"class.Kalmar::index.3"* %46, i32 0, i32 0
  %48 = load i32* %34, align 4
  store %"struct.Kalmar::index_impl.4"* %47, %"struct.Kalmar::index_impl.4"** %31, align 8
  store i32 %48, i32* %32, align 4
  %49 = load %"struct.Kalmar::index_impl.4"** %31
  %50 = bitcast %"struct.Kalmar::index_impl.4"* %49 to %"class.Kalmar::__index_leaf"*
  %51 = load i32* %32, align 4
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %50, i64 %52
  store %"class.Kalmar::__index_leaf"* %53, %"class.Kalmar::__index_leaf"** %30, align 8
  %54 = load %"class.Kalmar::__index_leaf"** %30
  %55 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %54, i32 0, i32 0
  %56 = load i32* %55
  %57 = load %"class.Kalmar::index.3"** %36, align 8
  store %"class.Kalmar::index.3"* %57, %"class.Kalmar::index.3"** %4, align 8
  store i32 2, i32* %5, align 4
  %58 = load %"class.Kalmar::index.3"** %4
  %59 = getelementptr inbounds %"class.Kalmar::index.3"* %58, i32 0, i32 0
  %60 = load i32* %5, align 4
  store %"struct.Kalmar::index_impl.4"* %59, %"struct.Kalmar::index_impl.4"** %2, align 8
  store i32 %60, i32* %3, align 4
  %61 = load %"struct.Kalmar::index_impl.4"** %2
  %62 = bitcast %"struct.Kalmar::index_impl.4"* %61 to %"class.Kalmar::__index_leaf"*
  %63 = load i32* %3, align 4
  %64 = zext i32 %63 to i64
  %65 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %62, i64 %64
  store %"class.Kalmar::__index_leaf"* %65, %"class.Kalmar::__index_leaf"** %1, align 8
  %66 = load %"class.Kalmar::__index_leaf"** %1
  %67 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %66, i32 0, i32 0
  %68 = load i32* %67
  %69 = icmp eq i32 %56, %68
  br i1 %69, label %70, label %_ZNK6Kalmar5indexILi3EEeqERKS1_.exit

; <label>:70                                      ; preds = %0
  %71 = load %"class.Kalmar::index.3"** %35, align 8
  %72 = load %"class.Kalmar::index.3"** %36, align 8
  store %"class.Kalmar::index.3"* %71, %"class.Kalmar::index.3"** %28, align 8
  store %"class.Kalmar::index.3"* %72, %"class.Kalmar::index.3"** %29, align 8
  %73 = load %"class.Kalmar::index.3"** %28, align 8
  store %"class.Kalmar::index.3"* %73, %"class.Kalmar::index.3"** %26, align 8
  store i32 1, i32* %27, align 4
  %74 = load %"class.Kalmar::index.3"** %26
  %75 = getelementptr inbounds %"class.Kalmar::index.3"* %74, i32 0, i32 0
  %76 = load i32* %27, align 4
  store %"struct.Kalmar::index_impl.4"* %75, %"struct.Kalmar::index_impl.4"** %24, align 8
  store i32 %76, i32* %25, align 4
  %77 = load %"struct.Kalmar::index_impl.4"** %24
  %78 = bitcast %"struct.Kalmar::index_impl.4"* %77 to %"class.Kalmar::__index_leaf"*
  %79 = load i32* %25, align 4
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %78, i64 %80
  store %"class.Kalmar::__index_leaf"* %81, %"class.Kalmar::__index_leaf"** %23, align 8
  %82 = load %"class.Kalmar::__index_leaf"** %23
  %83 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %82, i32 0, i32 0
  %84 = load i32* %83
  %85 = load %"class.Kalmar::index.3"** %29, align 8
  store %"class.Kalmar::index.3"* %85, %"class.Kalmar::index.3"** %9, align 8
  store i32 1, i32* %10, align 4
  %86 = load %"class.Kalmar::index.3"** %9
  %87 = getelementptr inbounds %"class.Kalmar::index.3"* %86, i32 0, i32 0
  %88 = load i32* %10, align 4
  store %"struct.Kalmar::index_impl.4"* %87, %"struct.Kalmar::index_impl.4"** %7, align 8
  store i32 %88, i32* %8, align 4
  %89 = load %"struct.Kalmar::index_impl.4"** %7
  %90 = bitcast %"struct.Kalmar::index_impl.4"* %89 to %"class.Kalmar::__index_leaf"*
  %91 = load i32* %8, align 4
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %90, i64 %92
  store %"class.Kalmar::__index_leaf"* %93, %"class.Kalmar::__index_leaf"** %6, align 8
  %94 = load %"class.Kalmar::__index_leaf"** %6
  %95 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %94, i32 0, i32 0
  %96 = load i32* %95
  %97 = icmp eq i32 %84, %96
  br i1 %97, label %98, label %_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i.i

; <label>:98                                      ; preds = %70
  %99 = load %"class.Kalmar::index.3"** %28, align 8
  %100 = load %"class.Kalmar::index.3"** %29, align 8
  store %"class.Kalmar::index.3"* %99, %"class.Kalmar::index.3"** %21, align 8
  store %"class.Kalmar::index.3"* %100, %"class.Kalmar::index.3"** %22, align 8
  %101 = load %"class.Kalmar::index.3"** %21, align 8
  store %"class.Kalmar::index.3"* %101, %"class.Kalmar::index.3"** %19, align 8
  store i32 0, i32* %20, align 4
  %102 = load %"class.Kalmar::index.3"** %19
  %103 = getelementptr inbounds %"class.Kalmar::index.3"* %102, i32 0, i32 0
  %104 = load i32* %20, align 4
  store %"struct.Kalmar::index_impl.4"* %103, %"struct.Kalmar::index_impl.4"** %17, align 8
  store i32 %104, i32* %18, align 4
  %105 = load %"struct.Kalmar::index_impl.4"** %17
  %106 = bitcast %"struct.Kalmar::index_impl.4"* %105 to %"class.Kalmar::__index_leaf"*
  %107 = load i32* %18, align 4
  %108 = zext i32 %107 to i64
  %109 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %106, i64 %108
  store %"class.Kalmar::__index_leaf"* %109, %"class.Kalmar::__index_leaf"** %16, align 8
  %110 = load %"class.Kalmar::__index_leaf"** %16
  %111 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %110, i32 0, i32 0
  %112 = load i32* %111
  %113 = load %"class.Kalmar::index.3"** %22, align 8
  store %"class.Kalmar::index.3"* %113, %"class.Kalmar::index.3"** %14, align 8
  store i32 0, i32* %15, align 4
  %114 = load %"class.Kalmar::index.3"** %14
  %115 = getelementptr inbounds %"class.Kalmar::index.3"* %114, i32 0, i32 0
  %116 = load i32* %15, align 4
  store %"struct.Kalmar::index_impl.4"* %115, %"struct.Kalmar::index_impl.4"** %12, align 8
  store i32 %116, i32* %13, align 4
  %117 = load %"struct.Kalmar::index_impl.4"** %12
  %118 = bitcast %"struct.Kalmar::index_impl.4"* %117 to %"class.Kalmar::__index_leaf"*
  %119 = load i32* %13, align 4
  %120 = zext i32 %119 to i64
  %121 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %118, i64 %120
  store %"class.Kalmar::__index_leaf"* %121, %"class.Kalmar::__index_leaf"** %11, align 8
  %122 = load %"class.Kalmar::__index_leaf"** %11
  %123 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %122, i32 0, i32 0
  %124 = load i32* %123
  %125 = icmp eq i32 %112, %124
  br label %_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i.i

_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i.i: ; preds = %98, %70
  %126 = phi i1 [ false, %70 ], [ %125, %98 ]
  br label %_ZNK6Kalmar5indexILi3EEeqERKS1_.exit

_ZNK6Kalmar5indexILi3EEeqERKS1_.exit:             ; preds = %_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i.i, %0
  %127 = phi i1 [ false, %0 ], [ %126, %_ZN6Kalmar12index_helperILi2ENS_5indexILi3EEEE5equalERKS2_S5_.exit.i.i ]
  %128 = xor i1 %127, true
  ret i1 %128
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEpLERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %rhs) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::__index_leaf", align 8
  %10 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %11 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %16 = alloca %"class.Kalmar::__index_leaf", align 4
  %17 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %18 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %19 = alloca %"class.Kalmar::index.3"*, align 8
  %20 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %19, align 8
  store %"class.Kalmar::index.3"* %rhs, %"class.Kalmar::index.3"** %20, align 8
  %21 = load %"class.Kalmar::index.3"** %19
  %22 = getelementptr inbounds %"class.Kalmar::index.3"* %21, i32 0, i32 0
  %23 = load %"class.Kalmar::index.3"** %20, align 8
  %24 = getelementptr inbounds %"class.Kalmar::index.3"* %23, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %22, %"struct.Kalmar::index_impl.4"** %14, align 8
  store %"struct.Kalmar::index_impl.4"* %24, %"struct.Kalmar::index_impl.4"** %15, align 8
  %25 = load %"struct.Kalmar::index_impl.4"** %14
  %26 = bitcast %"struct.Kalmar::index_impl.4"* %25 to %"class.Kalmar::__index_leaf"*
  %27 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %28 = bitcast %"struct.Kalmar::index_impl.4"* %27 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %28, %"class.Kalmar::__index_leaf"** %13, align 8
  %29 = load %"class.Kalmar::__index_leaf"** %13
  %30 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %29, i32 0, i32 0
  %31 = load i32* %30
  store %"class.Kalmar::__index_leaf"* %26, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %31, i32* %2, align 4
  %32 = load %"class.Kalmar::__index_leaf"** %1
  %33 = load i32* %2, align 4
  %34 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %32, i32 0, i32 0
  %35 = load i32* %34, align 4
  %36 = add nsw i32 %35, %33
  store i32 %36, i32* %34, align 4
  %37 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  %38 = bitcast %"class.Kalmar::__index_leaf"* %32 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %37, i8* %38, i64 8, i32 4, i1 false) #2
  %39 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %40 = getelementptr inbounds i8* %39, i64 8
  %41 = bitcast i8* %40 to %"class.Kalmar::__index_leaf.2"*
  %42 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %43 = bitcast %"struct.Kalmar::index_impl.4"* %42 to i8*
  %44 = getelementptr inbounds i8* %43, i64 8
  %45 = bitcast i8* %44 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %45, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %46 = load %"class.Kalmar::__index_leaf.2"** %3
  %47 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %46, i32 0, i32 0
  %48 = load i32* %47
  store %"class.Kalmar::__index_leaf.2"* %41, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %48, i32* %5, align 4
  %49 = load %"class.Kalmar::__index_leaf.2"** %4
  %50 = load i32* %5, align 4
  %51 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %49, i32 0, i32 0
  %52 = load i32* %51, align 4
  %53 = add nsw i32 %52, %50
  store i32 %53, i32* %51, align 4
  %54 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i8*
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %49 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %54, i8* %55, i64 8, i32 4, i1 false) #2
  %56 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %57 = getelementptr inbounds i8* %56, i64 16
  %58 = bitcast i8* %57 to %"class.Kalmar::__index_leaf.5"*
  %59 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %60 = bitcast %"struct.Kalmar::index_impl.4"* %59 to i8*
  %61 = getelementptr inbounds i8* %60, i64 16
  %62 = bitcast i8* %61 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %62, %"class.Kalmar::__index_leaf.5"** %6, align 8
  %63 = load %"class.Kalmar::__index_leaf.5"** %6
  %64 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %63, i32 0, i32 0
  %65 = load i32* %64
  store %"class.Kalmar::__index_leaf.5"* %58, %"class.Kalmar::__index_leaf.5"** %7, align 8
  store i32 %65, i32* %8, align 4
  %66 = load %"class.Kalmar::__index_leaf.5"** %7
  %67 = load i32* %8, align 4
  %68 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %66, i32 0, i32 0
  %69 = load i32* %68, align 4
  %70 = add nsw i32 %69, %67
  store i32 %70, i32* %68, align 4
  %71 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i8*
  %72 = bitcast %"class.Kalmar::__index_leaf.5"* %66 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %71, i8* %72, i64 8, i32 4, i1 false) #2
  %73 = bitcast %"class.Kalmar::__index_leaf"* %16 to i64*
  %74 = load i64* %73, align 1
  %75 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i64*
  %76 = load i64* %75, align 1
  %77 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i64*
  %78 = load i64* %77, align 1
  %79 = bitcast %"class.Kalmar::__index_leaf"* %9 to i64*
  store i64 %74, i64* %79, align 1
  %80 = bitcast %"class.Kalmar::__index_leaf.2"* %10 to i64*
  store i64 %76, i64* %80, align 1
  %81 = bitcast %"class.Kalmar::__index_leaf.5"* %11 to i64*
  store i64 %78, i64* %81, align 1
  store %"struct.Kalmar::index_impl.4"* %25, %"struct.Kalmar::index_impl.4"** %12, align 8
  %82 = load %"struct.Kalmar::index_impl.4"** %12
  ret %"class.Kalmar::index.3"* %21
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmIERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %rhs) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::__index_leaf", align 8
  %10 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %11 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %16 = alloca %"class.Kalmar::__index_leaf", align 4
  %17 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %18 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %19 = alloca %"class.Kalmar::index.3"*, align 8
  %20 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %19, align 8
  store %"class.Kalmar::index.3"* %rhs, %"class.Kalmar::index.3"** %20, align 8
  %21 = load %"class.Kalmar::index.3"** %19
  %22 = getelementptr inbounds %"class.Kalmar::index.3"* %21, i32 0, i32 0
  %23 = load %"class.Kalmar::index.3"** %20, align 8
  %24 = getelementptr inbounds %"class.Kalmar::index.3"* %23, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %22, %"struct.Kalmar::index_impl.4"** %14, align 8
  store %"struct.Kalmar::index_impl.4"* %24, %"struct.Kalmar::index_impl.4"** %15, align 8
  %25 = load %"struct.Kalmar::index_impl.4"** %14
  %26 = bitcast %"struct.Kalmar::index_impl.4"* %25 to %"class.Kalmar::__index_leaf"*
  %27 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %28 = bitcast %"struct.Kalmar::index_impl.4"* %27 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %28, %"class.Kalmar::__index_leaf"** %13, align 8
  %29 = load %"class.Kalmar::__index_leaf"** %13
  %30 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %29, i32 0, i32 0
  %31 = load i32* %30
  store %"class.Kalmar::__index_leaf"* %26, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %31, i32* %2, align 4
  %32 = load %"class.Kalmar::__index_leaf"** %1
  %33 = load i32* %2, align 4
  %34 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %32, i32 0, i32 0
  %35 = load i32* %34, align 4
  %36 = sub nsw i32 %35, %33
  store i32 %36, i32* %34, align 4
  %37 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  %38 = bitcast %"class.Kalmar::__index_leaf"* %32 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %37, i8* %38, i64 8, i32 4, i1 false) #2
  %39 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %40 = getelementptr inbounds i8* %39, i64 8
  %41 = bitcast i8* %40 to %"class.Kalmar::__index_leaf.2"*
  %42 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %43 = bitcast %"struct.Kalmar::index_impl.4"* %42 to i8*
  %44 = getelementptr inbounds i8* %43, i64 8
  %45 = bitcast i8* %44 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %45, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %46 = load %"class.Kalmar::__index_leaf.2"** %3
  %47 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %46, i32 0, i32 0
  %48 = load i32* %47
  store %"class.Kalmar::__index_leaf.2"* %41, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %48, i32* %5, align 4
  %49 = load %"class.Kalmar::__index_leaf.2"** %4
  %50 = load i32* %5, align 4
  %51 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %49, i32 0, i32 0
  %52 = load i32* %51, align 4
  %53 = sub nsw i32 %52, %50
  store i32 %53, i32* %51, align 4
  %54 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i8*
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %49 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %54, i8* %55, i64 8, i32 4, i1 false) #2
  %56 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %57 = getelementptr inbounds i8* %56, i64 16
  %58 = bitcast i8* %57 to %"class.Kalmar::__index_leaf.5"*
  %59 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %60 = bitcast %"struct.Kalmar::index_impl.4"* %59 to i8*
  %61 = getelementptr inbounds i8* %60, i64 16
  %62 = bitcast i8* %61 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %62, %"class.Kalmar::__index_leaf.5"** %6, align 8
  %63 = load %"class.Kalmar::__index_leaf.5"** %6
  %64 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %63, i32 0, i32 0
  %65 = load i32* %64
  store %"class.Kalmar::__index_leaf.5"* %58, %"class.Kalmar::__index_leaf.5"** %7, align 8
  store i32 %65, i32* %8, align 4
  %66 = load %"class.Kalmar::__index_leaf.5"** %7
  %67 = load i32* %8, align 4
  %68 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %66, i32 0, i32 0
  %69 = load i32* %68, align 4
  %70 = sub nsw i32 %69, %67
  store i32 %70, i32* %68, align 4
  %71 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i8*
  %72 = bitcast %"class.Kalmar::__index_leaf.5"* %66 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %71, i8* %72, i64 8, i32 4, i1 false) #2
  %73 = bitcast %"class.Kalmar::__index_leaf"* %16 to i64*
  %74 = load i64* %73, align 1
  %75 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i64*
  %76 = load i64* %75, align 1
  %77 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i64*
  %78 = load i64* %77, align 1
  %79 = bitcast %"class.Kalmar::__index_leaf"* %9 to i64*
  store i64 %74, i64* %79, align 1
  %80 = bitcast %"class.Kalmar::__index_leaf.2"* %10 to i64*
  store i64 %76, i64* %80, align 1
  %81 = bitcast %"class.Kalmar::__index_leaf.5"* %11 to i64*
  store i64 %78, i64* %81, align 1
  store %"struct.Kalmar::index_impl.4"* %25, %"struct.Kalmar::index_impl.4"** %12, align 8
  %82 = load %"struct.Kalmar::index_impl.4"** %12
  ret %"class.Kalmar::index.3"* %21
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmLERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %__r) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::__index_leaf", align 8
  %10 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %11 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %16 = alloca %"class.Kalmar::__index_leaf", align 4
  %17 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %18 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %19 = alloca %"class.Kalmar::index.3"*, align 8
  %20 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %19, align 8
  store %"class.Kalmar::index.3"* %__r, %"class.Kalmar::index.3"** %20, align 8
  %21 = load %"class.Kalmar::index.3"** %19
  %22 = getelementptr inbounds %"class.Kalmar::index.3"* %21, i32 0, i32 0
  %23 = load %"class.Kalmar::index.3"** %20, align 8
  %24 = getelementptr inbounds %"class.Kalmar::index.3"* %23, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %22, %"struct.Kalmar::index_impl.4"** %14, align 8
  store %"struct.Kalmar::index_impl.4"* %24, %"struct.Kalmar::index_impl.4"** %15, align 8
  %25 = load %"struct.Kalmar::index_impl.4"** %14
  %26 = bitcast %"struct.Kalmar::index_impl.4"* %25 to %"class.Kalmar::__index_leaf"*
  %27 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %28 = bitcast %"struct.Kalmar::index_impl.4"* %27 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %28, %"class.Kalmar::__index_leaf"** %13, align 8
  %29 = load %"class.Kalmar::__index_leaf"** %13
  %30 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %29, i32 0, i32 0
  %31 = load i32* %30
  store %"class.Kalmar::__index_leaf"* %26, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %31, i32* %2, align 4
  %32 = load %"class.Kalmar::__index_leaf"** %1
  %33 = load i32* %2, align 4
  %34 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %32, i32 0, i32 0
  %35 = load i32* %34, align 4
  %36 = mul nsw i32 %35, %33
  store i32 %36, i32* %34, align 4
  %37 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  %38 = bitcast %"class.Kalmar::__index_leaf"* %32 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %37, i8* %38, i64 8, i32 4, i1 false) #2
  %39 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %40 = getelementptr inbounds i8* %39, i64 8
  %41 = bitcast i8* %40 to %"class.Kalmar::__index_leaf.2"*
  %42 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %43 = bitcast %"struct.Kalmar::index_impl.4"* %42 to i8*
  %44 = getelementptr inbounds i8* %43, i64 8
  %45 = bitcast i8* %44 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %45, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %46 = load %"class.Kalmar::__index_leaf.2"** %3
  %47 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %46, i32 0, i32 0
  %48 = load i32* %47
  store %"class.Kalmar::__index_leaf.2"* %41, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %48, i32* %5, align 4
  %49 = load %"class.Kalmar::__index_leaf.2"** %4
  %50 = load i32* %5, align 4
  %51 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %49, i32 0, i32 0
  %52 = load i32* %51, align 4
  %53 = mul nsw i32 %52, %50
  store i32 %53, i32* %51, align 4
  %54 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i8*
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %49 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %54, i8* %55, i64 8, i32 4, i1 false) #2
  %56 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %57 = getelementptr inbounds i8* %56, i64 16
  %58 = bitcast i8* %57 to %"class.Kalmar::__index_leaf.5"*
  %59 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %60 = bitcast %"struct.Kalmar::index_impl.4"* %59 to i8*
  %61 = getelementptr inbounds i8* %60, i64 16
  %62 = bitcast i8* %61 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %62, %"class.Kalmar::__index_leaf.5"** %6, align 8
  %63 = load %"class.Kalmar::__index_leaf.5"** %6
  %64 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %63, i32 0, i32 0
  %65 = load i32* %64
  store %"class.Kalmar::__index_leaf.5"* %58, %"class.Kalmar::__index_leaf.5"** %7, align 8
  store i32 %65, i32* %8, align 4
  %66 = load %"class.Kalmar::__index_leaf.5"** %7
  %67 = load i32* %8, align 4
  %68 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %66, i32 0, i32 0
  %69 = load i32* %68, align 4
  %70 = mul nsw i32 %69, %67
  store i32 %70, i32* %68, align 4
  %71 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i8*
  %72 = bitcast %"class.Kalmar::__index_leaf.5"* %66 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %71, i8* %72, i64 8, i32 4, i1 false) #2
  %73 = bitcast %"class.Kalmar::__index_leaf"* %16 to i64*
  %74 = load i64* %73, align 1
  %75 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i64*
  %76 = load i64* %75, align 1
  %77 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i64*
  %78 = load i64* %77, align 1
  %79 = bitcast %"class.Kalmar::__index_leaf"* %9 to i64*
  store i64 %74, i64* %79, align 1
  %80 = bitcast %"class.Kalmar::__index_leaf.2"* %10 to i64*
  store i64 %76, i64* %80, align 1
  %81 = bitcast %"class.Kalmar::__index_leaf.5"* %11 to i64*
  store i64 %78, i64* %81, align 1
  store %"struct.Kalmar::index_impl.4"* %25, %"struct.Kalmar::index_impl.4"** %12, align 8
  %82 = load %"struct.Kalmar::index_impl.4"** %12
  ret %"class.Kalmar::index.3"* %21
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEdVERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %__r) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::__index_leaf", align 8
  %10 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %11 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %16 = alloca %"class.Kalmar::__index_leaf", align 4
  %17 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %18 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %19 = alloca %"class.Kalmar::index.3"*, align 8
  %20 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %19, align 8
  store %"class.Kalmar::index.3"* %__r, %"class.Kalmar::index.3"** %20, align 8
  %21 = load %"class.Kalmar::index.3"** %19
  %22 = getelementptr inbounds %"class.Kalmar::index.3"* %21, i32 0, i32 0
  %23 = load %"class.Kalmar::index.3"** %20, align 8
  %24 = getelementptr inbounds %"class.Kalmar::index.3"* %23, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %22, %"struct.Kalmar::index_impl.4"** %14, align 8
  store %"struct.Kalmar::index_impl.4"* %24, %"struct.Kalmar::index_impl.4"** %15, align 8
  %25 = load %"struct.Kalmar::index_impl.4"** %14
  %26 = bitcast %"struct.Kalmar::index_impl.4"* %25 to %"class.Kalmar::__index_leaf"*
  %27 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %28 = bitcast %"struct.Kalmar::index_impl.4"* %27 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %28, %"class.Kalmar::__index_leaf"** %13, align 8
  %29 = load %"class.Kalmar::__index_leaf"** %13
  %30 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %29, i32 0, i32 0
  %31 = load i32* %30
  store %"class.Kalmar::__index_leaf"* %26, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %31, i32* %2, align 4
  %32 = load %"class.Kalmar::__index_leaf"** %1
  %33 = load i32* %2, align 4
  %34 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %32, i32 0, i32 0
  %35 = load i32* %34, align 4
  %36 = sdiv i32 %35, %33
  store i32 %36, i32* %34, align 4
  %37 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  %38 = bitcast %"class.Kalmar::__index_leaf"* %32 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %37, i8* %38, i64 8, i32 4, i1 false) #2
  %39 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %40 = getelementptr inbounds i8* %39, i64 8
  %41 = bitcast i8* %40 to %"class.Kalmar::__index_leaf.2"*
  %42 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %43 = bitcast %"struct.Kalmar::index_impl.4"* %42 to i8*
  %44 = getelementptr inbounds i8* %43, i64 8
  %45 = bitcast i8* %44 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %45, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %46 = load %"class.Kalmar::__index_leaf.2"** %3
  %47 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %46, i32 0, i32 0
  %48 = load i32* %47
  store %"class.Kalmar::__index_leaf.2"* %41, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %48, i32* %5, align 4
  %49 = load %"class.Kalmar::__index_leaf.2"** %4
  %50 = load i32* %5, align 4
  %51 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %49, i32 0, i32 0
  %52 = load i32* %51, align 4
  %53 = sdiv i32 %52, %50
  store i32 %53, i32* %51, align 4
  %54 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i8*
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %49 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %54, i8* %55, i64 8, i32 4, i1 false) #2
  %56 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %57 = getelementptr inbounds i8* %56, i64 16
  %58 = bitcast i8* %57 to %"class.Kalmar::__index_leaf.5"*
  %59 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %60 = bitcast %"struct.Kalmar::index_impl.4"* %59 to i8*
  %61 = getelementptr inbounds i8* %60, i64 16
  %62 = bitcast i8* %61 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %62, %"class.Kalmar::__index_leaf.5"** %6, align 8
  %63 = load %"class.Kalmar::__index_leaf.5"** %6
  %64 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %63, i32 0, i32 0
  %65 = load i32* %64
  store %"class.Kalmar::__index_leaf.5"* %58, %"class.Kalmar::__index_leaf.5"** %7, align 8
  store i32 %65, i32* %8, align 4
  %66 = load %"class.Kalmar::__index_leaf.5"** %7
  %67 = load i32* %8, align 4
  %68 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %66, i32 0, i32 0
  %69 = load i32* %68, align 4
  %70 = sdiv i32 %69, %67
  store i32 %70, i32* %68, align 4
  %71 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i8*
  %72 = bitcast %"class.Kalmar::__index_leaf.5"* %66 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %71, i8* %72, i64 8, i32 4, i1 false) #2
  %73 = bitcast %"class.Kalmar::__index_leaf"* %16 to i64*
  %74 = load i64* %73, align 1
  %75 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i64*
  %76 = load i64* %75, align 1
  %77 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i64*
  %78 = load i64* %77, align 1
  %79 = bitcast %"class.Kalmar::__index_leaf"* %9 to i64*
  store i64 %74, i64* %79, align 1
  %80 = bitcast %"class.Kalmar::__index_leaf.2"* %10 to i64*
  store i64 %76, i64* %80, align 1
  %81 = bitcast %"class.Kalmar::__index_leaf.5"* %11 to i64*
  store i64 %78, i64* %81, align 1
  store %"struct.Kalmar::index_impl.4"* %25, %"struct.Kalmar::index_impl.4"** %12, align 8
  %82 = load %"struct.Kalmar::index_impl.4"** %12
  ret %"class.Kalmar::index.3"* %21
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EErMERKS1_(%"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"* dereferenceable(24) %__r) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %4 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %8 = alloca i32, align 4
  %9 = alloca %"class.Kalmar::__index_leaf", align 8
  %10 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %11 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %14 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %15 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %16 = alloca %"class.Kalmar::__index_leaf", align 4
  %17 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %18 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %19 = alloca %"class.Kalmar::index.3"*, align 8
  %20 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %19, align 8
  store %"class.Kalmar::index.3"* %__r, %"class.Kalmar::index.3"** %20, align 8
  %21 = load %"class.Kalmar::index.3"** %19
  %22 = getelementptr inbounds %"class.Kalmar::index.3"* %21, i32 0, i32 0
  %23 = load %"class.Kalmar::index.3"** %20, align 8
  %24 = getelementptr inbounds %"class.Kalmar::index.3"* %23, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %22, %"struct.Kalmar::index_impl.4"** %14, align 8
  store %"struct.Kalmar::index_impl.4"* %24, %"struct.Kalmar::index_impl.4"** %15, align 8
  %25 = load %"struct.Kalmar::index_impl.4"** %14
  %26 = bitcast %"struct.Kalmar::index_impl.4"* %25 to %"class.Kalmar::__index_leaf"*
  %27 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %28 = bitcast %"struct.Kalmar::index_impl.4"* %27 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %28, %"class.Kalmar::__index_leaf"** %13, align 8
  %29 = load %"class.Kalmar::__index_leaf"** %13
  %30 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %29, i32 0, i32 0
  %31 = load i32* %30
  store %"class.Kalmar::__index_leaf"* %26, %"class.Kalmar::__index_leaf"** %1, align 8
  store i32 %31, i32* %2, align 4
  %32 = load %"class.Kalmar::__index_leaf"** %1
  %33 = load i32* %2, align 4
  %34 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %32, i32 0, i32 0
  %35 = load i32* %34, align 4
  %36 = srem i32 %35, %33
  store i32 %36, i32* %34, align 4
  %37 = bitcast %"class.Kalmar::__index_leaf"* %16 to i8*
  %38 = bitcast %"class.Kalmar::__index_leaf"* %32 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %37, i8* %38, i64 8, i32 4, i1 false)
  %39 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %40 = getelementptr inbounds i8* %39, i64 8
  %41 = bitcast i8* %40 to %"class.Kalmar::__index_leaf.2"*
  %42 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %43 = bitcast %"struct.Kalmar::index_impl.4"* %42 to i8*
  %44 = getelementptr inbounds i8* %43, i64 8
  %45 = bitcast i8* %44 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %45, %"class.Kalmar::__index_leaf.2"** %3, align 8
  %46 = load %"class.Kalmar::__index_leaf.2"** %3
  %47 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %46, i32 0, i32 0
  %48 = load i32* %47
  store %"class.Kalmar::__index_leaf.2"* %41, %"class.Kalmar::__index_leaf.2"** %4, align 8
  store i32 %48, i32* %5, align 4
  %49 = load %"class.Kalmar::__index_leaf.2"** %4
  %50 = load i32* %5, align 4
  %51 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %49, i32 0, i32 0
  %52 = load i32* %51, align 4
  %53 = srem i32 %52, %50
  store i32 %53, i32* %51, align 4
  %54 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i8*
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %49 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %54, i8* %55, i64 8, i32 4, i1 false)
  %56 = bitcast %"struct.Kalmar::index_impl.4"* %25 to i8*
  %57 = getelementptr inbounds i8* %56, i64 16
  %58 = bitcast i8* %57 to %"class.Kalmar::__index_leaf.5"*
  %59 = load %"struct.Kalmar::index_impl.4"** %15, align 8
  %60 = bitcast %"struct.Kalmar::index_impl.4"* %59 to i8*
  %61 = getelementptr inbounds i8* %60, i64 16
  %62 = bitcast i8* %61 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %62, %"class.Kalmar::__index_leaf.5"** %6, align 8
  %63 = load %"class.Kalmar::__index_leaf.5"** %6
  %64 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %63, i32 0, i32 0
  %65 = load i32* %64
  store %"class.Kalmar::__index_leaf.5"* %58, %"class.Kalmar::__index_leaf.5"** %7, align 8
  store i32 %65, i32* %8, align 4
  %66 = load %"class.Kalmar::__index_leaf.5"** %7
  %67 = load i32* %8, align 4
  %68 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %66, i32 0, i32 0
  %69 = load i32* %68, align 4
  %70 = srem i32 %69, %67
  store i32 %70, i32* %68, align 4
  %71 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i8*
  %72 = bitcast %"class.Kalmar::__index_leaf.5"* %66 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %71, i8* %72, i64 8, i32 4, i1 false)
  %73 = bitcast %"class.Kalmar::__index_leaf"* %16 to i64*
  %74 = load i64* %73, align 1
  %75 = bitcast %"class.Kalmar::__index_leaf.2"* %17 to i64*
  %76 = load i64* %75, align 1
  %77 = bitcast %"class.Kalmar::__index_leaf.5"* %18 to i64*
  %78 = load i64* %77, align 1
  %79 = bitcast %"class.Kalmar::__index_leaf"* %9 to i64*
  store i64 %74, i64* %79, align 1
  %80 = bitcast %"class.Kalmar::__index_leaf.2"* %10 to i64*
  store i64 %76, i64* %80, align 1
  %81 = bitcast %"class.Kalmar::__index_leaf.5"* %11 to i64*
  store i64 %78, i64* %81, align 1
  store %"struct.Kalmar::index_impl.4"* %25, %"struct.Kalmar::index_impl.4"** %12, align 8
  %82 = load %"struct.Kalmar::index_impl.4"** %12
  ret %"class.Kalmar::index.3"* %21
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEpLEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf", align 8
  %6 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %8 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf", align 4
  %14 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %15 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %16 = alloca %"class.Kalmar::index.3"*, align 8
  %17 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %16, align 8
  store i32 %value, i32* %17, align 4
  %18 = load %"class.Kalmar::index.3"** %16
  %19 = getelementptr inbounds %"class.Kalmar::index.3"* %18, i32 0, i32 0
  %20 = load i32* %17, align 4
  store %"struct.Kalmar::index_impl.4"* %19, %"struct.Kalmar::index_impl.4"** %11, align 8
  store i32 %20, i32* %12, align 4
  %21 = load %"struct.Kalmar::index_impl.4"** %11
  %22 = bitcast %"struct.Kalmar::index_impl.4"* %21 to %"class.Kalmar::__index_leaf"*
  %23 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %22, %"class.Kalmar::__index_leaf"** %9, align 8
  store i32 %23, i32* %10, align 4
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = load i32* %10, align 4
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %27 = load i32* %26, align 4
  %28 = add nsw i32 %27, %25
  store i32 %28, i32* %26, align 4
  %29 = bitcast %"class.Kalmar::__index_leaf"* %13 to i8*
  %30 = bitcast %"class.Kalmar::__index_leaf"* %24 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %29, i8* %30, i64 8, i32 4, i1 false)
  %31 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %32 = getelementptr inbounds i8* %31, i64 8
  %33 = bitcast i8* %32 to %"class.Kalmar::__index_leaf.2"*
  %34 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.2"* %33, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %34, i32* %2, align 4
  %35 = load %"class.Kalmar::__index_leaf.2"** %1
  %36 = load i32* %2, align 4
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %35, i32 0, i32 0
  %38 = load i32* %37, align 4
  %39 = add nsw i32 %38, %36
  store i32 %39, i32* %37, align 4
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i8*
  %41 = bitcast %"class.Kalmar::__index_leaf.2"* %35 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %40, i8* %41, i64 8, i32 4, i1 false)
  %42 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %43 = getelementptr inbounds i8* %42, i64 16
  %44 = bitcast i8* %43 to %"class.Kalmar::__index_leaf.5"*
  %45 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.5"* %44, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %45, i32* %4, align 4
  %46 = load %"class.Kalmar::__index_leaf.5"** %3
  %47 = load i32* %4, align 4
  %48 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %46, i32 0, i32 0
  %49 = load i32* %48, align 4
  %50 = add nsw i32 %49, %47
  store i32 %50, i32* %48, align 4
  %51 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i8*
  %52 = bitcast %"class.Kalmar::__index_leaf.5"* %46 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %51, i8* %52, i64 8, i32 4, i1 false)
  %53 = bitcast %"class.Kalmar::__index_leaf"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i64*
  %56 = load i64* %55, align 1
  %57 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i64*
  %58 = load i64* %57, align 1
  %59 = bitcast %"class.Kalmar::__index_leaf"* %5 to i64*
  store i64 %54, i64* %59, align 1
  %60 = bitcast %"class.Kalmar::__index_leaf.2"* %6 to i64*
  store i64 %56, i64* %60, align 1
  %61 = bitcast %"class.Kalmar::__index_leaf.5"* %7 to i64*
  store i64 %58, i64* %61, align 1
  store %"struct.Kalmar::index_impl.4"* %21, %"struct.Kalmar::index_impl.4"** %8, align 8
  %62 = load %"struct.Kalmar::index_impl.4"** %8
  ret %"class.Kalmar::index.3"* %18
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmIEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf", align 8
  %6 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %8 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf", align 4
  %14 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %15 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %16 = alloca %"class.Kalmar::index.3"*, align 8
  %17 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %16, align 8
  store i32 %value, i32* %17, align 4
  %18 = load %"class.Kalmar::index.3"** %16
  %19 = getelementptr inbounds %"class.Kalmar::index.3"* %18, i32 0, i32 0
  %20 = load i32* %17, align 4
  store %"struct.Kalmar::index_impl.4"* %19, %"struct.Kalmar::index_impl.4"** %11, align 8
  store i32 %20, i32* %12, align 4
  %21 = load %"struct.Kalmar::index_impl.4"** %11
  %22 = bitcast %"struct.Kalmar::index_impl.4"* %21 to %"class.Kalmar::__index_leaf"*
  %23 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %22, %"class.Kalmar::__index_leaf"** %9, align 8
  store i32 %23, i32* %10, align 4
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = load i32* %10, align 4
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %27 = load i32* %26, align 4
  %28 = sub nsw i32 %27, %25
  store i32 %28, i32* %26, align 4
  %29 = bitcast %"class.Kalmar::__index_leaf"* %13 to i8*
  %30 = bitcast %"class.Kalmar::__index_leaf"* %24 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %29, i8* %30, i64 8, i32 4, i1 false)
  %31 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %32 = getelementptr inbounds i8* %31, i64 8
  %33 = bitcast i8* %32 to %"class.Kalmar::__index_leaf.2"*
  %34 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.2"* %33, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %34, i32* %2, align 4
  %35 = load %"class.Kalmar::__index_leaf.2"** %1
  %36 = load i32* %2, align 4
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %35, i32 0, i32 0
  %38 = load i32* %37, align 4
  %39 = sub nsw i32 %38, %36
  store i32 %39, i32* %37, align 4
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i8*
  %41 = bitcast %"class.Kalmar::__index_leaf.2"* %35 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %40, i8* %41, i64 8, i32 4, i1 false)
  %42 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %43 = getelementptr inbounds i8* %42, i64 16
  %44 = bitcast i8* %43 to %"class.Kalmar::__index_leaf.5"*
  %45 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.5"* %44, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %45, i32* %4, align 4
  %46 = load %"class.Kalmar::__index_leaf.5"** %3
  %47 = load i32* %4, align 4
  %48 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %46, i32 0, i32 0
  %49 = load i32* %48, align 4
  %50 = sub nsw i32 %49, %47
  store i32 %50, i32* %48, align 4
  %51 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i8*
  %52 = bitcast %"class.Kalmar::__index_leaf.5"* %46 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %51, i8* %52, i64 8, i32 4, i1 false)
  %53 = bitcast %"class.Kalmar::__index_leaf"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i64*
  %56 = load i64* %55, align 1
  %57 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i64*
  %58 = load i64* %57, align 1
  %59 = bitcast %"class.Kalmar::__index_leaf"* %5 to i64*
  store i64 %54, i64* %59, align 1
  %60 = bitcast %"class.Kalmar::__index_leaf.2"* %6 to i64*
  store i64 %56, i64* %60, align 1
  %61 = bitcast %"class.Kalmar::__index_leaf.5"* %7 to i64*
  store i64 %58, i64* %61, align 1
  store %"struct.Kalmar::index_impl.4"* %21, %"struct.Kalmar::index_impl.4"** %8, align 8
  %62 = load %"struct.Kalmar::index_impl.4"** %8
  ret %"class.Kalmar::index.3"* %18
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmLEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf", align 8
  %6 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %8 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf", align 4
  %14 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %15 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %16 = alloca %"class.Kalmar::index.3"*, align 8
  %17 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %16, align 8
  store i32 %value, i32* %17, align 4
  %18 = load %"class.Kalmar::index.3"** %16
  %19 = getelementptr inbounds %"class.Kalmar::index.3"* %18, i32 0, i32 0
  %20 = load i32* %17, align 4
  store %"struct.Kalmar::index_impl.4"* %19, %"struct.Kalmar::index_impl.4"** %11, align 8
  store i32 %20, i32* %12, align 4
  %21 = load %"struct.Kalmar::index_impl.4"** %11
  %22 = bitcast %"struct.Kalmar::index_impl.4"* %21 to %"class.Kalmar::__index_leaf"*
  %23 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %22, %"class.Kalmar::__index_leaf"** %9, align 8
  store i32 %23, i32* %10, align 4
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = load i32* %10, align 4
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %27 = load i32* %26, align 4
  %28 = mul nsw i32 %27, %25
  store i32 %28, i32* %26, align 4
  %29 = bitcast %"class.Kalmar::__index_leaf"* %13 to i8*
  %30 = bitcast %"class.Kalmar::__index_leaf"* %24 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %29, i8* %30, i64 8, i32 4, i1 false)
  %31 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %32 = getelementptr inbounds i8* %31, i64 8
  %33 = bitcast i8* %32 to %"class.Kalmar::__index_leaf.2"*
  %34 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.2"* %33, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %34, i32* %2, align 4
  %35 = load %"class.Kalmar::__index_leaf.2"** %1
  %36 = load i32* %2, align 4
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %35, i32 0, i32 0
  %38 = load i32* %37, align 4
  %39 = mul nsw i32 %38, %36
  store i32 %39, i32* %37, align 4
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i8*
  %41 = bitcast %"class.Kalmar::__index_leaf.2"* %35 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %40, i8* %41, i64 8, i32 4, i1 false)
  %42 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %43 = getelementptr inbounds i8* %42, i64 16
  %44 = bitcast i8* %43 to %"class.Kalmar::__index_leaf.5"*
  %45 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.5"* %44, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %45, i32* %4, align 4
  %46 = load %"class.Kalmar::__index_leaf.5"** %3
  %47 = load i32* %4, align 4
  %48 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %46, i32 0, i32 0
  %49 = load i32* %48, align 4
  %50 = mul nsw i32 %49, %47
  store i32 %50, i32* %48, align 4
  %51 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i8*
  %52 = bitcast %"class.Kalmar::__index_leaf.5"* %46 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %51, i8* %52, i64 8, i32 4, i1 false)
  %53 = bitcast %"class.Kalmar::__index_leaf"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i64*
  %56 = load i64* %55, align 1
  %57 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i64*
  %58 = load i64* %57, align 1
  %59 = bitcast %"class.Kalmar::__index_leaf"* %5 to i64*
  store i64 %54, i64* %59, align 1
  %60 = bitcast %"class.Kalmar::__index_leaf.2"* %6 to i64*
  store i64 %56, i64* %60, align 1
  %61 = bitcast %"class.Kalmar::__index_leaf.5"* %7 to i64*
  store i64 %58, i64* %61, align 1
  store %"struct.Kalmar::index_impl.4"* %21, %"struct.Kalmar::index_impl.4"** %8, align 8
  %62 = load %"struct.Kalmar::index_impl.4"** %8
  ret %"class.Kalmar::index.3"* %18
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEdVEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf", align 8
  %6 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %8 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf", align 4
  %14 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %15 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %16 = alloca %"class.Kalmar::index.3"*, align 8
  %17 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %16, align 8
  store i32 %value, i32* %17, align 4
  %18 = load %"class.Kalmar::index.3"** %16
  %19 = getelementptr inbounds %"class.Kalmar::index.3"* %18, i32 0, i32 0
  %20 = load i32* %17, align 4
  store %"struct.Kalmar::index_impl.4"* %19, %"struct.Kalmar::index_impl.4"** %11, align 8
  store i32 %20, i32* %12, align 4
  %21 = load %"struct.Kalmar::index_impl.4"** %11
  %22 = bitcast %"struct.Kalmar::index_impl.4"* %21 to %"class.Kalmar::__index_leaf"*
  %23 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %22, %"class.Kalmar::__index_leaf"** %9, align 8
  store i32 %23, i32* %10, align 4
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = load i32* %10, align 4
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %27 = load i32* %26, align 4
  %28 = sdiv i32 %27, %25
  store i32 %28, i32* %26, align 4
  %29 = bitcast %"class.Kalmar::__index_leaf"* %13 to i8*
  %30 = bitcast %"class.Kalmar::__index_leaf"* %24 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %29, i8* %30, i64 8, i32 4, i1 false)
  %31 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %32 = getelementptr inbounds i8* %31, i64 8
  %33 = bitcast i8* %32 to %"class.Kalmar::__index_leaf.2"*
  %34 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.2"* %33, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %34, i32* %2, align 4
  %35 = load %"class.Kalmar::__index_leaf.2"** %1
  %36 = load i32* %2, align 4
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %35, i32 0, i32 0
  %38 = load i32* %37, align 4
  %39 = sdiv i32 %38, %36
  store i32 %39, i32* %37, align 4
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i8*
  %41 = bitcast %"class.Kalmar::__index_leaf.2"* %35 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %40, i8* %41, i64 8, i32 4, i1 false)
  %42 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %43 = getelementptr inbounds i8* %42, i64 16
  %44 = bitcast i8* %43 to %"class.Kalmar::__index_leaf.5"*
  %45 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.5"* %44, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %45, i32* %4, align 4
  %46 = load %"class.Kalmar::__index_leaf.5"** %3
  %47 = load i32* %4, align 4
  %48 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %46, i32 0, i32 0
  %49 = load i32* %48, align 4
  %50 = sdiv i32 %49, %47
  store i32 %50, i32* %48, align 4
  %51 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i8*
  %52 = bitcast %"class.Kalmar::__index_leaf.5"* %46 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %51, i8* %52, i64 8, i32 4, i1 false)
  %53 = bitcast %"class.Kalmar::__index_leaf"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i64*
  %56 = load i64* %55, align 1
  %57 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i64*
  %58 = load i64* %57, align 1
  %59 = bitcast %"class.Kalmar::__index_leaf"* %5 to i64*
  store i64 %54, i64* %59, align 1
  %60 = bitcast %"class.Kalmar::__index_leaf.2"* %6 to i64*
  store i64 %56, i64* %60, align 1
  %61 = bitcast %"class.Kalmar::__index_leaf.5"* %7 to i64*
  store i64 %58, i64* %61, align 1
  store %"struct.Kalmar::index_impl.4"* %21, %"struct.Kalmar::index_impl.4"** %8, align 8
  %62 = load %"struct.Kalmar::index_impl.4"** %8
  ret %"class.Kalmar::index.3"* %18
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EErMEi(%"class.Kalmar::index.3"* %this, i32 %value) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf", align 8
  %6 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %8 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf", align 4
  %14 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %15 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %16 = alloca %"class.Kalmar::index.3"*, align 8
  %17 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %16, align 8
  store i32 %value, i32* %17, align 4
  %18 = load %"class.Kalmar::index.3"** %16
  %19 = getelementptr inbounds %"class.Kalmar::index.3"* %18, i32 0, i32 0
  %20 = load i32* %17, align 4
  store %"struct.Kalmar::index_impl.4"* %19, %"struct.Kalmar::index_impl.4"** %11, align 8
  store i32 %20, i32* %12, align 4
  %21 = load %"struct.Kalmar::index_impl.4"** %11
  %22 = bitcast %"struct.Kalmar::index_impl.4"* %21 to %"class.Kalmar::__index_leaf"*
  %23 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %22, %"class.Kalmar::__index_leaf"** %9, align 8
  store i32 %23, i32* %10, align 4
  %24 = load %"class.Kalmar::__index_leaf"** %9
  %25 = load i32* %10, align 4
  %26 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %24, i32 0, i32 0
  %27 = load i32* %26, align 4
  %28 = srem i32 %27, %25
  store i32 %28, i32* %26, align 4
  %29 = bitcast %"class.Kalmar::__index_leaf"* %13 to i8*
  %30 = bitcast %"class.Kalmar::__index_leaf"* %24 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %29, i8* %30, i64 8, i32 4, i1 false)
  %31 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %32 = getelementptr inbounds i8* %31, i64 8
  %33 = bitcast i8* %32 to %"class.Kalmar::__index_leaf.2"*
  %34 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.2"* %33, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %34, i32* %2, align 4
  %35 = load %"class.Kalmar::__index_leaf.2"** %1
  %36 = load i32* %2, align 4
  %37 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %35, i32 0, i32 0
  %38 = load i32* %37, align 4
  %39 = srem i32 %38, %36
  store i32 %39, i32* %37, align 4
  %40 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i8*
  %41 = bitcast %"class.Kalmar::__index_leaf.2"* %35 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %40, i8* %41, i64 8, i32 4, i1 false)
  %42 = bitcast %"struct.Kalmar::index_impl.4"* %21 to i8*
  %43 = getelementptr inbounds i8* %42, i64 16
  %44 = bitcast i8* %43 to %"class.Kalmar::__index_leaf.5"*
  %45 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.5"* %44, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %45, i32* %4, align 4
  %46 = load %"class.Kalmar::__index_leaf.5"** %3
  %47 = load i32* %4, align 4
  %48 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %46, i32 0, i32 0
  %49 = load i32* %48, align 4
  %50 = srem i32 %49, %47
  store i32 %50, i32* %48, align 4
  %51 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i8*
  %52 = bitcast %"class.Kalmar::__index_leaf.5"* %46 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %51, i8* %52, i64 8, i32 4, i1 false)
  %53 = bitcast %"class.Kalmar::__index_leaf"* %13 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i64*
  %56 = load i64* %55, align 1
  %57 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i64*
  %58 = load i64* %57, align 1
  %59 = bitcast %"class.Kalmar::__index_leaf"* %5 to i64*
  store i64 %54, i64* %59, align 1
  %60 = bitcast %"class.Kalmar::__index_leaf.2"* %6 to i64*
  store i64 %56, i64* %60, align 1
  %61 = bitcast %"class.Kalmar::__index_leaf.5"* %7 to i64*
  store i64 %58, i64* %61, align 1
  store %"struct.Kalmar::index_impl.4"* %21, %"struct.Kalmar::index_impl.4"** %8, align 8
  %62 = load %"struct.Kalmar::index_impl.4"** %8
  ret %"class.Kalmar::index.3"* %18
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEppEv(%"class.Kalmar::index.3"* %this) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf", align 8
  %6 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %8 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf", align 4
  %14 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %15 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %16 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %16, align 8
  %17 = load %"class.Kalmar::index.3"** %16
  %18 = getelementptr inbounds %"class.Kalmar::index.3"* %17, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %18, %"struct.Kalmar::index_impl.4"** %11, align 8
  store i32 1, i32* %12, align 4
  %19 = load %"struct.Kalmar::index_impl.4"** %11
  %20 = bitcast %"struct.Kalmar::index_impl.4"* %19 to %"class.Kalmar::__index_leaf"*
  %21 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %20, %"class.Kalmar::__index_leaf"** %9, align 8
  store i32 %21, i32* %10, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %9
  %23 = load i32* %10, align 4
  %24 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  %25 = load i32* %24, align 4
  %26 = add nsw i32 %25, %23
  store i32 %26, i32* %24, align 4
  %27 = bitcast %"class.Kalmar::__index_leaf"* %13 to i8*
  %28 = bitcast %"class.Kalmar::__index_leaf"* %22 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %27, i8* %28, i64 8, i32 4, i1 false)
  %29 = bitcast %"struct.Kalmar::index_impl.4"* %19 to i8*
  %30 = getelementptr inbounds i8* %29, i64 8
  %31 = bitcast i8* %30 to %"class.Kalmar::__index_leaf.2"*
  %32 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.2"* %31, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %32, i32* %2, align 4
  %33 = load %"class.Kalmar::__index_leaf.2"** %1
  %34 = load i32* %2, align 4
  %35 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %33, i32 0, i32 0
  %36 = load i32* %35, align 4
  %37 = add nsw i32 %36, %34
  store i32 %37, i32* %35, align 4
  %38 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i8*
  %39 = bitcast %"class.Kalmar::__index_leaf.2"* %33 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %38, i8* %39, i64 8, i32 4, i1 false)
  %40 = bitcast %"struct.Kalmar::index_impl.4"* %19 to i8*
  %41 = getelementptr inbounds i8* %40, i64 16
  %42 = bitcast i8* %41 to %"class.Kalmar::__index_leaf.5"*
  %43 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.5"* %42, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %43, i32* %4, align 4
  %44 = load %"class.Kalmar::__index_leaf.5"** %3
  %45 = load i32* %4, align 4
  %46 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %44, i32 0, i32 0
  %47 = load i32* %46, align 4
  %48 = add nsw i32 %47, %45
  store i32 %48, i32* %46, align 4
  %49 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i8*
  %50 = bitcast %"class.Kalmar::__index_leaf.5"* %44 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %49, i8* %50, i64 8, i32 4, i1 false)
  %51 = bitcast %"class.Kalmar::__index_leaf"* %13 to i64*
  %52 = load i64* %51, align 1
  %53 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i64*
  %56 = load i64* %55, align 1
  %57 = bitcast %"class.Kalmar::__index_leaf"* %5 to i64*
  store i64 %52, i64* %57, align 1
  %58 = bitcast %"class.Kalmar::__index_leaf.2"* %6 to i64*
  store i64 %54, i64* %58, align 1
  %59 = bitcast %"class.Kalmar::__index_leaf.5"* %7 to i64*
  store i64 %56, i64* %59, align 1
  store %"struct.Kalmar::index_impl.4"* %19, %"struct.Kalmar::index_impl.4"** %8, align 8
  %60 = load %"struct.Kalmar::index_impl.4"** %8
  ret %"class.Kalmar::index.3"* %17
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEppEi(%"class.Kalmar::index.3"* noalias sret %agg.result, %"class.Kalmar::index.3"* %this, i32) #0 align 2 {
  %2 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %8 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %9 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %10 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %11 = alloca i32, align 4
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca i32, align 4
  %14 = alloca %"class.Kalmar::__index_leaf", align 4
  %15 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %16 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %17 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %18 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %19 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %20 = alloca i32, align 4
  %21 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %22 = alloca i32, align 4
  %23 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %24 = alloca i32, align 4
  %25 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %26 = alloca i32, align 4
  %27 = alloca i32, align 4
  %28 = alloca i32, align 4
  %29 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %30 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %31 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %32 = alloca %"class.Kalmar::index.3"*, align 8
  %33 = alloca %"class.Kalmar::index.3"*, align 8
  %34 = alloca %"class.Kalmar::index.3"*, align 8
  %35 = alloca %"class.Kalmar::index.3"*, align 8
  %36 = alloca %"class.Kalmar::index.3"*, align 8
  %37 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %36, align 8
  store i32 %0, i32* %37, align 4
  %38 = load %"class.Kalmar::index.3"** %36
  store %"class.Kalmar::index.3"* %agg.result, %"class.Kalmar::index.3"** %34, align 8
  store %"class.Kalmar::index.3"* %38, %"class.Kalmar::index.3"** %35, align 8
  %39 = load %"class.Kalmar::index.3"** %34
  %40 = load %"class.Kalmar::index.3"** %35
  store %"class.Kalmar::index.3"* %39, %"class.Kalmar::index.3"** %32, align 8
  store %"class.Kalmar::index.3"* %40, %"class.Kalmar::index.3"** %33, align 8
  %41 = load %"class.Kalmar::index.3"** %32
  %42 = getelementptr inbounds %"class.Kalmar::index.3"* %41, i32 0, i32 0
  %43 = load %"class.Kalmar::index.3"** %33, align 8
  %44 = getelementptr inbounds %"class.Kalmar::index.3"* %43, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %42, %"struct.Kalmar::index_impl.4"** %30, align 8
  store %"struct.Kalmar::index_impl.4"* %44, %"struct.Kalmar::index_impl.4"** %31, align 8
  %45 = load %"struct.Kalmar::index_impl.4"** %30
  %46 = load %"struct.Kalmar::index_impl.4"** %31, align 8
  %47 = bitcast %"struct.Kalmar::index_impl.4"* %46 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %47, %"class.Kalmar::__index_leaf"** %29, align 8
  %48 = load %"class.Kalmar::__index_leaf"** %29
  %49 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %48, i32 0, i32 0
  %50 = load i32* %49
  %51 = load %"struct.Kalmar::index_impl.4"** %31, align 8
  %52 = bitcast %"struct.Kalmar::index_impl.4"* %51 to i8*
  %53 = getelementptr inbounds i8* %52, i64 8
  %54 = bitcast i8* %53 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %54, %"class.Kalmar::__index_leaf.2"** %17, align 8
  %55 = load %"class.Kalmar::__index_leaf.2"** %17
  %56 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %55, i32 0, i32 0
  %57 = load i32* %56
  %58 = load %"struct.Kalmar::index_impl.4"** %31, align 8
  %59 = bitcast %"struct.Kalmar::index_impl.4"* %58 to i8*
  %60 = getelementptr inbounds i8* %59, i64 16
  %61 = bitcast i8* %60 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %61, %"class.Kalmar::__index_leaf.5"** %18, align 8
  %62 = load %"class.Kalmar::__index_leaf.5"** %18
  %63 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %62, i32 0, i32 0
  %64 = load i32* %63
  store %"struct.Kalmar::index_impl.4"* %45, %"struct.Kalmar::index_impl.4"** %25, align 8
  store i32 %50, i32* %26, align 4
  store i32 %57, i32* %27, align 4
  store i32 %64, i32* %28, align 4
  %65 = load %"struct.Kalmar::index_impl.4"** %25
  %66 = bitcast %"struct.Kalmar::index_impl.4"* %65 to %"class.Kalmar::__index_leaf"*
  %67 = load i32* %26, align 4
  store %"class.Kalmar::__index_leaf"* %66, %"class.Kalmar::__index_leaf"** %23, align 8
  store i32 %67, i32* %24, align 4
  %68 = load %"class.Kalmar::__index_leaf"** %23
  %69 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %68, i32 0, i32 0
  %70 = load i32* %24, align 4
  store i32 %70, i32* %69, align 4
  %71 = bitcast %"struct.Kalmar::index_impl.4"* %65 to i8*
  %72 = getelementptr inbounds i8* %71, i64 8
  %73 = bitcast i8* %72 to %"class.Kalmar::__index_leaf.2"*
  %74 = load i32* %27, align 4
  store %"class.Kalmar::__index_leaf.2"* %73, %"class.Kalmar::__index_leaf.2"** %19, align 8
  store i32 %74, i32* %20, align 4
  %75 = load %"class.Kalmar::__index_leaf.2"** %19
  %76 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %75, i32 0, i32 0
  %77 = load i32* %20, align 4
  store i32 %77, i32* %76, align 4
  %78 = bitcast %"struct.Kalmar::index_impl.4"* %65 to i8*
  %79 = getelementptr inbounds i8* %78, i64 16
  %80 = bitcast i8* %79 to %"class.Kalmar::__index_leaf.5"*
  %81 = load i32* %28, align 4
  store %"class.Kalmar::__index_leaf.5"* %80, %"class.Kalmar::__index_leaf.5"** %21, align 8
  store i32 %81, i32* %22, align 4
  %82 = load %"class.Kalmar::__index_leaf.5"** %21
  %83 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %82, i32 0, i32 0
  %84 = load i32* %22, align 4
  store i32 %84, i32* %83, align 4
  %85 = getelementptr inbounds %"class.Kalmar::index.3"* %38, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %85, %"struct.Kalmar::index_impl.4"** %12, align 8
  store i32 1, i32* %13, align 4
  %86 = load %"struct.Kalmar::index_impl.4"** %12
  %87 = bitcast %"struct.Kalmar::index_impl.4"* %86 to %"class.Kalmar::__index_leaf"*
  %88 = load i32* %13, align 4
  store %"class.Kalmar::__index_leaf"* %87, %"class.Kalmar::__index_leaf"** %10, align 8
  store i32 %88, i32* %11, align 4
  %89 = load %"class.Kalmar::__index_leaf"** %10
  %90 = load i32* %11, align 4
  %91 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %89, i32 0, i32 0
  %92 = load i32* %91, align 4
  %93 = add nsw i32 %92, %90
  store i32 %93, i32* %91, align 4
  %94 = bitcast %"class.Kalmar::__index_leaf"* %14 to i8*
  %95 = bitcast %"class.Kalmar::__index_leaf"* %89 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %94, i8* %95, i64 8, i32 4, i1 false)
  %96 = bitcast %"struct.Kalmar::index_impl.4"* %86 to i8*
  %97 = getelementptr inbounds i8* %96, i64 8
  %98 = bitcast i8* %97 to %"class.Kalmar::__index_leaf.2"*
  %99 = load i32* %13, align 4
  store %"class.Kalmar::__index_leaf.2"* %98, %"class.Kalmar::__index_leaf.2"** %2, align 8
  store i32 %99, i32* %3, align 4
  %100 = load %"class.Kalmar::__index_leaf.2"** %2
  %101 = load i32* %3, align 4
  %102 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %100, i32 0, i32 0
  %103 = load i32* %102, align 4
  %104 = add nsw i32 %103, %101
  store i32 %104, i32* %102, align 4
  %105 = bitcast %"class.Kalmar::__index_leaf.2"* %15 to i8*
  %106 = bitcast %"class.Kalmar::__index_leaf.2"* %100 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %105, i8* %106, i64 8, i32 4, i1 false)
  %107 = bitcast %"struct.Kalmar::index_impl.4"* %86 to i8*
  %108 = getelementptr inbounds i8* %107, i64 16
  %109 = bitcast i8* %108 to %"class.Kalmar::__index_leaf.5"*
  %110 = load i32* %13, align 4
  store %"class.Kalmar::__index_leaf.5"* %109, %"class.Kalmar::__index_leaf.5"** %4, align 8
  store i32 %110, i32* %5, align 4
  %111 = load %"class.Kalmar::__index_leaf.5"** %4
  %112 = load i32* %5, align 4
  %113 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %111, i32 0, i32 0
  %114 = load i32* %113, align 4
  %115 = add nsw i32 %114, %112
  store i32 %115, i32* %113, align 4
  %116 = bitcast %"class.Kalmar::__index_leaf.5"* %16 to i8*
  %117 = bitcast %"class.Kalmar::__index_leaf.5"* %111 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %116, i8* %117, i64 8, i32 4, i1 false)
  %118 = bitcast %"class.Kalmar::__index_leaf"* %14 to i64*
  %119 = load i64* %118, align 1
  %120 = bitcast %"class.Kalmar::__index_leaf.2"* %15 to i64*
  %121 = load i64* %120, align 1
  %122 = bitcast %"class.Kalmar::__index_leaf.5"* %16 to i64*
  %123 = load i64* %122, align 1
  %124 = bitcast %"class.Kalmar::__index_leaf"* %6 to i64*
  store i64 %119, i64* %124, align 1
  %125 = bitcast %"class.Kalmar::__index_leaf.2"* %7 to i64*
  store i64 %121, i64* %125, align 1
  %126 = bitcast %"class.Kalmar::__index_leaf.5"* %8 to i64*
  store i64 %123, i64* %126, align 1
  store %"struct.Kalmar::index_impl.4"* %86, %"struct.Kalmar::index_impl.4"** %9, align 8
  %127 = load %"struct.Kalmar::index_impl.4"** %9
  ret void
}

; Function Attrs: alwaysinline uwtable
define weak_odr dereferenceable(24) %"class.Kalmar::index.3"* @_ZN6Kalmar5indexILi3EEmmEv(%"class.Kalmar::index.3"* %this) #0 align 2 {
  %1 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %2 = alloca i32, align 4
  %3 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %"class.Kalmar::__index_leaf", align 8
  %6 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %8 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %9 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %10 = alloca i32, align 4
  %11 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %12 = alloca i32, align 4
  %13 = alloca %"class.Kalmar::__index_leaf", align 4
  %14 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %15 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %16 = alloca %"class.Kalmar::index.3"*, align 8
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %16, align 8
  %17 = load %"class.Kalmar::index.3"** %16
  %18 = getelementptr inbounds %"class.Kalmar::index.3"* %17, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %18, %"struct.Kalmar::index_impl.4"** %11, align 8
  store i32 1, i32* %12, align 4
  %19 = load %"struct.Kalmar::index_impl.4"** %11
  %20 = bitcast %"struct.Kalmar::index_impl.4"* %19 to %"class.Kalmar::__index_leaf"*
  %21 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf"* %20, %"class.Kalmar::__index_leaf"** %9, align 8
  store i32 %21, i32* %10, align 4
  %22 = load %"class.Kalmar::__index_leaf"** %9
  %23 = load i32* %10, align 4
  %24 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %22, i32 0, i32 0
  %25 = load i32* %24, align 4
  %26 = sub nsw i32 %25, %23
  store i32 %26, i32* %24, align 4
  %27 = bitcast %"class.Kalmar::__index_leaf"* %13 to i8*
  %28 = bitcast %"class.Kalmar::__index_leaf"* %22 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %27, i8* %28, i64 8, i32 4, i1 false)
  %29 = bitcast %"struct.Kalmar::index_impl.4"* %19 to i8*
  %30 = getelementptr inbounds i8* %29, i64 8
  %31 = bitcast i8* %30 to %"class.Kalmar::__index_leaf.2"*
  %32 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.2"* %31, %"class.Kalmar::__index_leaf.2"** %1, align 8
  store i32 %32, i32* %2, align 4
  %33 = load %"class.Kalmar::__index_leaf.2"** %1
  %34 = load i32* %2, align 4
  %35 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %33, i32 0, i32 0
  %36 = load i32* %35, align 4
  %37 = sub nsw i32 %36, %34
  store i32 %37, i32* %35, align 4
  %38 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i8*
  %39 = bitcast %"class.Kalmar::__index_leaf.2"* %33 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %38, i8* %39, i64 8, i32 4, i1 false)
  %40 = bitcast %"struct.Kalmar::index_impl.4"* %19 to i8*
  %41 = getelementptr inbounds i8* %40, i64 16
  %42 = bitcast i8* %41 to %"class.Kalmar::__index_leaf.5"*
  %43 = load i32* %12, align 4
  store %"class.Kalmar::__index_leaf.5"* %42, %"class.Kalmar::__index_leaf.5"** %3, align 8
  store i32 %43, i32* %4, align 4
  %44 = load %"class.Kalmar::__index_leaf.5"** %3
  %45 = load i32* %4, align 4
  %46 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %44, i32 0, i32 0
  %47 = load i32* %46, align 4
  %48 = sub nsw i32 %47, %45
  store i32 %48, i32* %46, align 4
  %49 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i8*
  %50 = bitcast %"class.Kalmar::__index_leaf.5"* %44 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %49, i8* %50, i64 8, i32 4, i1 false)
  %51 = bitcast %"class.Kalmar::__index_leaf"* %13 to i64*
  %52 = load i64* %51, align 1
  %53 = bitcast %"class.Kalmar::__index_leaf.2"* %14 to i64*
  %54 = load i64* %53, align 1
  %55 = bitcast %"class.Kalmar::__index_leaf.5"* %15 to i64*
  %56 = load i64* %55, align 1
  %57 = bitcast %"class.Kalmar::__index_leaf"* %5 to i64*
  store i64 %52, i64* %57, align 1
  %58 = bitcast %"class.Kalmar::__index_leaf.2"* %6 to i64*
  store i64 %54, i64* %58, align 1
  %59 = bitcast %"class.Kalmar::__index_leaf.5"* %7 to i64*
  store i64 %56, i64* %59, align 1
  store %"struct.Kalmar::index_impl.4"* %19, %"struct.Kalmar::index_impl.4"** %8, align 8
  %60 = load %"struct.Kalmar::index_impl.4"** %8
  ret %"class.Kalmar::index.3"* %17
}

; Function Attrs: alwaysinline uwtable
define weak_odr void @_ZN6Kalmar5indexILi3EEmmEi(%"class.Kalmar::index.3"* noalias sret %agg.result, %"class.Kalmar::index.3"* %this, i32) #0 align 2 {
  %2 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %3 = alloca i32, align 4
  %4 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %"class.Kalmar::__index_leaf", align 8
  %7 = alloca %"class.Kalmar::__index_leaf.2", align 8
  %8 = alloca %"class.Kalmar::__index_leaf.5", align 8
  %9 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %10 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %11 = alloca i32, align 4
  %12 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %13 = alloca i32, align 4
  %14 = alloca %"class.Kalmar::__index_leaf", align 4
  %15 = alloca %"class.Kalmar::__index_leaf.2", align 4
  %16 = alloca %"class.Kalmar::__index_leaf.5", align 4
  %17 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %18 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %19 = alloca %"class.Kalmar::__index_leaf.2"*, align 8
  %20 = alloca i32, align 4
  %21 = alloca %"class.Kalmar::__index_leaf.5"*, align 8
  %22 = alloca i32, align 4
  %23 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %24 = alloca i32, align 4
  %25 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %26 = alloca i32, align 4
  %27 = alloca i32, align 4
  %28 = alloca i32, align 4
  %29 = alloca %"class.Kalmar::__index_leaf"*, align 8
  %30 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %31 = alloca %"struct.Kalmar::index_impl.4"*, align 8
  %32 = alloca %"class.Kalmar::index.3"*, align 8
  %33 = alloca %"class.Kalmar::index.3"*, align 8
  %34 = alloca %"class.Kalmar::index.3"*, align 8
  %35 = alloca %"class.Kalmar::index.3"*, align 8
  %36 = alloca %"class.Kalmar::index.3"*, align 8
  %37 = alloca i32, align 4
  store %"class.Kalmar::index.3"* %this, %"class.Kalmar::index.3"** %36, align 8
  store i32 %0, i32* %37, align 4
  %38 = load %"class.Kalmar::index.3"** %36
  store %"class.Kalmar::index.3"* %agg.result, %"class.Kalmar::index.3"** %34, align 8
  store %"class.Kalmar::index.3"* %38, %"class.Kalmar::index.3"** %35, align 8
  %39 = load %"class.Kalmar::index.3"** %34
  %40 = load %"class.Kalmar::index.3"** %35
  store %"class.Kalmar::index.3"* %39, %"class.Kalmar::index.3"** %32, align 8
  store %"class.Kalmar::index.3"* %40, %"class.Kalmar::index.3"** %33, align 8
  %41 = load %"class.Kalmar::index.3"** %32
  %42 = getelementptr inbounds %"class.Kalmar::index.3"* %41, i32 0, i32 0
  %43 = load %"class.Kalmar::index.3"** %33, align 8
  %44 = getelementptr inbounds %"class.Kalmar::index.3"* %43, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %42, %"struct.Kalmar::index_impl.4"** %30, align 8
  store %"struct.Kalmar::index_impl.4"* %44, %"struct.Kalmar::index_impl.4"** %31, align 8
  %45 = load %"struct.Kalmar::index_impl.4"** %30
  %46 = load %"struct.Kalmar::index_impl.4"** %31, align 8
  %47 = bitcast %"struct.Kalmar::index_impl.4"* %46 to %"class.Kalmar::__index_leaf"*
  store %"class.Kalmar::__index_leaf"* %47, %"class.Kalmar::__index_leaf"** %29, align 8
  %48 = load %"class.Kalmar::__index_leaf"** %29
  %49 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %48, i32 0, i32 0
  %50 = load i32* %49
  %51 = load %"struct.Kalmar::index_impl.4"** %31, align 8
  %52 = bitcast %"struct.Kalmar::index_impl.4"* %51 to i8*
  %53 = getelementptr inbounds i8* %52, i64 8
  %54 = bitcast i8* %53 to %"class.Kalmar::__index_leaf.2"*
  store %"class.Kalmar::__index_leaf.2"* %54, %"class.Kalmar::__index_leaf.2"** %17, align 8
  %55 = load %"class.Kalmar::__index_leaf.2"** %17
  %56 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %55, i32 0, i32 0
  %57 = load i32* %56
  %58 = load %"struct.Kalmar::index_impl.4"** %31, align 8
  %59 = bitcast %"struct.Kalmar::index_impl.4"* %58 to i8*
  %60 = getelementptr inbounds i8* %59, i64 16
  %61 = bitcast i8* %60 to %"class.Kalmar::__index_leaf.5"*
  store %"class.Kalmar::__index_leaf.5"* %61, %"class.Kalmar::__index_leaf.5"** %18, align 8
  %62 = load %"class.Kalmar::__index_leaf.5"** %18
  %63 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %62, i32 0, i32 0
  %64 = load i32* %63
  store %"struct.Kalmar::index_impl.4"* %45, %"struct.Kalmar::index_impl.4"** %25, align 8
  store i32 %50, i32* %26, align 4
  store i32 %57, i32* %27, align 4
  store i32 %64, i32* %28, align 4
  %65 = load %"struct.Kalmar::index_impl.4"** %25
  %66 = bitcast %"struct.Kalmar::index_impl.4"* %65 to %"class.Kalmar::__index_leaf"*
  %67 = load i32* %26, align 4
  store %"class.Kalmar::__index_leaf"* %66, %"class.Kalmar::__index_leaf"** %23, align 8
  store i32 %67, i32* %24, align 4
  %68 = load %"class.Kalmar::__index_leaf"** %23
  %69 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %68, i32 0, i32 0
  %70 = load i32* %24, align 4
  store i32 %70, i32* %69, align 4
  %71 = bitcast %"struct.Kalmar::index_impl.4"* %65 to i8*
  %72 = getelementptr inbounds i8* %71, i64 8
  %73 = bitcast i8* %72 to %"class.Kalmar::__index_leaf.2"*
  %74 = load i32* %27, align 4
  store %"class.Kalmar::__index_leaf.2"* %73, %"class.Kalmar::__index_leaf.2"** %19, align 8
  store i32 %74, i32* %20, align 4
  %75 = load %"class.Kalmar::__index_leaf.2"** %19
  %76 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %75, i32 0, i32 0
  %77 = load i32* %20, align 4
  store i32 %77, i32* %76, align 4
  %78 = bitcast %"struct.Kalmar::index_impl.4"* %65 to i8*
  %79 = getelementptr inbounds i8* %78, i64 16
  %80 = bitcast i8* %79 to %"class.Kalmar::__index_leaf.5"*
  %81 = load i32* %28, align 4
  store %"class.Kalmar::__index_leaf.5"* %80, %"class.Kalmar::__index_leaf.5"** %21, align 8
  store i32 %81, i32* %22, align 4
  %82 = load %"class.Kalmar::__index_leaf.5"** %21
  %83 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %82, i32 0, i32 0
  %84 = load i32* %22, align 4
  store i32 %84, i32* %83, align 4
  %85 = getelementptr inbounds %"class.Kalmar::index.3"* %38, i32 0, i32 0
  store %"struct.Kalmar::index_impl.4"* %85, %"struct.Kalmar::index_impl.4"** %12, align 8
  store i32 1, i32* %13, align 4
  %86 = load %"struct.Kalmar::index_impl.4"** %12
  %87 = bitcast %"struct.Kalmar::index_impl.4"* %86 to %"class.Kalmar::__index_leaf"*
  %88 = load i32* %13, align 4
  store %"class.Kalmar::__index_leaf"* %87, %"class.Kalmar::__index_leaf"** %10, align 8
  store i32 %88, i32* %11, align 4
  %89 = load %"class.Kalmar::__index_leaf"** %10
  %90 = load i32* %11, align 4
  %91 = getelementptr inbounds %"class.Kalmar::__index_leaf"* %89, i32 0, i32 0
  %92 = load i32* %91, align 4
  %93 = sub nsw i32 %92, %90
  store i32 %93, i32* %91, align 4
  %94 = bitcast %"class.Kalmar::__index_leaf"* %14 to i8*
  %95 = bitcast %"class.Kalmar::__index_leaf"* %89 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %94, i8* %95, i64 8, i32 4, i1 false)
  %96 = bitcast %"struct.Kalmar::index_impl.4"* %86 to i8*
  %97 = getelementptr inbounds i8* %96, i64 8
  %98 = bitcast i8* %97 to %"class.Kalmar::__index_leaf.2"*
  %99 = load i32* %13, align 4
  store %"class.Kalmar::__index_leaf.2"* %98, %"class.Kalmar::__index_leaf.2"** %2, align 8
  store i32 %99, i32* %3, align 4
  %100 = load %"class.Kalmar::__index_leaf.2"** %2
  %101 = load i32* %3, align 4
  %102 = getelementptr inbounds %"class.Kalmar::__index_leaf.2"* %100, i32 0, i32 0
  %103 = load i32* %102, align 4
  %104 = sub nsw i32 %103, %101
  store i32 %104, i32* %102, align 4
  %105 = bitcast %"class.Kalmar::__index_leaf.2"* %15 to i8*
  %106 = bitcast %"class.Kalmar::__index_leaf.2"* %100 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %105, i8* %106, i64 8, i32 4, i1 false)
  %107 = bitcast %"struct.Kalmar::index_impl.4"* %86 to i8*
  %108 = getelementptr inbounds i8* %107, i64 16
  %109 = bitcast i8* %108 to %"class.Kalmar::__index_leaf.5"*
  %110 = load i32* %13, align 4
  store %"class.Kalmar::__index_leaf.5"* %109, %"class.Kalmar::__index_leaf.5"** %4, align 8
  store i32 %110, i32* %5, align 4
  %111 = load %"class.Kalmar::__index_leaf.5"** %4
  %112 = load i32* %5, align 4
  %113 = getelementptr inbounds %"class.Kalmar::__index_leaf.5"* %111, i32 0, i32 0
  %114 = load i32* %113, align 4
  %115 = sub nsw i32 %114, %112
  store i32 %115, i32* %113, align 4
  %116 = bitcast %"class.Kalmar::__index_leaf.5"* %16 to i8*
  %117 = bitcast %"class.Kalmar::__index_leaf.5"* %111 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %116, i8* %117, i64 8, i32 4, i1 false)
  %118 = bitcast %"class.Kalmar::__index_leaf"* %14 to i64*
  %119 = load i64* %118, align 1
  %120 = bitcast %"class.Kalmar::__index_leaf.2"* %15 to i64*
  %121 = load i64* %120, align 1
  %122 = bitcast %"class.Kalmar::__index_leaf.5"* %16 to i64*
  %123 = load i64* %122, align 1
  %124 = bitcast %"class.Kalmar::__index_leaf"* %6 to i64*
  store i64 %119, i64* %124, align 1
  %125 = bitcast %"class.Kalmar::__index_leaf.2"* %7 to i64*
  store i64 %121, i64* %125, align 1
  %126 = bitcast %"class.Kalmar::__index_leaf.5"* %8 to i64*
  store i64 %123, i64* %126, align 1
  store %"struct.Kalmar::index_impl.4"* %86, %"struct.Kalmar::index_impl.4"** %9, align 8
  %127 = load %"struct.Kalmar::index_impl.4"** %9
  ret void
}

define internal void @__cxx_global_var_init() section ".text.startup" {
  %1 = alloca %"class.std::__1::set"*, align 8
  %2 = alloca %"struct.std::__1::less"*, align 8
  %3 = alloca %"struct.std::__1::less", align 1
  store %"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE, %"class.std::__1::set"** %1, align 8
  store %"struct.std::__1::less"* %3, %"struct.std::__1::less"** %2, align 8
  %4 = load %"class.std::__1::set"** %1
  %5 = getelementptr inbounds %"class.std::__1::set"* %4, i32 0, i32 0
  %6 = load %"struct.std::__1::less"** %2, align 8
  call void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEEC2ERKS8_(%"class.std::__1::__tree"* %5, %"struct.std::__1::less"* dereferenceable(1) %6) #2
  %7 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::__1::set"*)* @_ZNSt3__13setINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEED2Ev to void (i8*)*), i8* bitcast (%"class.std::__1::set"* @_ZN6KalmarL20__mcw_cxxamp_kernelsE to i8*), i8* @__dso_handle) #2
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_ZNSt3__13setINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEED2Ev(%"class.std::__1::set"* %this) unnamed_addr #1 align 2 {
  %1 = alloca %"class.std::__1::set"*, align 8
  store %"class.std::__1::set"* %this, %"class.std::__1::set"** %1, align 8
  %2 = load %"class.std::__1::set"** %1
  %3 = getelementptr inbounds %"class.std::__1::set"* %2, i32 0, i32 0
  call void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEED2Ev(%"class.std::__1::__tree"* %3) #2
  ret void
}

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: nounwind uwtable
define i32 @__hsail_atomic_fetch_add_int(i32* %dest, i32 %val) #3 {
  %1 = alloca i32*, align 8
  %2 = alloca i32, align 4
  store i32* %dest, i32** %1, align 8
  store i32 %val, i32* %2, align 4
  %3 = load i32** %1, align 8
  %4 = load i32* %2, align 4
  %5 = atomicrmw add i32* %3, i32 %4 seq_cst
  ret i32 %5
}

; Function Attrs: nounwind uwtable
define i32 @__hsail_atomic_fetch_add_unsigned(i32* %dest, i32 %val) #3 {
  %1 = alloca i32*, align 8
  %2 = alloca i32, align 4
  store i32* %dest, i32** %1, align 8
  store i32 %val, i32* %2, align 4
  %3 = load i32** %1, align 8
  %4 = load i32* %2, align 4
  %5 = atomicrmw add i32* %3, i32 %4 seq_cst
  ret i32 %5
}

; Function Attrs: nounwind uwtable
define i64 @__hsail_atomic_fetch_add_int64(i64* %dest, i64 %val) #3 {
  %1 = alloca i64*, align 8
  %2 = alloca i64, align 8
  store i64* %dest, i64** %1, align 8
  store i64 %val, i64* %2, align 8
  %3 = load i64** %1, align 8
  %4 = load i64* %2, align 8
  %5 = atomicrmw add i64* %3, i64 %4 seq_cst
  ret i64 %5
}

; Function Attrs: nounwind uwtable
define i32 @__hsail_atomic_exchange_int(i32* %dest, i32 %val) #3 {
  %1 = alloca i32*, align 8
  %2 = alloca i32, align 4
  store i32* %dest, i32** %1, align 8
  store i32 %val, i32* %2, align 4
  %3 = load i32** %1, align 8
  %4 = load i32* %2, align 4
  %5 = atomicrmw xchg i32* %3, i32 %4 seq_cst
  ret i32 %5
}

; Function Attrs: nounwind uwtable
define i32 @__hsail_atomic_exchange_unsigned(i32* %dest, i32 %val) #3 {
  %1 = alloca i32*, align 8
  %2 = alloca i32, align 4
  store i32* %dest, i32** %1, align 8
  store i32 %val, i32* %2, align 4
  %3 = load i32** %1, align 8
  %4 = load i32* %2, align 4
  %5 = atomicrmw xchg i32* %3, i32 %4 seq_cst
  ret i32 %5
}

; Function Attrs: nounwind uwtable
define i64 @__hsail_atomic_exchange_int64(i64* %dest, i64 %val) #3 {
  %1 = alloca i64*, align 8
  %2 = alloca i64, align 8
  store i64* %dest, i64** %1, align 8
  store i64 %val, i64* %2, align 8
  %3 = load i64** %1, align 8
  %4 = load i64* %2, align 8
  %5 = atomicrmw xchg i64* %3, i64 %4 seq_cst
  ret i64 %5
}

; Function Attrs: nounwind uwtable
define i32 @__hsail_atomic_compare_exchange_int(i32* %dest, i32 %compare, i32 %val) #3 {
  %1 = alloca i32*, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32* %dest, i32** %1, align 8
  store i32 %compare, i32* %2, align 4
  store i32 %val, i32* %3, align 4
  %4 = load i32** %1, align 8
  %5 = load i32* %2, align 4
  %6 = load i32* %3, align 4
  %7 = cmpxchg i32* %4, i32 %5, i32 %6 seq_cst seq_cst
  %8 = extractvalue { i32, i1 } %7, 0
  ret i32 %8
}

; Function Attrs: nounwind uwtable
define i32 @__hsail_atomic_compare_exchange_unsigned(i32* %dest, i32 %compare, i32 %val) #3 {
  %1 = alloca i32*, align 8
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32* %dest, i32** %1, align 8
  store i32 %compare, i32* %2, align 4
  store i32 %val, i32* %3, align 4
  %4 = load i32** %1, align 8
  %5 = load i32* %2, align 4
  %6 = load i32* %3, align 4
  %7 = cmpxchg i32* %4, i32 %5, i32 %6 seq_cst seq_cst
  %8 = extractvalue { i32, i1 } %7, 0
  ret i32 %8
}

; Function Attrs: nounwind uwtable
define i64 @__hsail_atomic_compare_exchange_int64(i64* %dest, i64 %compare, i64 %val) #3 {
  %1 = alloca i64*, align 8
  %2 = alloca i64, align 8
  %3 = alloca i64, align 8
  store i64* %dest, i64** %1, align 8
  store i64 %compare, i64* %2, align 8
  store i64 %val, i64* %3, align 8
  %4 = load i64** %1, align 8
  %5 = load i64* %2, align 8
  %6 = load i64* %3, align 8
  %7 = cmpxchg i64* %4, i64 %5, i64 %6 seq_cst seq_cst
  %8 = extractvalue { i64, i1 } %7, 0
  ret i64 %8
}

; Function Attrs: noinline nounwind uwtable
define void @_Z6kernel16grid_launch_parmP3fooPi(%struct.grid_launch_parm* byval align 8 %lp, %struct.foo* %f, i32* %x) #4 {
  %1 = alloca %struct.foo*, align 8
  %2 = alloca i32*, align 8
  %idx = alloca i32, align 4
  store %struct.foo* %f, %struct.foo** %1, align 8
  store i32* %x, i32** %2, align 8
  %3 = getelementptr inbounds %struct.grid_launch_parm* %lp, i32 0, i32 3
  %4 = getelementptr inbounds %struct.uint3* %3, i32 0, i32 0
  %5 = load i32* %4, align 4
  %6 = getelementptr inbounds %struct.grid_launch_parm* %lp, i32 0, i32 1
  %7 = getelementptr inbounds %struct.uint3* %6, i32 0, i32 0
  %8 = load i32* %7, align 4
  %9 = getelementptr inbounds %struct.grid_launch_parm* %lp, i32 0, i32 2
  %10 = getelementptr inbounds %struct.uint3* %9, i32 0, i32 0
  %11 = load i32* %10, align 4
  %12 = mul nsw i32 %8, %11
  %13 = add nsw i32 %5, %12
  store i32 %13, i32* %idx, align 4
  %14 = load i32* %idx, align 4
  %15 = load i32* %idx, align 4
  %16 = sext i32 %15 to i64
  %17 = load i32** %2, align 8
  %18 = getelementptr inbounds i32* %17, i64 %16
  store i32 %14, i32* %18, align 4
  ret void
}

; Function Attrs: uwtable
define i32 @main() #5 {
  %1 = alloca i32, align 4
  %f = alloca %struct.foo*, align 8
  %cf = alloca %"class.hc::completion_future", align 8
  %x = alloca i32*, align 8
  %grid = alloca %struct.uint3, align 4
  %2 = alloca i8*
  %3 = alloca i32
  %4 = alloca { i64, i32 }
  %block = alloca %struct.uint3, align 4
  %5 = alloca { i64, i32 }
  %lp = alloca %struct.grid_launch_parm, align 4
  %6 = alloca %struct.uint3, align 4
  %7 = alloca %struct.uint3, align 4
  %8 = alloca { i64, i32 }
  %9 = alloca { i64, i32 }
  %10 = alloca %struct.grid_launch_parm, align 8
  %ret = alloca i8, align 1
  %i = alloca i32, align 4
  %11 = alloca i32
  store i32 0, i32* %1
  %12 = call noalias i8* @malloc(i64 8) #2
  %13 = bitcast i8* %12 to %struct.foo*
  store %struct.foo* %13, %struct.foo** %f, align 8
  call void @_ZN2hc17completion_futureC2Ev(%"class.hc::completion_future"* %cf)
  %14 = load %struct.foo** %f, align 8
  %15 = getelementptr inbounds %struct.foo* %14, i32 0, i32 0
  store %"class.hc::completion_future"* %cf, %"class.hc::completion_future"** %15, align 8
  %16 = call noalias i8* @malloc(i64 64) #2
  %17 = bitcast i8* %16 to i32*
  store i32* %17, i32** %x, align 8
  %18 = load i32** %x, align 8
  %19 = bitcast i32* %18 to i8*
  call void @llvm.memset.p0i8.i64(i8* %19, i8 0, i64 64, i32 4, i1 false)
  %20 = invoke { i64, i32 } (i32, ...)* @_Z9dim3_evaliz(i32 1, i32 1, i64 0)
          to label %21 unwind label %72

; <label>:21                                      ; preds = %0
  store { i64, i32 } %20, { i64, i32 }* %4
  %22 = bitcast { i64, i32 }* %4 to i8*
  %23 = bitcast %struct.uint3* %grid to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %23, i8* %22, i64 12, i32 1, i1 false)
  %24 = invoke { i64, i32 } (i32, ...)* @_Z9dim3_evaliz(i32 16, i32 1, i64 0)
          to label %25 unwind label %72

; <label>:25                                      ; preds = %21
  store { i64, i32 } %24, { i64, i32 }* %5
  %26 = bitcast { i64, i32 }* %5 to i8*
  %27 = bitcast %struct.uint3* %block to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %27, i8* %26, i64 12, i32 1, i1 false)
  %28 = bitcast %struct.uint3* %6 to i8*
  %29 = bitcast %struct.uint3* %grid to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %28, i8* %29, i64 12, i32 4, i1 false)
  %30 = bitcast %struct.uint3* %7 to i8*
  %31 = bitcast %struct.uint3* %block to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %30, i8* %31, i64 12, i32 4, i1 false)
  %32 = bitcast { i64, i32 }* %8 to i8*
  %33 = bitcast %struct.uint3* %6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %32, i8* %33, i64 12, i32 0, i1 false)
  %34 = getelementptr { i64, i32 }* %8, i32 0, i32 0
  %35 = load i64* %34, align 1
  %36 = getelementptr { i64, i32 }* %8, i32 0, i32 1
  %37 = load i32* %36, align 1
  %38 = bitcast { i64, i32 }* %9 to i8*
  %39 = bitcast %struct.uint3* %7 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %38, i8* %39, i64 12, i32 0, i1 false)
  %40 = getelementptr { i64, i32 }* %9, i32 0, i32 0
  %41 = load i64* %40, align 1
  %42 = getelementptr { i64, i32 }* %9, i32 0, i32 1
  %43 = load i32* %42, align 1
  invoke void @_Z21hipCreateLaunchParam25uint3S_(%struct.grid_launch_parm* sret %lp, i64 %35, i32 %37, i64 %41, i32 %43)
          to label %44 unwind label %72

; <label>:44                                      ; preds = %25
  %45 = bitcast %struct.grid_launch_parm* %10 to i8*
  %46 = bitcast %struct.grid_launch_parm* %lp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %45, i8* %46, i64 52, i32 4, i1 false)
  %47 = load %struct.foo** %f, align 8
  %48 = load i32** %x, align 8
  ; CHECK-NOT: invoke void @_Z6kernel16grid_launch_parmP3fooPi(%struct.grid_launch_parm* byval align 8 %10, %struct.foo* %47, i32* %48)
  ; CHECK: invoke void @__hcLaunchKernel__Z6kernel16grid_launch_parmP3fooPi(%struct.grid_launch_parm* byval align 8 %10, %struct.foo* %47, i32* %48)
  invoke void @_Z6kernel16grid_launch_parmP3fooPi(%struct.grid_launch_parm* byval align 8 %10, %struct.foo* %47, i32* %48)
          to label %49 unwind label %72

; <label>:49                                      ; preds = %44
  store i8 1, i8* %ret, align 1
  store i32 0, i32* %i, align 4
  br label %50

; <label>:50                                      ; preds = %69, %49
  %51 = load i32* %i, align 4
  %52 = icmp slt i32 %51, 16
  br i1 %52, label %53, label %76

; <label>:53                                      ; preds = %50
  %54 = load i32* %i, align 4
  %55 = sext i32 %54 to i64
  %56 = load i32** %x, align 8
  %57 = getelementptr inbounds i32* %56, i64 %55
  %58 = load i32* %57, align 4
  %59 = load i32* %i, align 4
  %60 = sub nsw i32 %58, %59
  %61 = icmp eq i32 %60, 0
  %62 = zext i1 %61 to i32
  %63 = load i8* %ret, align 1
  %64 = trunc i8 %63 to i1
  %65 = zext i1 %64 to i32
  %66 = and i32 %65, %62
  %67 = icmp ne i32 %66, 0
  %68 = zext i1 %67 to i8
  store i8 %68, i8* %ret, align 1
  br label %69

; <label>:69                                      ; preds = %53
  %70 = load i32* %i, align 4
  %71 = add nsw i32 %70, 1
  store i32 %71, i32* %i, align 4
  br label %50

; <label>:72                                      ; preds = %44, %25, %21, %0
  %73 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          cleanup
  %74 = extractvalue { i8*, i32 } %73, 0
  store i8* %74, i8** %2
  %75 = extractvalue { i8*, i32 } %73, 1
  store i32 %75, i32* %3
  call void @_ZN2hc17completion_futureD2Ev(%"class.hc::completion_future"* %cf) #2
  br label %82

; <label>:76                                      ; preds = %50
  %77 = load i8* %ret, align 1
  %78 = trunc i8 %77 to i1
  %79 = xor i1 %78, true
  %80 = zext i1 %79 to i32
  store i32 %80, i32* %1
  store i32 1, i32* %11
  call void @_ZN2hc17completion_futureD2Ev(%"class.hc::completion_future"* %cf) #2
  %81 = load i32* %1
  ret i32 %81

; <label>:82                                      ; preds = %72
  %83 = load i8** %2
  %84 = load i32* %3
  %85 = insertvalue { i8*, i32 } undef, i8* %83, 0
  %86 = insertvalue { i8*, i32 } %85, i32 %84, 1
  resume { i8*, i32 } %86
}

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) #6

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN2hc17completion_futureC2Ev(%"class.hc::completion_future"* %this) unnamed_addr #3 align 2 {
  %1 = alloca %"class.std::__1::shared_ptr"*, align 8
  %2 = alloca i8*, align 8
  %3 = alloca %"class.std::__1::shared_future"*, align 8
  %4 = alloca %"class.hc::completion_future"*, align 8
  store %"class.hc::completion_future"* %this, %"class.hc::completion_future"** %4, align 8
  %5 = load %"class.hc::completion_future"** %4
  %6 = getelementptr inbounds %"class.hc::completion_future"* %5, i32 0, i32 0
  store %"class.std::__1::shared_future"* %6, %"class.std::__1::shared_future"** %3, align 8
  %7 = load %"class.std::__1::shared_future"** %3
  %8 = getelementptr inbounds %"class.std::__1::shared_future"* %7, i32 0, i32 0
  store %"class.std::__1::__assoc_sub_state"* null, %"class.std::__1::__assoc_sub_state"** %8, align 8
  %9 = getelementptr inbounds %"class.hc::completion_future"* %5, i32 0, i32 1
  store %"class.std::__1::thread"* null, %"class.std::__1::thread"** %9, align 8
  %10 = getelementptr inbounds %"class.hc::completion_future"* %5, i32 0, i32 2
  store %"class.std::__1::shared_ptr"* %10, %"class.std::__1::shared_ptr"** %1, align 8
  store i8* null, i8** %2, align 8
  %11 = load %"class.std::__1::shared_ptr"** %1
  %12 = getelementptr inbounds %"class.std::__1::shared_ptr"* %11, i32 0, i32 0
  store %"class.Kalmar::KalmarAsyncOp"* null, %"class.Kalmar::KalmarAsyncOp"** %12, align 8
  %13 = getelementptr inbounds %"class.std::__1::shared_ptr"* %11, i32 0, i32 1
  store %"class.std::__1::__shared_weak_count"* null, %"class.std::__1::__shared_weak_count"** %13, align 8
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #2

; Function Attrs: inlinehint uwtable
define linkonce_odr { i64, i32 } @_Z9dim3_evaliz(i32 %x, ...) #7 {
  %1 = alloca %struct.uint3, align 4
  %2 = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  %args = alloca [1 x %struct.__va_list_tag], align 16
  %3 = alloca { i64, i32 }
  %4 = alloca { i64, i32 }
  %5 = alloca { i64, i32 }
  %6 = alloca { i64, i32 }
  store i32 %x, i32* %2, align 4
  %7 = getelementptr inbounds [1 x %struct.__va_list_tag]* %args, i32 0, i32 0
  %8 = bitcast %struct.__va_list_tag* %7 to i8*
  call void @llvm.va_start(i8* %8)
  %9 = getelementptr inbounds [1 x %struct.__va_list_tag]* %args, i32 0, i32 0
  %10 = getelementptr inbounds %struct.__va_list_tag* %9, i32 0, i32 0
  %11 = load i32* %10
  %12 = icmp ule i32 %11, 40
  br i1 %12, label %13, label %19

; <label>:13                                      ; preds = %0
  %14 = getelementptr inbounds %struct.__va_list_tag* %9, i32 0, i32 3
  %15 = load i8** %14
  %16 = getelementptr i8* %15, i32 %11
  %17 = bitcast i8* %16 to i32*
  %18 = add i32 %11, 8
  store i32 %18, i32* %10
  br label %24

; <label>:19                                      ; preds = %0
  %20 = getelementptr inbounds %struct.__va_list_tag* %9, i32 0, i32 2
  %21 = load i8** %20
  %22 = bitcast i8* %21 to i32*
  %23 = getelementptr i8* %21, i32 8
  store i8* %23, i8** %20
  br label %24

; <label>:24                                      ; preds = %19, %13
  %25 = phi i32* [ %17, %13 ], [ %22, %19 ]
  %26 = load i32* %25
  store i32 %26, i32* %y, align 4
  %27 = icmp ne i32 %26, 0
  br i1 %27, label %35, label %28

; <label>:28                                      ; preds = %24
  %29 = getelementptr inbounds [1 x %struct.__va_list_tag]* %args, i32 0, i32 0
  %30 = bitcast %struct.__va_list_tag* %29 to i8*
  call void @llvm.va_end(i8* %30)
  %31 = load i32* %2, align 4
  %32 = call { i64, i32 } @_Z9dim3_initiii(i32 %31, i32 1, i32 1)
  store { i64, i32 } %32, { i64, i32 }* %3
  %33 = bitcast { i64, i32 }* %3 to i8*
  %34 = bitcast %struct.uint3* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %34, i8* %33, i64 12, i32 1, i1 false)
  br label %72

; <label>:35                                      ; preds = %24
  %36 = getelementptr inbounds [1 x %struct.__va_list_tag]* %args, i32 0, i32 0
  %37 = getelementptr inbounds %struct.__va_list_tag* %36, i32 0, i32 0
  %38 = load i32* %37
  %39 = icmp ule i32 %38, 40
  br i1 %39, label %40, label %46

; <label>:40                                      ; preds = %35
  %41 = getelementptr inbounds %struct.__va_list_tag* %36, i32 0, i32 3
  %42 = load i8** %41
  %43 = getelementptr i8* %42, i32 %38
  %44 = bitcast i8* %43 to i32*
  %45 = add i32 %38, 8
  store i32 %45, i32* %37
  br label %51

; <label>:46                                      ; preds = %35
  %47 = getelementptr inbounds %struct.__va_list_tag* %36, i32 0, i32 2
  %48 = load i8** %47
  %49 = bitcast i8* %48 to i32*
  %50 = getelementptr i8* %48, i32 8
  store i8* %50, i8** %47
  br label %51

; <label>:51                                      ; preds = %46, %40
  %52 = phi i32* [ %44, %40 ], [ %49, %46 ]
  %53 = load i32* %52
  store i32 %53, i32* %z, align 4
  %54 = icmp ne i32 %53, 0
  br i1 %54, label %63, label %55

; <label>:55                                      ; preds = %51
  %56 = getelementptr inbounds [1 x %struct.__va_list_tag]* %args, i32 0, i32 0
  %57 = bitcast %struct.__va_list_tag* %56 to i8*
  call void @llvm.va_end(i8* %57)
  %58 = load i32* %2, align 4
  %59 = load i32* %y, align 4
  %60 = call { i64, i32 } @_Z9dim3_initiii(i32 %58, i32 %59, i32 1)
  store { i64, i32 } %60, { i64, i32 }* %4
  %61 = bitcast { i64, i32 }* %4 to i8*
  %62 = bitcast %struct.uint3* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %62, i8* %61, i64 12, i32 1, i1 false)
  br label %72

; <label>:63                                      ; preds = %51
  %64 = getelementptr inbounds [1 x %struct.__va_list_tag]* %args, i32 0, i32 0
  %65 = bitcast %struct.__va_list_tag* %64 to i8*
  call void @llvm.va_end(i8* %65)
  %66 = load i32* %2, align 4
  %67 = load i32* %y, align 4
  %68 = load i32* %z, align 4
  %69 = call { i64, i32 } @_Z9dim3_initiii(i32 %66, i32 %67, i32 %68)
  store { i64, i32 } %69, { i64, i32 }* %5
  %70 = bitcast { i64, i32 }* %5 to i8*
  %71 = bitcast %struct.uint3* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %71, i8* %70, i64 12, i32 1, i1 false)
  br label %72

; <label>:72                                      ; preds = %63, %55, %28
  %73 = bitcast { i64, i32 }* %6 to i8*
  %74 = bitcast %struct.uint3* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %73, i8* %74, i64 12, i32 1, i1 false)
  %75 = load { i64, i32 }* %6
  ret { i64, i32 } %75
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr void @_Z21hipCreateLaunchParam25uint3S_(%struct.grid_launch_parm* noalias sret %agg.result, i64 %gridDim.coerce0, i32 %gridDim.coerce1, i64 %groupDim.coerce0, i32 %groupDim.coerce1) #1 {
  %gridDim = alloca %struct.uint3, align 8
  %1 = alloca { i64, i32 }, align 8
  %groupDim = alloca %struct.uint3, align 8
  %2 = alloca { i64, i32 }, align 8
  %3 = getelementptr { i64, i32 }* %1, i32 0, i32 0
  store i64 %gridDim.coerce0, i64* %3
  %4 = getelementptr { i64, i32 }* %1, i32 0, i32 1
  store i32 %gridDim.coerce1, i32* %4
  %5 = bitcast %struct.uint3* %gridDim to i8*
  %6 = bitcast { i64, i32 }* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %5, i8* %6, i64 12, i32 8, i1 false)
  %7 = getelementptr { i64, i32 }* %2, i32 0, i32 0
  store i64 %groupDim.coerce0, i64* %7
  %8 = getelementptr { i64, i32 }* %2, i32 0, i32 1
  store i32 %groupDim.coerce1, i32* %8
  %9 = bitcast %struct.uint3* %groupDim to i8*
  %10 = bitcast { i64, i32 }* %2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %9, i8* %10, i64 12, i32 8, i1 false)
  %11 = getelementptr inbounds %struct.uint3* %gridDim, i32 0, i32 0
  %12 = load i32* %11, align 4
  %13 = getelementptr inbounds %struct.grid_launch_parm* %agg.result, i32 0, i32 0
  %14 = getelementptr inbounds %struct.uint3* %13, i32 0, i32 0
  store i32 %12, i32* %14, align 4
  %15 = getelementptr inbounds %struct.uint3* %gridDim, i32 0, i32 1
  %16 = load i32* %15, align 4
  %17 = getelementptr inbounds %struct.grid_launch_parm* %agg.result, i32 0, i32 0
  %18 = getelementptr inbounds %struct.uint3* %17, i32 0, i32 1
  store i32 %16, i32* %18, align 4
  %19 = getelementptr inbounds %struct.uint3* %gridDim, i32 0, i32 2
  %20 = load i32* %19, align 4
  %21 = getelementptr inbounds %struct.grid_launch_parm* %agg.result, i32 0, i32 0
  %22 = getelementptr inbounds %struct.uint3* %21, i32 0, i32 2
  store i32 %20, i32* %22, align 4
  %23 = getelementptr inbounds %struct.uint3* %groupDim, i32 0, i32 0
  %24 = load i32* %23, align 4
  %25 = getelementptr inbounds %struct.grid_launch_parm* %agg.result, i32 0, i32 1
  %26 = getelementptr inbounds %struct.uint3* %25, i32 0, i32 0
  store i32 %24, i32* %26, align 4
  %27 = getelementptr inbounds %struct.uint3* %groupDim, i32 0, i32 1
  %28 = load i32* %27, align 4
  %29 = getelementptr inbounds %struct.grid_launch_parm* %agg.result, i32 0, i32 1
  %30 = getelementptr inbounds %struct.uint3* %29, i32 0, i32 1
  store i32 %28, i32* %30, align 4
  %31 = getelementptr inbounds %struct.uint3* %groupDim, i32 0, i32 2
  %32 = load i32* %31, align 4
  %33 = getelementptr inbounds %struct.grid_launch_parm* %agg.result, i32 0, i32 1
  %34 = getelementptr inbounds %struct.uint3* %33, i32 0, i32 2
  store i32 %32, i32* %34, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZN2hc17completion_futureD2Ev(%"class.hc::completion_future"* %this) unnamed_addr #3 align 2 {
  %1 = alloca %"class.std::__1::__shared_weak_count"**, align 8
  %2 = alloca %"class.std::__1::__shared_weak_count"**, align 8
  %3 = alloca %"class.std::__1::__shared_weak_count"**, align 8
  %4 = alloca %"class.std::__1::__shared_weak_count"**, align 8
  %5 = alloca %"class.std::__1::__shared_weak_count"**, align 8
  %__t.i1.i.i = alloca %"class.std::__1::__shared_weak_count"*, align 8
  %6 = alloca %"class.Kalmar::KalmarAsyncOp"**, align 8
  %7 = alloca %"class.Kalmar::KalmarAsyncOp"**, align 8
  %8 = alloca %"class.Kalmar::KalmarAsyncOp"**, align 8
  %9 = alloca %"class.Kalmar::KalmarAsyncOp"**, align 8
  %10 = alloca %"class.Kalmar::KalmarAsyncOp"**, align 8
  %__t.i.i.i = alloca %"class.Kalmar::KalmarAsyncOp"*, align 8
  %11 = alloca %"class.std::__1::shared_ptr"*, align 8
  %12 = alloca %"class.std::__1::shared_ptr"*, align 8
  %13 = alloca %"class.std::__1::shared_ptr"*, align 8
  %14 = alloca %"class.std::__1::shared_ptr"*, align 8
  %15 = alloca %"class.std::__1::shared_ptr"*, align 8
  %16 = alloca %"class.std::__1::shared_ptr"*, align 8
  %17 = alloca %"class.std::__1::shared_ptr"*, align 8
  %18 = alloca %"class.std::__1::shared_ptr", align 8
  %19 = alloca %"class.std::__1::shared_ptr"*, align 8
  %20 = alloca i8*, align 8
  %21 = alloca %"class.std::__1::shared_ptr"*, align 8
  %22 = alloca %"class.std::__1::shared_ptr"*, align 8
  %23 = alloca %"class.std::__1::shared_ptr"*, align 8
  %24 = alloca i8*, align 8
  %25 = alloca %"class.hc::completion_future"*, align 8
  %26 = alloca i8*
  %27 = alloca i32
  %28 = alloca %"class.std::__1::shared_ptr", align 8
  store %"class.hc::completion_future"* %this, %"class.hc::completion_future"** %25, align 8
  %29 = load %"class.hc::completion_future"** %25
  %30 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 1
  %31 = load %"class.std::__1::thread"** %30, align 8
  %32 = icmp ne %"class.std::__1::thread"* %31, null
  br i1 %32, label %33, label %43

; <label>:33                                      ; preds = %0
  %34 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 1
  %35 = load %"class.std::__1::thread"** %34, align 8
  invoke void @_ZNSt3__16thread4joinEv(%"class.std::__1::thread"* %35)
          to label %36 unwind label %37

; <label>:36                                      ; preds = %33
  br label %43

; <label>:37                                      ; preds = %33
  %38 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  %39 = extractvalue { i8*, i32 } %38, 0
  store i8* %39, i8** %26
  %40 = extractvalue { i8*, i32 } %38, 1
  store i32 %40, i32* %27
  %41 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 2
  call void @_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev(%"class.std::__1::shared_ptr"* %41) #2
  %42 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 0
  call void @_ZNSt3__113shared_futureIvED1Ev(%"class.std::__1::shared_future"* %42) #2
  br label %109

; <label>:43                                      ; preds = %36, %0
  %44 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 1
  %45 = load %"class.std::__1::thread"** %44, align 8
  %46 = icmp eq %"class.std::__1::thread"* %45, null
  br i1 %46, label %49, label %47

; <label>:47                                      ; preds = %43
  call void @_ZNSt3__16threadD1Ev(%"class.std::__1::thread"* %45) #2
  %48 = bitcast %"class.std::__1::thread"* %45 to i8*
  call void @_ZdlPv(i8* %48) #11
  br label %49

; <label>:49                                      ; preds = %47, %43
  %50 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 1
  store %"class.std::__1::thread"* null, %"class.std::__1::thread"** %50, align 8
  %51 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 2
  store %"class.std::__1::shared_ptr"* %51, %"class.std::__1::shared_ptr"** %23, align 8
  store i8* null, i8** %24, align 8
  %52 = load %"class.std::__1::shared_ptr"** %23, align 8
  store %"class.std::__1::shared_ptr"* %52, %"class.std::__1::shared_ptr"** %22, align 8
  %53 = load %"class.std::__1::shared_ptr"** %22
  store %"class.std::__1::shared_ptr"* %53, %"class.std::__1::shared_ptr"** %21, align 8
  %54 = load %"class.std::__1::shared_ptr"** %21
  %55 = getelementptr inbounds %"class.std::__1::shared_ptr"* %54, i32 0, i32 0
  %56 = load %"class.Kalmar::KalmarAsyncOp"** %55, align 8
  %57 = icmp ne %"class.Kalmar::KalmarAsyncOp"* %56, null
  br i1 %57, label %58, label %106

; <label>:58                                      ; preds = %49
  %59 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 2
  store %"class.std::__1::shared_ptr"* %28, %"class.std::__1::shared_ptr"** %19, align 8
  store i8* null, i8** %20, align 8
  %60 = load %"class.std::__1::shared_ptr"** %19
  %61 = getelementptr inbounds %"class.std::__1::shared_ptr"* %60, i32 0, i32 0
  store %"class.Kalmar::KalmarAsyncOp"* null, %"class.Kalmar::KalmarAsyncOp"** %61, align 8
  %62 = getelementptr inbounds %"class.std::__1::shared_ptr"* %60, i32 0, i32 1
  store %"class.std::__1::__shared_weak_count"* null, %"class.std::__1::__shared_weak_count"** %62, align 8
  store %"class.std::__1::shared_ptr"* %59, %"class.std::__1::shared_ptr"** %16, align 8
  store %"class.std::__1::shared_ptr"* %28, %"class.std::__1::shared_ptr"** %17, align 8
  %63 = load %"class.std::__1::shared_ptr"** %16
  %64 = load %"class.std::__1::shared_ptr"** %17, align 8
  store %"class.std::__1::shared_ptr"* %64, %"class.std::__1::shared_ptr"** %15, align 8
  %65 = load %"class.std::__1::shared_ptr"** %15, align 8
  store %"class.std::__1::shared_ptr"* %18, %"class.std::__1::shared_ptr"** %13, align 8
  store %"class.std::__1::shared_ptr"* %65, %"class.std::__1::shared_ptr"** %14, align 8
  %66 = load %"class.std::__1::shared_ptr"** %13
  %67 = getelementptr inbounds %"class.std::__1::shared_ptr"* %66, i32 0, i32 0
  %68 = load %"class.std::__1::shared_ptr"** %14, align 8
  %69 = getelementptr inbounds %"class.std::__1::shared_ptr"* %68, i32 0, i32 0
  %70 = load %"class.Kalmar::KalmarAsyncOp"** %69, align 8
  store %"class.Kalmar::KalmarAsyncOp"* %70, %"class.Kalmar::KalmarAsyncOp"** %67, align 8
  %71 = getelementptr inbounds %"class.std::__1::shared_ptr"* %66, i32 0, i32 1
  %72 = load %"class.std::__1::shared_ptr"** %14, align 8
  %73 = getelementptr inbounds %"class.std::__1::shared_ptr"* %72, i32 0, i32 1
  %74 = load %"class.std::__1::__shared_weak_count"** %73, align 8
  store %"class.std::__1::__shared_weak_count"* %74, %"class.std::__1::__shared_weak_count"** %71, align 8
  %75 = load %"class.std::__1::shared_ptr"** %14, align 8
  %76 = getelementptr inbounds %"class.std::__1::shared_ptr"* %75, i32 0, i32 0
  store %"class.Kalmar::KalmarAsyncOp"* null, %"class.Kalmar::KalmarAsyncOp"** %76, align 8
  %77 = load %"class.std::__1::shared_ptr"** %14, align 8
  %78 = getelementptr inbounds %"class.std::__1::shared_ptr"* %77, i32 0, i32 1
  store %"class.std::__1::__shared_weak_count"* null, %"class.std::__1::__shared_weak_count"** %78, align 8
  store %"class.std::__1::shared_ptr"* %18, %"class.std::__1::shared_ptr"** %11, align 8
  store %"class.std::__1::shared_ptr"* %63, %"class.std::__1::shared_ptr"** %12, align 8
  %79 = load %"class.std::__1::shared_ptr"** %11
  %80 = getelementptr inbounds %"class.std::__1::shared_ptr"* %79, i32 0, i32 0
  %81 = load %"class.std::__1::shared_ptr"** %12, align 8
  %82 = getelementptr inbounds %"class.std::__1::shared_ptr"* %81, i32 0, i32 0
  store %"class.Kalmar::KalmarAsyncOp"** %80, %"class.Kalmar::KalmarAsyncOp"*** %9, align 8
  store %"class.Kalmar::KalmarAsyncOp"** %82, %"class.Kalmar::KalmarAsyncOp"*** %10, align 8
  %83 = load %"class.Kalmar::KalmarAsyncOp"*** %9, align 8
  store %"class.Kalmar::KalmarAsyncOp"** %83, %"class.Kalmar::KalmarAsyncOp"*** %8, align 8
  %84 = load %"class.Kalmar::KalmarAsyncOp"*** %8, align 8
  %85 = load %"class.Kalmar::KalmarAsyncOp"** %84
  store %"class.Kalmar::KalmarAsyncOp"* %85, %"class.Kalmar::KalmarAsyncOp"** %__t.i.i.i, align 8
  %86 = load %"class.Kalmar::KalmarAsyncOp"*** %10, align 8
  store %"class.Kalmar::KalmarAsyncOp"** %86, %"class.Kalmar::KalmarAsyncOp"*** %6, align 8
  %87 = load %"class.Kalmar::KalmarAsyncOp"*** %6, align 8
  %88 = load %"class.Kalmar::KalmarAsyncOp"** %87
  %89 = load %"class.Kalmar::KalmarAsyncOp"*** %9, align 8
  store %"class.Kalmar::KalmarAsyncOp"* %88, %"class.Kalmar::KalmarAsyncOp"** %89, align 8
  store %"class.Kalmar::KalmarAsyncOp"** %__t.i.i.i, %"class.Kalmar::KalmarAsyncOp"*** %7, align 8
  %90 = load %"class.Kalmar::KalmarAsyncOp"*** %7, align 8
  %91 = load %"class.Kalmar::KalmarAsyncOp"** %90
  %92 = load %"class.Kalmar::KalmarAsyncOp"*** %10, align 8
  store %"class.Kalmar::KalmarAsyncOp"* %91, %"class.Kalmar::KalmarAsyncOp"** %92, align 8
  %93 = getelementptr inbounds %"class.std::__1::shared_ptr"* %79, i32 0, i32 1
  %94 = load %"class.std::__1::shared_ptr"** %12, align 8
  %95 = getelementptr inbounds %"class.std::__1::shared_ptr"* %94, i32 0, i32 1
  store %"class.std::__1::__shared_weak_count"** %93, %"class.std::__1::__shared_weak_count"*** %4, align 8
  store %"class.std::__1::__shared_weak_count"** %95, %"class.std::__1::__shared_weak_count"*** %5, align 8
  %96 = load %"class.std::__1::__shared_weak_count"*** %4, align 8
  store %"class.std::__1::__shared_weak_count"** %96, %"class.std::__1::__shared_weak_count"*** %3, align 8
  %97 = load %"class.std::__1::__shared_weak_count"*** %3, align 8
  %98 = load %"class.std::__1::__shared_weak_count"** %97
  store %"class.std::__1::__shared_weak_count"* %98, %"class.std::__1::__shared_weak_count"** %__t.i1.i.i, align 8
  %99 = load %"class.std::__1::__shared_weak_count"*** %5, align 8
  store %"class.std::__1::__shared_weak_count"** %99, %"class.std::__1::__shared_weak_count"*** %1, align 8
  %100 = load %"class.std::__1::__shared_weak_count"*** %1, align 8
  %101 = load %"class.std::__1::__shared_weak_count"** %100
  %102 = load %"class.std::__1::__shared_weak_count"*** %4, align 8
  store %"class.std::__1::__shared_weak_count"* %101, %"class.std::__1::__shared_weak_count"** %102, align 8
  store %"class.std::__1::__shared_weak_count"** %__t.i1.i.i, %"class.std::__1::__shared_weak_count"*** %2, align 8
  %103 = load %"class.std::__1::__shared_weak_count"*** %2, align 8
  %104 = load %"class.std::__1::__shared_weak_count"** %103
  %105 = load %"class.std::__1::__shared_weak_count"*** %5, align 8
  store %"class.std::__1::__shared_weak_count"* %104, %"class.std::__1::__shared_weak_count"** %105, align 8
  call void @_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev(%"class.std::__1::shared_ptr"* %18) #2
  call void @_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev(%"class.std::__1::shared_ptr"* %28) #2
  br label %106

; <label>:106                                     ; preds = %58, %49
  %107 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 2
  call void @_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev(%"class.std::__1::shared_ptr"* %107) #2
  %108 = getelementptr inbounds %"class.hc::completion_future"* %29, i32 0, i32 0
  call void @_ZNSt3__113shared_futureIvED1Ev(%"class.std::__1::shared_future"* %108) #2
  ret void

; <label>:109                                     ; preds = %37
  %110 = load i8** %26
  call void @__clang_call_terminate(i8* %110) #12
  unreachable
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEEC2ERKS8_(%"class.std::__1::__tree"* %this, %"struct.std::__1::less"* dereferenceable(1) %__comp) unnamed_addr #3 align 2 {
  %1 = alloca %"class.std::__1::__tree_end_node"*, align 8
  %2 = alloca %"class.std::__1::__tree_end_node"*, align 8
  %3 = alloca %"class.std::__1::__libcpp_compressed_pair_imp"*, align 8
  %4 = alloca %"class.std::__1::__compressed_pair"*, align 8
  %5 = alloca %"class.std::__1::__tree"*, align 8
  %6 = alloca %"class.std::__1::__tree"*, align 8
  %7 = alloca %"struct.std::__1::less"*, align 8
  %8 = alloca i64*, align 8
  %9 = alloca %"struct.std::__1::less"*, align 8
  %__t2.i.i = alloca %"struct.std::__1::less", align 1
  %10 = alloca %"class.std::__1::__libcpp_compressed_pair_imp.7"*, align 8
  %11 = alloca i64, align 8
  %12 = alloca i64*, align 8
  %__t2.i = alloca %"struct.std::__1::less", align 1
  %13 = alloca %"class.std::__1::__compressed_pair.6"*, align 8
  %14 = alloca i64, align 8
  %15 = alloca %"struct.std::__1::less", align 1
  %16 = alloca %"class.std::__1::__tree_end_node"*, align 8
  %17 = alloca %"class.std::__1::allocator"*, align 8
  %18 = alloca %"class.std::__1::__libcpp_compressed_pair_imp"*, align 8
  %19 = alloca %"class.std::__1::__compressed_pair"*, align 8
  %20 = alloca %"class.std::__1::__tree"*, align 8
  %21 = alloca %"struct.std::__1::less"*, align 8
  %22 = alloca %"struct.std::__1::less", align 1
  store %"class.std::__1::__tree"* %this, %"class.std::__1::__tree"** %20, align 8
  store %"struct.std::__1::less"* %__comp, %"struct.std::__1::less"** %21, align 8
  %23 = load %"class.std::__1::__tree"** %20
  %24 = getelementptr inbounds %"class.std::__1::__tree"* %23, i32 0, i32 1
  store %"class.std::__1::__compressed_pair"* %24, %"class.std::__1::__compressed_pair"** %19, align 8
  %25 = load %"class.std::__1::__compressed_pair"** %19
  %26 = bitcast %"class.std::__1::__compressed_pair"* %25 to %"class.std::__1::__libcpp_compressed_pair_imp"*
  store %"class.std::__1::__libcpp_compressed_pair_imp"* %26, %"class.std::__1::__libcpp_compressed_pair_imp"** %18, align 8
  %27 = load %"class.std::__1::__libcpp_compressed_pair_imp"** %18
  %28 = bitcast %"class.std::__1::__libcpp_compressed_pair_imp"* %27 to %"class.std::__1::allocator"*
  store %"class.std::__1::allocator"* %28, %"class.std::__1::allocator"** %17, align 8
  %29 = load %"class.std::__1::allocator"** %17
  %30 = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp"* %27, i32 0, i32 0
  store %"class.std::__1::__tree_end_node"* %30, %"class.std::__1::__tree_end_node"** %16, align 8
  %31 = load %"class.std::__1::__tree_end_node"** %16
  %32 = getelementptr inbounds %"class.std::__1::__tree_end_node"* %31, i32 0, i32 0
  store %"class.std::__1::__tree_node_base"* null, %"class.std::__1::__tree_node_base"** %32, align 8
  br label %33

; <label>:33                                      ; preds = %0
  %34 = getelementptr inbounds %"class.std::__1::__tree"* %23, i32 0, i32 2
  %35 = load %"struct.std::__1::less"** %21, align 8
  store %"class.std::__1::__compressed_pair.6"* %34, %"class.std::__1::__compressed_pair.6"** %13, align 8
  store i64 0, i64* %14, align 8
  %36 = load %"class.std::__1::__compressed_pair.6"** %13
  %37 = bitcast %"class.std::__1::__compressed_pair.6"* %36 to %"class.std::__1::__libcpp_compressed_pair_imp.7"*
  store i64* %14, i64** %12, align 8
  %38 = load i64** %12, align 8
  %39 = load i64* %38
  store %"struct.std::__1::less"* %__t2.i, %"struct.std::__1::less"** %7, align 8
  %40 = load %"struct.std::__1::less"** %7, align 8
  store %"class.std::__1::__libcpp_compressed_pair_imp.7"* %37, %"class.std::__1::__libcpp_compressed_pair_imp.7"** %10, align 8
  store i64 %39, i64* %11, align 8
  %41 = load %"class.std::__1::__libcpp_compressed_pair_imp.7"** %10
  %42 = bitcast %"class.std::__1::__libcpp_compressed_pair_imp.7"* %41 to %"struct.std::__1::less"*
  store %"struct.std::__1::less"* %__t2.i.i, %"struct.std::__1::less"** %9, align 8
  %43 = load %"struct.std::__1::less"** %9, align 8
  %44 = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp.7"* %41, i32 0, i32 0
  store i64* %11, i64** %8, align 8
  %45 = load i64** %8, align 8
  %46 = load i64* %45
  store i64 %46, i64* %44, align 8
  br label %47

; <label>:47                                      ; preds = %33
  store %"class.std::__1::__tree"* %23, %"class.std::__1::__tree"** %5, align 8
  %48 = load %"class.std::__1::__tree"** %5
  %49 = getelementptr inbounds %"class.std::__1::__tree"* %48, i32 0, i32 1
  store %"class.std::__1::__compressed_pair"* %49, %"class.std::__1::__compressed_pair"** %4, align 8
  %50 = load %"class.std::__1::__compressed_pair"** %4
  %51 = bitcast %"class.std::__1::__compressed_pair"* %50 to %"class.std::__1::__libcpp_compressed_pair_imp"*
  store %"class.std::__1::__libcpp_compressed_pair_imp"* %51, %"class.std::__1::__libcpp_compressed_pair_imp"** %3, align 8
  %52 = load %"class.std::__1::__libcpp_compressed_pair_imp"** %3
  %53 = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp"* %52, i32 0, i32 0
  store %"class.std::__1::__tree_end_node"* %53, %"class.std::__1::__tree_end_node"** %2, align 8
  %54 = load %"class.std::__1::__tree_end_node"** %2, align 8
  store %"class.std::__1::__tree_end_node"* %54, %"class.std::__1::__tree_end_node"** %1, align 8
  %55 = load %"class.std::__1::__tree_end_node"** %1, align 8
  %56 = bitcast %"class.std::__1::__tree_end_node"* %55 to i8*
  %57 = bitcast i8* %56 to %"class.std::__1::__tree_end_node"*
  %58 = bitcast %"class.std::__1::__tree_end_node"* %57 to %"class.std::__1::__tree_node"*
  store %"class.std::__1::__tree"* %23, %"class.std::__1::__tree"** %6, align 8
  %59 = load %"class.std::__1::__tree"** %6
  %60 = getelementptr inbounds %"class.std::__1::__tree"* %59, i32 0, i32 0
  store %"class.std::__1::__tree_node"* %58, %"class.std::__1::__tree_node"** %60
  ret void
                                                  ; No predecessors!
  %62 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  %63 = extractvalue { i8*, i32 } %62, 0
  call void @__clang_call_terminate(i8* %63) #12
  unreachable
}

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) #8 {
  %2 = call i8* @__cxa_begin_catch(i8* %0) #2
  call void @_ZSt9terminatev() #12
  unreachable
}

declare i8* @__cxa_begin_catch(i8*)

declare void @_ZSt9terminatev()

declare void @_ZNSt3__16thread4joinEv(%"class.std::__1::thread"*) #9

; Function Attrs: nounwind
declare void @_ZNSt3__16threadD1Ev(%"class.std::__1::thread"*) #6

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) #10

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZNSt3__110shared_ptrIN6Kalmar13KalmarAsyncOpEED2Ev(%"class.std::__1::shared_ptr"* %this) unnamed_addr #3 align 2 {
  %1 = alloca %"class.std::__1::shared_ptr"*, align 8
  store %"class.std::__1::shared_ptr"* %this, %"class.std::__1::shared_ptr"** %1, align 8
  %2 = load %"class.std::__1::shared_ptr"** %1
  %3 = getelementptr inbounds %"class.std::__1::shared_ptr"* %2, i32 0, i32 1
  %4 = load %"class.std::__1::__shared_weak_count"** %3, align 8
  %5 = icmp ne %"class.std::__1::__shared_weak_count"* %4, null
  br i1 %5, label %6, label %9

; <label>:6                                       ; preds = %0
  %7 = getelementptr inbounds %"class.std::__1::shared_ptr"* %2, i32 0, i32 1
  %8 = load %"class.std::__1::__shared_weak_count"** %7, align 8
  call void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"* %8) #2
  br label %9

; <label>:9                                       ; preds = %6, %0
  ret void
}

; Function Attrs: nounwind
declare void @_ZNSt3__113shared_futureIvED1Ev(%"class.std::__1::shared_future"*) #6

; Function Attrs: nounwind
declare void @_ZNSt3__119__shared_weak_count16__release_sharedEv(%"class.std::__1::__shared_weak_count"*) #6

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) #2

; Function Attrs: nounwind
declare void @llvm.va_end(i8*) #2

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr { i64, i32 } @_Z9dim3_initiii(i32 %x, i32 %y, i32 %z) #1 {
  %1 = alloca %struct.uint3, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca { i64, i32 }
  store i32 %x, i32* %2, align 4
  store i32 %y, i32* %3, align 4
  store i32 %z, i32* %4, align 4
  %6 = load i32* %2, align 4
  %7 = getelementptr inbounds %struct.uint3* %1, i32 0, i32 0
  store i32 %6, i32* %7, align 4
  %8 = load i32* %3, align 4
  %9 = getelementptr inbounds %struct.uint3* %1, i32 0, i32 1
  store i32 %8, i32* %9, align 4
  %10 = load i32* %4, align 4
  %11 = getelementptr inbounds %struct.uint3* %1, i32 0, i32 2
  store i32 %10, i32* %11, align 4
  %12 = bitcast { i64, i32 }* %5 to i8*
  %13 = bitcast %struct.uint3* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %12, i8* %13, i64 12, i32 1, i1 false)
  %14 = load { i64, i32 }* %5
  ret { i64, i32 } %14
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEED2Ev(%"class.std::__1::__tree"* %this) unnamed_addr #3 align 2 {
  %1 = alloca %"class.std::__1::__tree_end_node"*, align 8
  %2 = alloca %"class.std::__1::__tree_end_node"*, align 8
  %3 = alloca %"class.std::__1::__libcpp_compressed_pair_imp"*, align 8
  %4 = alloca %"class.std::__1::__compressed_pair"*, align 8
  %5 = alloca %"class.std::__1::__tree"*, align 8
  %6 = alloca %"class.std::__1::__tree"*, align 8
  %7 = alloca %"class.std::__1::__tree"*, align 8
  store %"class.std::__1::__tree"* %this, %"class.std::__1::__tree"** %7, align 8
  %8 = load %"class.std::__1::__tree"** %7
  store %"class.std::__1::__tree"* %8, %"class.std::__1::__tree"** %6, align 8
  %9 = load %"class.std::__1::__tree"** %6
  store %"class.std::__1::__tree"* %9, %"class.std::__1::__tree"** %5, align 8
  %10 = load %"class.std::__1::__tree"** %5
  %11 = getelementptr inbounds %"class.std::__1::__tree"* %10, i32 0, i32 1
  store %"class.std::__1::__compressed_pair"* %11, %"class.std::__1::__compressed_pair"** %4, align 8
  %12 = load %"class.std::__1::__compressed_pair"** %4
  %13 = bitcast %"class.std::__1::__compressed_pair"* %12 to %"class.std::__1::__libcpp_compressed_pair_imp"*
  store %"class.std::__1::__libcpp_compressed_pair_imp"* %13, %"class.std::__1::__libcpp_compressed_pair_imp"** %3, align 8
  %14 = load %"class.std::__1::__libcpp_compressed_pair_imp"** %3
  %15 = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp"* %14, i32 0, i32 0
  store %"class.std::__1::__tree_end_node"* %15, %"class.std::__1::__tree_end_node"** %2, align 8
  %16 = load %"class.std::__1::__tree_end_node"** %2, align 8
  store %"class.std::__1::__tree_end_node"* %16, %"class.std::__1::__tree_end_node"** %1, align 8
  %17 = load %"class.std::__1::__tree_end_node"** %1, align 8
  %18 = bitcast %"class.std::__1::__tree_end_node"* %17 to i8*
  %19 = bitcast i8* %18 to %"class.std::__1::__tree_end_node"*
  %20 = bitcast %"class.std::__1::__tree_end_node"* %19 to %"class.std::__1::__tree_node"*
  %21 = bitcast %"class.std::__1::__tree_node"* %20 to %"class.std::__1::__tree_end_node"*
  %22 = getelementptr inbounds %"class.std::__1::__tree_end_node"* %21, i32 0, i32 0
  %23 = load %"class.std::__1::__tree_node_base"** %22, align 8
  %24 = bitcast %"class.std::__1::__tree_node_base"* %23 to %"class.std::__1::__tree_node"*
  call void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEE7destroyEPNS_11__tree_nodeIS6_PvEE(%"class.std::__1::__tree"* %8, %"class.std::__1::__tree_node"* %24) #2
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEE7destroyEPNS_11__tree_nodeIS6_PvEE(%"class.std::__1::__tree"* %this, %"class.std::__1::__tree_node"* %__nd) #3 align 2 {
  %1 = alloca %"class.std::__1::allocator"*, align 8
  %2 = alloca %"class.std::__1::basic_string"*, align 8
  %3 = alloca %"class.std::__1::allocator"*, align 8
  %4 = alloca %"class.std::__1::basic_string"*, align 8
  %5 = alloca %"struct.std::__1::integral_constant", align 1
  %6 = alloca %"struct.std::__1::__has_destroy", align 1
  %7 = alloca %"class.std::__1::basic_string"*, align 8
  %8 = alloca %"class.std::__1::__libcpp_compressed_pair_imp"*, align 8
  %9 = alloca %"class.std::__1::__compressed_pair"*, align 8
  %10 = alloca %"class.std::__1::__tree"*, align 8
  %11 = alloca %"class.std::__1::allocator"*, align 8
  %12 = alloca %"class.std::__1::__tree_node"*, align 8
  %13 = alloca i64, align 8
  %14 = alloca %"class.std::__1::allocator"*, align 8
  %15 = alloca %"class.std::__1::__tree_node"*, align 8
  %16 = alloca i64, align 8
  %17 = alloca %"class.std::__1::__tree"*, align 8
  %18 = alloca %"class.std::__1::__tree_node"*, align 8
  %__na = alloca %"class.std::__1::allocator"*, align 8
  store %"class.std::__1::__tree"* %this, %"class.std::__1::__tree"** %17, align 8
  store %"class.std::__1::__tree_node"* %__nd, %"class.std::__1::__tree_node"** %18, align 8
  %19 = load %"class.std::__1::__tree"** %17
  %20 = load %"class.std::__1::__tree_node"** %18, align 8
  %21 = icmp ne %"class.std::__1::__tree_node"* %20, null
  br i1 %21, label %22, label %58

; <label>:22                                      ; preds = %0
  %23 = load %"class.std::__1::__tree_node"** %18, align 8
  %24 = bitcast %"class.std::__1::__tree_node"* %23 to %"class.std::__1::__tree_end_node"*
  %25 = getelementptr inbounds %"class.std::__1::__tree_end_node"* %24, i32 0, i32 0
  %26 = load %"class.std::__1::__tree_node_base"** %25, align 8
  %27 = bitcast %"class.std::__1::__tree_node_base"* %26 to %"class.std::__1::__tree_node"*
  call void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEE7destroyEPNS_11__tree_nodeIS6_PvEE(%"class.std::__1::__tree"* %19, %"class.std::__1::__tree_node"* %27) #2
  %28 = load %"class.std::__1::__tree_node"** %18, align 8
  %29 = bitcast %"class.std::__1::__tree_node"* %28 to %"class.std::__1::__tree_node_base"*
  %30 = getelementptr inbounds %"class.std::__1::__tree_node_base"* %29, i32 0, i32 1
  %31 = load %"class.std::__1::__tree_node_base"** %30, align 8
  %32 = bitcast %"class.std::__1::__tree_node_base"* %31 to %"class.std::__1::__tree_node"*
  call void @_ZNSt3__16__treeINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS_4lessIS6_EENS4_IS6_EEE7destroyEPNS_11__tree_nodeIS6_PvEE(%"class.std::__1::__tree"* %19, %"class.std::__1::__tree_node"* %32) #2
  store %"class.std::__1::__tree"* %19, %"class.std::__1::__tree"** %10, align 8
  %33 = load %"class.std::__1::__tree"** %10
  %34 = getelementptr inbounds %"class.std::__1::__tree"* %33, i32 0, i32 1
  store %"class.std::__1::__compressed_pair"* %34, %"class.std::__1::__compressed_pair"** %9, align 8
  %35 = load %"class.std::__1::__compressed_pair"** %9
  %36 = bitcast %"class.std::__1::__compressed_pair"* %35 to %"class.std::__1::__libcpp_compressed_pair_imp"*
  store %"class.std::__1::__libcpp_compressed_pair_imp"* %36, %"class.std::__1::__libcpp_compressed_pair_imp"** %8, align 8
  %37 = load %"class.std::__1::__libcpp_compressed_pair_imp"** %8
  %38 = bitcast %"class.std::__1::__libcpp_compressed_pair_imp"* %37 to %"class.std::__1::allocator"*
  store %"class.std::__1::allocator"* %38, %"class.std::__1::allocator"** %__na, align 8
  %39 = load %"class.std::__1::allocator"** %__na, align 8
  %40 = load %"class.std::__1::__tree_node"** %18, align 8
  %41 = getelementptr inbounds %"class.std::__1::__tree_node"* %40, i32 0, i32 1
  store %"class.std::__1::basic_string"* %41, %"class.std::__1::basic_string"** %7, align 8
  %42 = load %"class.std::__1::basic_string"** %7, align 8
  %43 = bitcast %"class.std::__1::basic_string"* %42 to i8*
  %44 = bitcast i8* %43 to %"class.std::__1::basic_string"*
  store %"class.std::__1::allocator"* %39, %"class.std::__1::allocator"** %3, align 8
  store %"class.std::__1::basic_string"* %44, %"class.std::__1::basic_string"** %4, align 8
  %45 = bitcast %"struct.std::__1::__has_destroy"* %6 to %"struct.std::__1::integral_constant"*
  %46 = load %"class.std::__1::allocator"** %3, align 8
  %47 = load %"class.std::__1::basic_string"** %4, align 8
  store %"class.std::__1::allocator"* %46, %"class.std::__1::allocator"** %1, align 8
  store %"class.std::__1::basic_string"* %47, %"class.std::__1::basic_string"** %2, align 8
  %48 = load %"class.std::__1::basic_string"** %2, align 8
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED2Ev(%"class.std::__1::basic_string"* %48) #2
  br label %49

; <label>:49                                      ; preds = %22
  %50 = load %"class.std::__1::allocator"** %__na, align 8
  %51 = load %"class.std::__1::__tree_node"** %18, align 8
  store %"class.std::__1::allocator"* %50, %"class.std::__1::allocator"** %14, align 8
  store %"class.std::__1::__tree_node"* %51, %"class.std::__1::__tree_node"** %15, align 8
  store i64 1, i64* %16, align 8
  %52 = load %"class.std::__1::allocator"** %14, align 8
  %53 = load %"class.std::__1::__tree_node"** %15, align 8
  %54 = load i64* %16, align 8
  store %"class.std::__1::allocator"* %52, %"class.std::__1::allocator"** %11, align 8
  store %"class.std::__1::__tree_node"* %53, %"class.std::__1::__tree_node"** %12, align 8
  store i64 %54, i64* %13, align 8
  %55 = load %"class.std::__1::allocator"** %11
  %56 = load %"class.std::__1::__tree_node"** %12, align 8
  %57 = bitcast %"class.std::__1::__tree_node"* %56 to i8*
  call void @_ZdlPv(i8* %57) #2
  br label %58

; <label>:58                                      ; preds = %49, %0
  ret void
                                                  ; No predecessors!
  %60 = landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
          catch i8* null
  %61 = extractvalue { i8*, i32 } %60, 0
  call void @__clang_call_terminate(i8* %61) #12
  unreachable
}

; Function Attrs: nounwind uwtable
define linkonce_odr void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED2Ev(%"class.std::__1::basic_string"* %this) unnamed_addr #3 align 2 {
  %1 = alloca %"class.std::__1::__libcpp_compressed_pair_imp.9"*, align 8
  %2 = alloca %"class.std::__1::__compressed_pair.8"*, align 8
  %3 = alloca %"class.std::__1::basic_string"*, align 8
  %4 = alloca %"class.std::__1::__libcpp_compressed_pair_imp.9"*, align 8
  %5 = alloca %"class.std::__1::__compressed_pair.8"*, align 8
  %6 = alloca %"class.std::__1::basic_string"*, align 8
  %7 = alloca %"class.std::__1::__libcpp_compressed_pair_imp.9"*, align 8
  %8 = alloca %"class.std::__1::__compressed_pair.8"*, align 8
  %9 = alloca %"class.std::__1::basic_string"*, align 8
  %10 = alloca %"class.std::__1::allocator.10"*, align 8
  %11 = alloca i8*, align 8
  %12 = alloca i64, align 8
  %13 = alloca %"class.std::__1::allocator.10"*, align 8
  %14 = alloca i8*, align 8
  %15 = alloca i64, align 8
  %16 = alloca %"class.std::__1::__libcpp_compressed_pair_imp.9"*, align 8
  %17 = alloca %"class.std::__1::__compressed_pair.8"*, align 8
  %18 = alloca %"class.std::__1::basic_string"*, align 8
  %19 = alloca %"class.std::__1::basic_string"*, align 8
  store %"class.std::__1::basic_string"* %this, %"class.std::__1::basic_string"** %19, align 8
  %20 = load %"class.std::__1::basic_string"** %19
  store %"class.std::__1::basic_string"* %20, %"class.std::__1::basic_string"** %18, align 8
  %21 = load %"class.std::__1::basic_string"** %18
  %22 = getelementptr inbounds %"class.std::__1::basic_string"* %21, i32 0, i32 0
  store %"class.std::__1::__compressed_pair.8"* %22, %"class.std::__1::__compressed_pair.8"** %17, align 8
  %23 = load %"class.std::__1::__compressed_pair.8"** %17
  %24 = bitcast %"class.std::__1::__compressed_pair.8"* %23 to %"class.std::__1::__libcpp_compressed_pair_imp.9"*
  store %"class.std::__1::__libcpp_compressed_pair_imp.9"* %24, %"class.std::__1::__libcpp_compressed_pair_imp.9"** %16, align 8
  %25 = load %"class.std::__1::__libcpp_compressed_pair_imp.9"** %16
  %26 = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp.9"* %25, i32 0, i32 0
  %27 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep"* %26, i32 0, i32 0
  %28 = bitcast %union.anon* %27 to %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short"*
  %29 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__short"* %28, i32 0, i32 0
  %30 = bitcast %union.anon.12* %29 to i8*
  %31 = load i8* %30, align 1
  %32 = zext i8 %31 to i32
  %33 = and i32 %32, 1
  %34 = icmp ne i32 %33, 0
  br i1 %34, label %35, label %68

; <label>:35                                      ; preds = %0
  store %"class.std::__1::basic_string"* %20, %"class.std::__1::basic_string"** %9, align 8
  %36 = load %"class.std::__1::basic_string"** %9
  %37 = getelementptr inbounds %"class.std::__1::basic_string"* %36, i32 0, i32 0
  store %"class.std::__1::__compressed_pair.8"* %37, %"class.std::__1::__compressed_pair.8"** %8, align 8
  %38 = load %"class.std::__1::__compressed_pair.8"** %8
  %39 = bitcast %"class.std::__1::__compressed_pair.8"* %38 to %"class.std::__1::__libcpp_compressed_pair_imp.9"*
  store %"class.std::__1::__libcpp_compressed_pair_imp.9"* %39, %"class.std::__1::__libcpp_compressed_pair_imp.9"** %7, align 8
  %40 = load %"class.std::__1::__libcpp_compressed_pair_imp.9"** %7
  %41 = bitcast %"class.std::__1::__libcpp_compressed_pair_imp.9"* %40 to %"class.std::__1::allocator.10"*
  store %"class.std::__1::basic_string"* %20, %"class.std::__1::basic_string"** %3, align 8
  %42 = load %"class.std::__1::basic_string"** %3
  %43 = getelementptr inbounds %"class.std::__1::basic_string"* %42, i32 0, i32 0
  store %"class.std::__1::__compressed_pair.8"* %43, %"class.std::__1::__compressed_pair.8"** %2, align 8
  %44 = load %"class.std::__1::__compressed_pair.8"** %2
  %45 = bitcast %"class.std::__1::__compressed_pair.8"* %44 to %"class.std::__1::__libcpp_compressed_pair_imp.9"*
  store %"class.std::__1::__libcpp_compressed_pair_imp.9"* %45, %"class.std::__1::__libcpp_compressed_pair_imp.9"** %1, align 8
  %46 = load %"class.std::__1::__libcpp_compressed_pair_imp.9"** %1
  %47 = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp.9"* %46, i32 0, i32 0
  %48 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep"* %47, i32 0, i32 0
  %49 = bitcast %union.anon* %48 to %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long"*
  %50 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long"* %49, i32 0, i32 2
  %51 = load i8** %50, align 8
  store %"class.std::__1::basic_string"* %20, %"class.std::__1::basic_string"** %6, align 8
  %52 = load %"class.std::__1::basic_string"** %6
  %53 = getelementptr inbounds %"class.std::__1::basic_string"* %52, i32 0, i32 0
  store %"class.std::__1::__compressed_pair.8"* %53, %"class.std::__1::__compressed_pair.8"** %5, align 8
  %54 = load %"class.std::__1::__compressed_pair.8"** %5
  %55 = bitcast %"class.std::__1::__compressed_pair.8"* %54 to %"class.std::__1::__libcpp_compressed_pair_imp.9"*
  store %"class.std::__1::__libcpp_compressed_pair_imp.9"* %55, %"class.std::__1::__libcpp_compressed_pair_imp.9"** %4, align 8
  %56 = load %"class.std::__1::__libcpp_compressed_pair_imp.9"** %4
  %57 = getelementptr inbounds %"class.std::__1::__libcpp_compressed_pair_imp.9"* %56, i32 0, i32 0
  %58 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep"* %57, i32 0, i32 0
  %59 = bitcast %union.anon* %58 to %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long"*
  %60 = getelementptr inbounds %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long"* %59, i32 0, i32 0
  %61 = load i64* %60, align 8
  %62 = and i64 %61, -2
  store %"class.std::__1::allocator.10"* %41, %"class.std::__1::allocator.10"** %13, align 8
  store i8* %51, i8** %14, align 8
  store i64 %62, i64* %15, align 8
  %63 = load %"class.std::__1::allocator.10"** %13, align 8
  %64 = load i8** %14, align 8
  %65 = load i64* %15, align 8
  store %"class.std::__1::allocator.10"* %63, %"class.std::__1::allocator.10"** %10, align 8
  store i8* %64, i8** %11, align 8
  store i64 %65, i64* %12, align 8
  %66 = load %"class.std::__1::allocator.10"** %10
  %67 = load i8** %11, align 8
  call void @_ZdlPv(i8* %67) #2
  br label %68

; <label>:68                                      ; preds = %35, %0
  ret void
}

define internal void @_GLOBAL__sub_I_directfunction_handle.cpp() section ".text.startup" {
  call void @__cxx_global_var_init()
  ret void
}

; CHECK: declare void @__hcLaunchKernel__Z6kernel16grid_launch_parmP3fooPi(%struct.grid_launch_parm*, %struct.foo*, i32*)

attributes #0 = { alwaysinline uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline nounwind uwtable "hc_grid_launch" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { inlinehint uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noinline noreturn nounwind }
attributes #9 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { nobuiltin nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { builtin nounwind }
attributes #12 = { noreturn nounwind }

!llvm.ident = !{!0}

!0 = metadata !{metadata !"HCC clang version 3.5.0 (tags/RELEASE_350/final) (based on HCC 0.8.1543-ffaace2-8c155af-e65013b LLVM 3.5.0svn)"}
