; RUN: %spirify %s
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

%"class.Concurrency::index.4.24.144" = type { %"struct.Concurrency::index_impl.2.23.143" }
%"struct.Concurrency::index_impl.2.23.143" = type { %"class.Concurrency::__index_leaf.0.22.142" }
%"class.Concurrency::__index_leaf.0.22.142" = type { i32, i32 }
%"class.Concurrency::index.0.7.27.147" = type { %"struct.Concurrency::index_impl.1.6.26.146" }
%"struct.Concurrency::index_impl.1.6.26.146" = type { %"class.Concurrency::__index_leaf.0.22.142", %"class.Concurrency::__index_leaf.2.5.25.145" }
%"class.Concurrency::__index_leaf.2.5.25.145" = type { i32, i32 }
%"class.Concurrency::index.3.10.30.150" = type { %"struct.Concurrency::index_impl.4.9.29.149" }
%"struct.Concurrency::index_impl.4.9.29.149" = type { %"class.Concurrency::__index_leaf.0.22.142", %"class.Concurrency::__index_leaf.2.5.25.145", %"class.Concurrency::__index_leaf.5.8.28.148" }
%"class.Concurrency::__index_leaf.5.8.28.148" = type { i32, i32 }
%"class.Concurrency::tiled_extent.12.32.152" = type { %"class.Concurrency::extent.11.31.151" }
%"class.Concurrency::extent.11.31.151" = type { %"struct.Concurrency::index_impl.2.23.143" }
%class.anon.15.35.155 = type { %"class.Concurrency::array.14.34.154"*, i32 }
%"class.Concurrency::array.14.34.154" = type { %"class.Concurrency::extent.11.31.151", %"class.Concurrency::_data.13.33.153", i32 }
%"class.Concurrency::_data.13.33.153" = type { i32* }
%"class.Concurrency::tiled_extent.7.17.37.157" = type { %"class.Concurrency::extent.8.16.36.156" }
%"class.Concurrency::extent.8.16.36.156" = type { %"struct.Concurrency::index_impl.1.6.26.146" }
%class.anon.6.18.38.158 = type { %"class.Concurrency::array.14.34.154"*, %"class.Concurrency::array.14.34.154"*, i32, i32 }
%class.anon.10.19.39.159 = type { %"class.Concurrency::array.14.34.154"*, i32, i32 }
%class.anon.11.20.40.160 = type { %"class.Concurrency::array.14.34.154"*, %"class.Concurrency::array.14.34.154"*, i32, i32 }
%class.anon.12.21.41.161 = type { %"class.Concurrency::array.14.34.154"*, i32, i32 }

@.str = external unnamed_addr constant [22 x i8], section "llvm.metadata"
@.str1 = external unnamed_addr constant [55 x i8], section "llvm.metadata"
@.str2 = external unnamed_addr constant [20 x i8], section "llvm.metadata"
@.str3 = external unnamed_addr constant [83 x i8], section "llvm.metadata"
@_ZZ19bitonic_sort_kernelIiEvRN11Concurrency5arrayIT_Li1EEEjjNS0_11tiled_indexILi256ELi0ELi0EEEE7sh_data = external global [256 x i32], section "clamp_opencl_local", align 4
@_ZZ16transpose_kernelIiEvRN11Concurrency5arrayIT_Li1EEES4_jjNS0_11tiled_indexILi16ELi16ELi0EEEE21transpose_shared_data = external global [16 x [16 x i32]], section "clamp_opencl_local", align 4
@llvm.global.annotations = external global [8 x { i8*, i8*, i8*, i32 }], section "llvm.metadata"
@llvm.used = external global [5 x i8*], section "llvm.metadata"

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEC1Ev(%"class.Concurrency::index.4.24.144"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEC2Ev(%"class.Concurrency::index.4.24.144"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEC1ERKS1_(%"class.Concurrency::index.4.24.144"* nocapture, %"class.Concurrency::index.4.24.144"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEC2ERKS1_(%"class.Concurrency::index.4.24.144"* nocapture, %"class.Concurrency::index.4.24.144"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEC1EPi(%"class.Concurrency::index.4.24.144"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEC2EPi(%"class.Concurrency::index.4.24.144"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEC1EPKi(%"class.Concurrency::index.4.24.144"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEC2EPKi(%"class.Concurrency::index.4.24.144"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEaSERKS1_(%"class.Concurrency::index.4.24.144"*, %"class.Concurrency::index.4.24.144"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind readonly
declare i32 @_ZNK11Concurrency5indexILi1EEixEj(%"class.Concurrency::index.4.24.144"* nocapture, i32) #1 align 2

; Function Attrs: alwaysinline nounwind readnone
declare i32* @_ZN11Concurrency5indexILi1EEixEj(%"class.Concurrency::index.4.24.144"*, i32) #2 align 2

; Function Attrs: alwaysinline nounwind readonly
declare zeroext i1 @_ZNK11Concurrency5indexILi1EEeqERKS1_(%"class.Concurrency::index.4.24.144"* nocapture, %"class.Concurrency::index.4.24.144"* nocapture) #1 align 2

; Function Attrs: alwaysinline nounwind readonly
declare zeroext i1 @_ZNK11Concurrency5indexILi1EEneERKS1_(%"class.Concurrency::index.4.24.144"* nocapture, %"class.Concurrency::index.4.24.144"* nocapture) #1 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEpLERKS1_(%"class.Concurrency::index.4.24.144"*, %"class.Concurrency::index.4.24.144"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEmIERKS1_(%"class.Concurrency::index.4.24.144"*, %"class.Concurrency::index.4.24.144"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEmLERKS1_(%"class.Concurrency::index.4.24.144"*, %"class.Concurrency::index.4.24.144"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEdVERKS1_(%"class.Concurrency::index.4.24.144"*, %"class.Concurrency::index.4.24.144"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EErMERKS1_(%"class.Concurrency::index.4.24.144"*, %"class.Concurrency::index.4.24.144"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEpLEi(%"class.Concurrency::index.4.24.144"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEmIEi(%"class.Concurrency::index.4.24.144"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEmLEi(%"class.Concurrency::index.4.24.144"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEdVEi(%"class.Concurrency::index.4.24.144"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EErMEi(%"class.Concurrency::index.4.24.144"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEppEv(%"class.Concurrency::index.4.24.144"*) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEppEi(%"class.Concurrency::index.4.24.144"* noalias nocapture sret, %"class.Concurrency::index.4.24.144"* nocapture, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.4.24.144"* @_ZN11Concurrency5indexILi1EEmmEv(%"class.Concurrency::index.4.24.144"*) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EEmmEi(%"class.Concurrency::index.4.24.144"* noalias nocapture sret, %"class.Concurrency::index.4.24.144"* nocapture, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi1EE21__cxxamp_opencl_indexEv(%"class.Concurrency::index.4.24.144"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EEC1Ev(%"class.Concurrency::index.0.7.27.147"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EEC2Ev(%"class.Concurrency::index.0.7.27.147"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EEC1ERKS1_(%"class.Concurrency::index.0.7.27.147"* nocapture, %"class.Concurrency::index.0.7.27.147"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EEC2ERKS1_(%"class.Concurrency::index.0.7.27.147"* nocapture, %"class.Concurrency::index.0.7.27.147"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EEC1EPi(%"class.Concurrency::index.0.7.27.147"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EEC2EPi(%"class.Concurrency::index.0.7.27.147"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EEC1EPKi(%"class.Concurrency::index.0.7.27.147"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EEC2EPKi(%"class.Concurrency::index.0.7.27.147"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEaSERKS1_(%"class.Concurrency::index.0.7.27.147"*, %"class.Concurrency::index.0.7.27.147"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind readonly
declare i32 @_ZNK11Concurrency5indexILi2EEixEj(%"class.Concurrency::index.0.7.27.147"* nocapture, i32) #1 align 2

; Function Attrs: alwaysinline nounwind readnone
declare i32* @_ZN11Concurrency5indexILi2EEixEj(%"class.Concurrency::index.0.7.27.147"*, i32) #2 align 2

; Function Attrs: alwaysinline nounwind readonly
declare zeroext i1 @_ZNK11Concurrency5indexILi2EEeqERKS1_(%"class.Concurrency::index.0.7.27.147"* nocapture, %"class.Concurrency::index.0.7.27.147"* nocapture) #1 align 2

; Function Attrs: alwaysinline nounwind readonly
declare zeroext i1 @_ZNK11Concurrency5indexILi2EEneERKS1_(%"class.Concurrency::index.0.7.27.147"* nocapture, %"class.Concurrency::index.0.7.27.147"* nocapture) #1 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEpLERKS1_(%"class.Concurrency::index.0.7.27.147"*, %"class.Concurrency::index.0.7.27.147"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEmIERKS1_(%"class.Concurrency::index.0.7.27.147"*, %"class.Concurrency::index.0.7.27.147"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEmLERKS1_(%"class.Concurrency::index.0.7.27.147"*, %"class.Concurrency::index.0.7.27.147"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEdVERKS1_(%"class.Concurrency::index.0.7.27.147"*, %"class.Concurrency::index.0.7.27.147"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EErMERKS1_(%"class.Concurrency::index.0.7.27.147"*, %"class.Concurrency::index.0.7.27.147"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEpLEi(%"class.Concurrency::index.0.7.27.147"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEmIEi(%"class.Concurrency::index.0.7.27.147"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEmLEi(%"class.Concurrency::index.0.7.27.147"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEdVEi(%"class.Concurrency::index.0.7.27.147"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EErMEi(%"class.Concurrency::index.0.7.27.147"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEppEv(%"class.Concurrency::index.0.7.27.147"*) #0 align 2

; Function Attrs: alwaysinline nounwind
define weak_odr void @_ZN11Concurrency5indexILi2EEppEi(%"class.Concurrency::index.0.7.27.147"* noalias nocapture sret %agg.result, %"class.Concurrency::index.0.7.27.147"* nocapture %this, i32) #0 align 2 {
entry:
  ret void
}

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.0.7.27.147"* @_ZN11Concurrency5indexILi2EEmmEv(%"class.Concurrency::index.0.7.27.147"*) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EEmmEi(%"class.Concurrency::index.0.7.27.147"* noalias nocapture sret, %"class.Concurrency::index.0.7.27.147"* nocapture, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi2EE21__cxxamp_opencl_indexEv(%"class.Concurrency::index.0.7.27.147"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EEC1Ev(%"class.Concurrency::index.3.10.30.150"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EEC2Ev(%"class.Concurrency::index.3.10.30.150"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EEC1ERKS1_(%"class.Concurrency::index.3.10.30.150"* nocapture, %"class.Concurrency::index.3.10.30.150"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EEC2ERKS1_(%"class.Concurrency::index.3.10.30.150"* nocapture, %"class.Concurrency::index.3.10.30.150"* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EEC1EPi(%"class.Concurrency::index.3.10.30.150"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EEC2EPi(%"class.Concurrency::index.3.10.30.150"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
define weak_odr void @_ZN11Concurrency5indexILi3EEC1EPKi(%"class.Concurrency::index.3.10.30.150"* nocapture %this, i32* nocapture %components) unnamed_addr #0 align 2 {
entry:
  ret void
}

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EEC2EPKi(%"class.Concurrency::index.3.10.30.150"* nocapture, i32* nocapture) unnamed_addr #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEaSERKS1_(%"class.Concurrency::index.3.10.30.150"*, %"class.Concurrency::index.3.10.30.150"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind readonly
declare i32 @_ZNK11Concurrency5indexILi3EEixEj(%"class.Concurrency::index.3.10.30.150"* nocapture, i32) #1 align 2

; Function Attrs: alwaysinline nounwind readnone
declare i32* @_ZN11Concurrency5indexILi3EEixEj(%"class.Concurrency::index.3.10.30.150"*, i32) #2 align 2

; Function Attrs: alwaysinline nounwind readonly
declare zeroext i1 @_ZNK11Concurrency5indexILi3EEeqERKS1_(%"class.Concurrency::index.3.10.30.150"* nocapture, %"class.Concurrency::index.3.10.30.150"* nocapture) #1 align 2

; Function Attrs: alwaysinline nounwind readonly
declare zeroext i1 @_ZNK11Concurrency5indexILi3EEneERKS1_(%"class.Concurrency::index.3.10.30.150"* nocapture, %"class.Concurrency::index.3.10.30.150"* nocapture) #1 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEpLERKS1_(%"class.Concurrency::index.3.10.30.150"*, %"class.Concurrency::index.3.10.30.150"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEmIERKS1_(%"class.Concurrency::index.3.10.30.150"*, %"class.Concurrency::index.3.10.30.150"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEmLERKS1_(%"class.Concurrency::index.3.10.30.150"*, %"class.Concurrency::index.3.10.30.150"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEdVERKS1_(%"class.Concurrency::index.3.10.30.150"*, %"class.Concurrency::index.3.10.30.150"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EErMERKS1_(%"class.Concurrency::index.3.10.30.150"*, %"class.Concurrency::index.3.10.30.150"* nocapture) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEpLEi(%"class.Concurrency::index.3.10.30.150"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEmIEi(%"class.Concurrency::index.3.10.30.150"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEmLEi(%"class.Concurrency::index.3.10.30.150"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEdVEi(%"class.Concurrency::index.3.10.30.150"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EErMEi(%"class.Concurrency::index.3.10.30.150"*, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEppEv(%"class.Concurrency::index.3.10.30.150"*) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EEppEi(%"class.Concurrency::index.3.10.30.150"* noalias nocapture sret, %"class.Concurrency::index.3.10.30.150"* nocapture, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare %"class.Concurrency::index.3.10.30.150"* @_ZN11Concurrency5indexILi3EEmmEv(%"class.Concurrency::index.3.10.30.150"*) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EEmmEi(%"class.Concurrency::index.3.10.30.150"* noalias nocapture sret, %"class.Concurrency::index.3.10.30.150"* nocapture, i32) #0 align 2

; Function Attrs: alwaysinline nounwind
declare void @_ZN11Concurrency5indexILi3EE21__cxxamp_opencl_indexEv(%"class.Concurrency::index.3.10.30.150"* nocapture) #0 align 2

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi256EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi256ELi0ELi0EEEE_EEvNS_12tiled_extentIXT_ELi0ELi0EEERKT0_(%"class.Concurrency::tiled_extent.12.32.152"* nocapture, %class.anon.15.35.155* nocapture) #3

define cc76 void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi256ELi0ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEj(i32, i32, i32*, i32, i32) #4 align 2 {
entry:
  %arrayidx.i.i = getelementptr inbounds [256 x i32]* @_ZZ19bitonic_sort_kernelIiEvRN11Concurrency5arrayIT_Li1EEEjjNS0_11tiled_indexILi256ELi0ELi0EEEE7sh_data, i32 0, i32 undef
  unreachable
}

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi16ELi16EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi16ELi16ELi0EEEE_EEvNS_12tiled_extentIXT_EXT0_ELi0EEERKT1_(%"class.Concurrency::tiled_extent.7.17.37.157"* nocapture, %class.anon.6.18.38.158* nocapture) #3

declare cc76 void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj(i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32) #4 align 2

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi256EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi256ELi0ELi0EEEE0_EEvNS_12tiled_extentIXT_ELi0ELi0EEERKT0_(%"class.Concurrency::tiled_extent.12.32.152"* nocapture, %class.anon.10.19.39.159* nocapture) #3

define cc76 void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi256ELi0ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj(i32, i32, i32*, i32, i32, i32) #4 align 2 {
entry:
  %arrayidx.i.i = getelementptr inbounds [256 x i32]* @_ZZ19bitonic_sort_kernelIiEvRN11Concurrency5arrayIT_Li1EEEjjNS0_11tiled_indexILi256ELi0ELi0EEEE7sh_data, i32 0, i32 undef
  unreachable
}

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi16ELi16EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi16ELi16ELi0EEEE0_EEvNS_12tiled_extentIXT_EXT0_ELi0EEERKT1_(%"class.Concurrency::tiled_extent.7.17.37.157"* nocapture, %class.anon.11.20.40.160* nocapture) #3

declare cc76 void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj(i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32) #4 align 2

; Function Attrs: noinline nounwind readonly
declare void @_ZN11Concurrency17parallel_for_eachILi256EZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS2_9allocatorIS4_EEEES8_EUlNS_11tiled_indexILi256ELi0ELi0EEEE1_EEvNS_12tiled_extentIXT_ELi0ELi0EEERKT0_(%"class.Concurrency::tiled_extent.12.32.152"* nocapture, %class.anon.12.21.41.161* nocapture) #3

declare cc76 void @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi256ELi0ELi0EEEE1_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj(i32, i32, i32*, i32, i32, i32) #4 align 2

; Function Attrs: noduplicate
declare void @barrier(i32) #5

; Function Attrs: nounwind readonly
declare i32 @get_global_id(i32) #6

; Function Attrs: nounwind readonly
declare i32 @get_local_id(i32) #6

attributes #0 = { alwaysinline nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { alwaysinline nounwind readnone "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noduplicate "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!hcc.kernels = !{!0, !6, !12, !13, !14}

!0 = metadata !{void (i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi256ELi0ELi0EEEE1_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj, metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
!1 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!2 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!3 = metadata !{metadata !"kernel_arg_type", metadata !"int", metadata !"int", metadata !"int*", metadata !"enum Concurrency::access_type", metadata !"uint", metadata !"uint"}
!4 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!5 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!6 = metadata !{void (i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj, metadata !7, metadata !8, metadata !9, metadata !10, metadata !11}
!7 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!8 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!9 = metadata !{metadata !"kernel_arg_type", metadata !"int", metadata !"int", metadata !"int*", metadata !"enum Concurrency::access_type", metadata !"int", metadata !"int", metadata !"int*", metadata !"enum Concurrency::access_type", metadata !"uint", metadata !"uint"}
!10 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!11 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!12 = metadata !{void (i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi256ELi0ELi0EEEE0_19__cxxamp_trampolineEiiPiNS8_11access_typeEjj, metadata !1, metadata !2, metadata !3, metadata !4, metadata !5}
!13 = metadata !{void (i32, i32, i32*, i32, i32, i32, i32*, i32, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi16ELi16ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEiiSC_SD_jj, metadata !7, metadata !8, metadata !9, metadata !10, metadata !11}
!14 = metadata !{void (i32, i32, i32*, i32, i32)* @_ZZ16bitonic_sort_ampIiEvRNSt3__16vectorIT_NS0_9allocatorIS2_EEEES6_ENS_IiEUlN11Concurrency11tiled_indexILi256ELi0ELi0EEEE_19__cxxamp_trampolineEiiPiNS8_11access_typeEj, metadata !15, metadata !16, metadata !17, metadata !18, metadata !19}
!15 = metadata !{metadata !"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 0, i32 0}
!16 = metadata !{metadata !"kernel_arg_access_qual", metadata !"none", metadata !"none", metadata !"none", metadata !"none", metadata !"none"}
!17 = metadata !{metadata !"kernel_arg_type", metadata !"int", metadata !"int", metadata !"int*", metadata !"enum Concurrency::access_type", metadata !"uint"}
!18 = metadata !{metadata !"kernel_arg_type_qual", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
!19 = metadata !{metadata !"kernel_arg_name", metadata !"", metadata !"", metadata !"", metadata !"", metadata !""}
