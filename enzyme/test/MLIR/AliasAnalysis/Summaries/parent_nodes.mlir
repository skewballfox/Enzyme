// RUN: %eopt --print-activity-analysis='use-annotations' --split-input-file %s | FileCheck %s

#alias_scope_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "mat_mult">
#alias_scope_domain1 = #llvm.alias_scope_domain<id = distinct[1]<>, description = "mat_mult">
#alias_scope_domain2 = #llvm.alias_scope_domain<id = distinct[2]<>, description = "mat_mult">
#alias_scope_domain3 = #llvm.alias_scope_domain<id = distinct[3]<>, description = "mat_mult">
#alias_scope_domain4 = #llvm.alias_scope_domain<id = distinct[4]<>, description = "mat_mult">
#alias_scope_domain5 = #llvm.alias_scope_domain<id = distinct[5]<>, description = "mat_mult">
#alias_scope_domain6 = #llvm.alias_scope_domain<id = distinct[6]<>, description = "mat_mult">
#loop_unroll = #llvm.loop_unroll<disable = true>
#tbaa_root = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
#alias_scope = #llvm.alias_scope<id = distinct[7]<>, domain = #alias_scope_domain, description = "mat_mult: %rhs">
#alias_scope1 = #llvm.alias_scope<id = distinct[8]<>, domain = #alias_scope_domain, description = "mat_mult: %lhs">
#alias_scope2 = #llvm.alias_scope<id = distinct[9]<>, domain = #alias_scope_domain, description = "mat_mult: %out">
#alias_scope3 = #llvm.alias_scope<id = distinct[10]<>, domain = #alias_scope_domain1, description = "mat_mult: %lhs">
#alias_scope4 = #llvm.alias_scope<id = distinct[11]<>, domain = #alias_scope_domain1, description = "mat_mult: %rhs">
#alias_scope5 = #llvm.alias_scope<id = distinct[12]<>, domain = #alias_scope_domain1, description = "mat_mult: %out">
#alias_scope6 = #llvm.alias_scope<id = distinct[13]<>, domain = #alias_scope_domain2, description = "mat_mult: %lhs">
#alias_scope7 = #llvm.alias_scope<id = distinct[14]<>, domain = #alias_scope_domain2, description = "mat_mult: %rhs">
#alias_scope8 = #llvm.alias_scope<id = distinct[15]<>, domain = #alias_scope_domain2, description = "mat_mult: %out">
#alias_scope9 = #llvm.alias_scope<id = distinct[16]<>, domain = #alias_scope_domain3, description = "mat_mult: %out">
#alias_scope10 = #llvm.alias_scope<id = distinct[17]<>, domain = #alias_scope_domain3, description = "mat_mult: %lhs">
#alias_scope11 = #llvm.alias_scope<id = distinct[18]<>, domain = #alias_scope_domain3, description = "mat_mult: %rhs">
#alias_scope12 = #llvm.alias_scope<id = distinct[19]<>, domain = #alias_scope_domain4, description = "mat_mult: %lhs">
#alias_scope13 = #llvm.alias_scope<id = distinct[20]<>, domain = #alias_scope_domain4, description = "mat_mult: %out">
#alias_scope14 = #llvm.alias_scope<id = distinct[21]<>, domain = #alias_scope_domain4, description = "mat_mult: %rhs">
#alias_scope15 = #llvm.alias_scope<id = distinct[22]<>, domain = #alias_scope_domain5, description = "mat_mult: %lhs">
#alias_scope16 = #llvm.alias_scope<id = distinct[23]<>, domain = #alias_scope_domain5, description = "mat_mult: %rhs">
#alias_scope17 = #llvm.alias_scope<id = distinct[24]<>, domain = #alias_scope_domain5, description = "mat_mult: %out">
#alias_scope18 = #llvm.alias_scope<id = distinct[25]<>, domain = #alias_scope_domain6, description = "mat_mult: %lhs">
#alias_scope19 = #llvm.alias_scope<id = distinct[26]<>, domain = #alias_scope_domain6, description = "mat_mult: %rhs">
#alias_scope20 = #llvm.alias_scope<id = distinct[27]<>, domain = #alias_scope_domain6, description = "mat_mult: %out">
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
#tbaa_type_desc1 = #llvm.tbaa_type_desc<id = "int", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc2 = #llvm.tbaa_type_desc<id = "any pointer", members = {<#tbaa_type_desc, 0>}>
#tbaa_type_desc3 = #llvm.tbaa_type_desc<id = "double", members = {<#tbaa_type_desc, 0>}>
#tbaa_tag1 = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
#tbaa_tag2 = #llvm.tbaa_tag<base_type = #tbaa_type_desc1, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_type_desc4 = #llvm.tbaa_type_desc<id = "Matrix", members = {<#tbaa_type_desc1, 0>, <#tbaa_type_desc1, 4>, <#tbaa_type_desc2, 8>}>
#tbaa_tag3 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc1, offset = 0>
#tbaa_tag4 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc1, offset = 4>
#tbaa_tag5 = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc2, offset = 8>

llvm.func local_unnamed_addr @malloc(i64 {llvm.noundef}) -> (!llvm.ptr {llvm.noalias, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nofree", "nounwind", "willreturn", ["allockind", "9"], ["allocsize", "4294967295"], ["alloc-family", "malloc"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], sym_visibility = "private", target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>}
llvm.func local_unnamed_addr @free(!llvm.ptr {llvm.allocptr, llvm.nocapture, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite>, passthrough = ["mustprogress", "nounwind", "willreturn", ["allockind", "4"], ["alloc-family", "malloc"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], sym_visibility = "private", target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>}

llvm.func @euler_angles_to_rotation_matrix(!llvm.ptr, !llvm.ptr) attributes {
  p2psummary = [
    [distinct[30]<"arg-euler_angles_to_rotation_matrix-1">, [distinct[31]<"arg-euler_angles_to_rotation_matrix-1-deref">, distinct[32]<"fresh-euler_angles_malloc">]],
    [distinct[32]<"fresh-euler_angles_malloc">, []],
    [distinct[33]<"fresh-malloc_RX">, []],
    [distinct[34]<"fresh-malloc_RY">, []],
    [distinct[35]<"fresh-malloc_RZ">, []],
    [distinct[36]<"fresh-malloc_tmp">, []]
  ]}

// CHECK-LABEL: processing function @get_posed_relatives
// CHECK: p2p summary:
// CHECK-NEXT:    distinct[0]<"arg-get_posed_relatives-1"> -> [distinct[0]<"arg-get_posed_relatives-1-deref">]
// CHECK-NEXT:    distinct[0]<"arg-get_posed_relatives-2"> -> [distinct[0]<"arg-get_posed_relatives-2-deref">]
// CHECK-NEXT:    distinct[0]<"arg-get_posed_relatives-3"> -> [distinct[0]<"arg-get_posed_relatives-3-deref">, distinct[1]<"fresh-malloc4">]
// CHECK-NEXT:    distinct[0]<"fresh-malloc1"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-malloc2"> -> [distinct[0]<"fresh-euler_angles_malloc">, distinct[1]<"fresh-malloc3">]
// CHECK-NEXT:    distinct[0]<"fresh-malloc3"> -> []
// CHECK-NEXT:    distinct[0]<"fresh-malloc4"> -> []
llvm.func local_unnamed_addr @get_posed_relatives(%arg0: i32 {llvm.noundef}, %arg1: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg2: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef, llvm.readonly}, %arg3: !llvm.ptr {llvm.noalias, llvm.nocapture, llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<"non-leaf">, passthrough = ["nounwind", "ssp", ["uwtable", "1"], ["approx-func-fp-math", "true"], ["no-infs-fp-math", "true"], ["no-nans-fp-math", "true"], ["no-signed-zeros-fp-math", "true"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "apple-m1"], ["unsafe-fp-math", "true"]], target_cpu = "apple-m1", target_features = #llvm.target_features<["+aes", "+complxnum", "+crc", "+dotprod", "+fp-armv8", "+fp16fml", "+fullfp16", "+jsconv", "+lse", "+neon", "+ras", "+rcpc", "+rdm", "+sha2", "+sha3", "+v8.1a", "+v8.2a", "+v8.3a", "+v8.4a", "+v8.5a", "+v8a", "+zcm", "+zcz"]>} {
  %0 = llvm.mlir.constant(128 : i64) : i64
  %1 = llvm.mlir.constant(16 : i64) : i64
  %2 = llvm.mlir.constant(3 : i32) : i32
  %3 = llvm.mlir.constant(0 : i64) : i64
  %4 = llvm.mlir.constant(1 : i32) : i32
  %5 = llvm.mlir.constant(72 : i64) : i64
  %6 = llvm.mlir.constant(2 : i32) : i32
  %7 = llvm.mlir.constant(0 : i32) : i32
  %8 = llvm.mlir.constant(2 : i64) : i64
  %9 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %10 = llvm.mlir.constant(0.000000e+00 : f64) : f64
  %11 = llvm.mlir.constant(1 : i64) : i64
  %12 = llvm.mlir.constant(4 : i64) : i64
  %13 = llvm.mlir.constant(3 : i64) : i64
  %14 = llvm.mlir.constant(5 : i64) : i64
  %15 = llvm.mlir.zero {nullptr} : !llvm.ptr
  %16 = llvm.mlir.constant(4 : i32) : i32
  %17 = llvm.call @malloc(%0) {tag = "malloc1"} : (i64) -> !llvm.ptr
  %18 = llvm.call @malloc(%1) {tag = "malloc2"} : (i64) -> !llvm.ptr
  llvm.store %2, %18 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %19 = llvm.getelementptr inbounds %18[%3, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.store %2, %19 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  %20 = llvm.call @malloc(%5) {tag = "malloc3"} : (i64) -> !llvm.ptr
  %21 = llvm.getelementptr inbounds %18[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.store %20, %21 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  %22 = llvm.icmp "sgt" %arg0, %7 : i32
  llvm.cond_br %22, ^bb1, ^bb25
^bb1:  // pred: ^bb0
  %23 = llvm.getelementptr inbounds %arg2[%3, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %24 = llvm.load %23 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %25 = llvm.load %arg2 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %26 = llvm.sext %25 : i32 to i64
  %27 = llvm.zext %arg0 : i32 to i64
  llvm.br ^bb2(%3 : i64)
^bb2(%28: i64):  // 2 preds: ^bb1, ^bb24
  llvm.br ^bb3(%3 : i64)
^bb3(%29: i64):  // 2 preds: ^bb2, ^bb5
  %30 = llvm.shl %29, %8  : i64
  %31 = llvm.getelementptr %17[%30] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb4(%3 : i64)
^bb4(%32: i64):  // 2 preds: ^bb3, ^bb4
  %33 = llvm.icmp "eq" %29, %32 : i64
  %34 = llvm.select %33, %9, %10 : i1, f64
  %35 = llvm.getelementptr %31[%32] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %34, %35 {alignment = 8 : i64, tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %36 = llvm.add %32, %11 overflow<nsw, nuw>  : i64
  %37 = llvm.icmp "eq" %36, %12 : i64
  llvm.cond_br %37, ^bb5, ^bb4(%36 : i64) {loop_annotation = #loop_annotation}
^bb5:  // pred: ^bb4
  %38 = llvm.add %29, %11 overflow<nsw, nuw>  : i64
  %39 = llvm.icmp "eq" %38, %12 : i64
  llvm.cond_br %39, ^bb6, ^bb3(%38 : i64) {loop_annotation = #loop_annotation}
^bb6:  // pred: ^bb5
  %40 = llvm.add %28, %13 overflow<nsw, nuw>  : i64
  %41 = llvm.mul %40, %26 overflow<nsw>  : i64
  %42 = llvm.getelementptr inbounds %24[%41] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.call @euler_angles_to_rotation_matrix(%42, %18) : (!llvm.ptr, !llvm.ptr) -> ()
  %43 = llvm.load %19 {alignment = 4 : i64, tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %44 = llvm.icmp "sgt" %43, %7 : i32
  llvm.cond_br %44, ^bb7, ^bb11
^bb7:  // pred: ^bb6
  %45 = llvm.load %18 {alignment = 8 : i64, tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %46 = llvm.icmp "sgt" %45, %7 : i32
  %47 = llvm.sext %45 : i32 to i64
  %48 = llvm.zext %43 : i32 to i64
  %49 = llvm.zext %45 : i32 to i64
  %50 = llvm.shl %49, %13 overflow<nsw, nuw>  : i64
  llvm.br ^bb8(%3 : i64)
^bb8(%51: i64):  // 2 preds: ^bb7, ^bb10
  llvm.cond_br %46, ^bb9, ^bb10
^bb9:  // pred: ^bb8
  %52 = llvm.shl %51, %14 overflow<nsw, nuw>  : i64
  %53 = llvm.getelementptr %17[%52] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %54 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %55 = llvm.mul %51, %47 overflow<nsw>  : i64
  %56 = llvm.getelementptr %54[%55] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  "llvm.intr.memcpy"(%53, %56, %50) <{isVolatile = false, tbaa = [#tbaa_tag1]}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  llvm.br ^bb10
^bb10:  // 2 preds: ^bb8, ^bb9
  %57 = llvm.add %51, %11 overflow<nsw, nuw>  : i64
  %58 = llvm.icmp "eq" %57, %48 : i64
  llvm.cond_br %58, ^bb11, ^bb8(%57 : i64) {loop_annotation = #loop_annotation}
^bb11:  // 2 preds: ^bb6, ^bb10
  %59 = llvm.getelementptr inbounds %arg1[%28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %60 = llvm.getelementptr inbounds %arg3[%28] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  llvm.intr.experimental.noalias.scope.decl #alias_scope12
  llvm.intr.experimental.noalias.scope.decl #alias_scope13
  %61 = llvm.load %59 {alias_scopes = [#alias_scope12], alignment = 8 : i64, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %62 = llvm.load %60 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag3]} : !llvm.ptr -> i32
  %63 = llvm.getelementptr inbounds %arg3[%28, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %64 = llvm.load %63 {alias_scopes = [#alias_scope13], alignment = 4 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %65 = llvm.mul %64, %62 overflow<nsw>  : i32
  %66 = llvm.shl %61, %6 overflow<nsw>  : i32
  %67 = llvm.icmp "eq" %65, %66 : i32
  llvm.cond_br %67, ^bb17, ^bb12
^bb12:  // pred: ^bb11
  %68 = llvm.getelementptr inbounds %arg3[%28, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %69 = llvm.load %68 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %70 = llvm.icmp "eq" %69, %15 : !llvm.ptr
  llvm.cond_br %70, ^bb14, ^bb13
^bb13:  // pred: ^bb12
  llvm.call @free(%69) {noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13]} : (!llvm.ptr) -> ()
  llvm.br ^bb14
^bb14:  // 2 preds: ^bb12, ^bb13
  %71 = llvm.icmp "sgt" %61, %7 : i32
  llvm.cond_br %71, ^bb15, ^bb16(%15 : !llvm.ptr)
^bb15:  // pred: ^bb14
  %72 = llvm.zext %66 : i32 to i64
  %73 = llvm.shl %72, %13 overflow<nsw, nuw>  : i64
  %74 = llvm.call @malloc(%73) {tag = "malloc4"} : (i64) -> !llvm.ptr
  llvm.br ^bb16(%74 : !llvm.ptr)
^bb16(%75: !llvm.ptr):  // 2 preds: ^bb14, ^bb15
  llvm.store %75, %68 {debugme, alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb17
^bb17:  // 2 preds: ^bb11, ^bb16
  llvm.store %16, %63 {alias_scopes = [#alias_scope13], alignment = 4 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag4]} : i32, !llvm.ptr
  llvm.store %61, %60 {bookmark, alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag3]} : i32, !llvm.ptr
  %76 = llvm.icmp "sgt" %61, %7 : i32
  llvm.cond_br %76, ^bb18, ^bb24
^bb18:  // pred: ^bb17
  %77 = llvm.getelementptr inbounds %arg1[%28, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %78 = llvm.getelementptr inbounds %arg3[%28, 2] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %79 = llvm.getelementptr inbounds %arg1[%28, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<"struct.Matrix", (i32, i32, ptr)>
  %80 = llvm.zext %61 : i32 to i64
  %81 = llvm.load %77 {alias_scopes = [#alias_scope12], alignment = 8 : i64, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %82 = llvm.load %78 {alias_scopes = [#alias_scope13], alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14], tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %83 = llvm.load %79 {alias_scopes = [#alias_scope12], alignment = 4 : i64, noalias_scopes = [#alias_scope14, #alias_scope13], tbaa = [#tbaa_tag4]} : !llvm.ptr -> i32
  %84 = llvm.icmp "sgt" %83, %4 : i32
  %85 = llvm.zext %83 : i32 to i64
  llvm.br ^bb19(%3 : i64)
^bb19(%86: i64):  // 2 preds: ^bb18, ^bb23
  %87 = llvm.getelementptr inbounds %81[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %88 = llvm.getelementptr %82[%86] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.br ^bb20(%3 : i64)
^bb20(%89: i64):  // 2 preds: ^bb19, ^bb22
  %90 = llvm.load %87 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %91 = llvm.shl %89, %8 overflow<nsw>  : i64
  %92 = llvm.getelementptr inbounds %17[%91] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %93 = llvm.load %92 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %94 = llvm.fmul %93, %90  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %95 = llvm.mul %89, %80 overflow<nsw, nuw>  : i64
  %96 = llvm.getelementptr %88[%95] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  llvm.store %94, %96 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  llvm.cond_br %84, ^bb21(%11, %94 : i64, f64), ^bb22
^bb21(%97: i64, %98: f64):  // 2 preds: ^bb20, ^bb21
  %99 = llvm.mul %97, %80 overflow<nsw, nuw>  : i64
  %100 = llvm.getelementptr %87[%99] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %101 = llvm.load %100 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %102 = llvm.getelementptr %92[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %103 = llvm.load %102 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : !llvm.ptr -> f64
  %104 = llvm.fmul %103, %101  {fastmathFlags = #llvm.fastmath<fast>} : f64
  %105 = llvm.fadd %104, %98  {fastmathFlags = #llvm.fastmath<fast>} : f64
  llvm.store %105, %96 {alignment = 8 : i64, noalias_scopes = [#alias_scope12, #alias_scope14, #alias_scope13], tbaa = [#tbaa_tag1]} : f64, !llvm.ptr
  %106 = llvm.add %97, %11 overflow<nsw, nuw>  : i64
  %107 = llvm.icmp "eq" %106, %85 : i64
  llvm.cond_br %107, ^bb22, ^bb21(%106, %105 : i64, f64) {loop_annotation = #loop_annotation}
^bb22:  // 2 preds: ^bb20, ^bb21
  %108 = llvm.add %89, %11 overflow<nsw, nuw>  : i64
  %109 = llvm.icmp "eq" %108, %12 : i64
  llvm.cond_br %109, ^bb23, ^bb20(%108 : i64) {loop_annotation = #loop_annotation}
^bb23:  // pred: ^bb22
  %110 = llvm.add %86, %11 overflow<nsw, nuw>  : i64
  %111 = llvm.icmp "eq" %110, %80 : i64
  llvm.cond_br %111, ^bb24, ^bb19(%110 : i64) {loop_annotation = #loop_annotation}
^bb24:  // 2 preds: ^bb17, ^bb23
  %112 = llvm.add %28, %11 overflow<nsw, nuw>  : i64
  %113 = llvm.icmp "eq" %112, %27 : i64
  llvm.cond_br %113, ^bb25, ^bb2(%112 : i64) {loop_annotation = #loop_annotation}
^bb25:  // 2 preds: ^bb0, ^bb24
  %114 = llvm.icmp "eq" %17, %15 : !llvm.ptr
  llvm.cond_br %114, ^bb27, ^bb26
^bb26:  // pred: ^bb25
  llvm.call @free(%17) : (!llvm.ptr) -> ()
  llvm.br ^bb27
^bb27:  // 2 preds: ^bb25, ^bb26
  %115 = llvm.load %21 {alignment = 8 : i64, tbaa = [#tbaa_tag5]} : !llvm.ptr -> !llvm.ptr
  %116 = llvm.icmp "eq" %115, %15 : !llvm.ptr
  llvm.cond_br %116, ^bb29, ^bb28
^bb28:  // pred: ^bb27
  llvm.call @free(%115) : (!llvm.ptr) -> ()
  llvm.br ^bb29
^bb29:  // 2 preds: ^bb27, ^bb28
  llvm.call @free(%18) : (!llvm.ptr) -> ()
  llvm.return
}
