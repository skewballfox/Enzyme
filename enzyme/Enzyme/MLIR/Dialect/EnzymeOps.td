//===- EnzymeOps.td - Enzyme dialect ops ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//include "mlir/Dialect/Linalg/IR/LinalgBase.td"

#ifndef ENZYME_OPS
#define ENZYME_OPS

include "Dialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/IR/AttrTypeBase.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Activity : I32EnumAttr<"Activity",
    "Possible activity states for variables",
    [
      I32EnumAttrCase<"enzyme_out",  0>,
      I32EnumAttrCase<"enzyme_dup",  1>,
      I32EnumAttrCase<"enzyme_const",2>,
      I32EnumAttrCase<"enzyme_dupnoneed", 3>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzyme";
}

def ActivityAttr : EnumAttr<Enzyme_Dialect, Activity, "activity">;

def ActivityArrayAttr : TypedArrayAttrBase<
    ActivityAttr, "Array of argument activity states">;

def PlaceholderOp : Enzyme_Op<"placeholder",
    [Pure]> {
  let results = (outs AnyType:$output);
}

def ForwardDiffOp : Enzyme_Op<"fwddiff",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Perform forward mode AD on a funcop";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def AutoDiffOp : Enzyme_Op<"autodiff",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Perform reverse mode AD on a funcop";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def Cache : Enzyme_Type<"Cache"> {
  let summary = "Cache for reverse pass";
  let description = [{
    "Cache for reverse pass"
  }];
  let parameters = (ins "Type":$elementType);
  let mnemonic = "Cache";
  let assemblyFormat = "`<` $elementType `>`";
}

def ShadowedGradient : Enzyme_Type<"ShadowedGradient"> {
  let summary = "Stores gradients which need to be initialized with shadow values from the forward pass.";
  let description = [{
    "Cache for reverse pass"
  }];
  let parameters = (ins "Type":$basetype);
  let mnemonic = "ShadowedGradient";
  let assemblyFormat = "`<` $basetype `>`";
}

def PushOp : Enzyme_Op<"push"> {
  let summary = "Push value to cache or ShadowedGradient";
  let arguments = (ins AnyTypeOf<[Cache, ShadowedGradient]>:$cache, AnyType:$value);

  let assemblyFormat = [{ $value `,` $cache attr-dict `:` type($value) `->` type($cache) }];
}

def PopOp : Enzyme_Op<"pop"> {
  let summary = "Retrieve information for the reverse mode pass.";
  let arguments = (ins AnyTypeOf<[Cache, ShadowedGradient]>:$cache);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{ $cache attr-dict `:` type($cache) `->` type($result) }];
}

def ClearOp : Enzyme_Op<"clear"> {
  let summary = "Remove top element from ShadowedGradient";
  let arguments = (ins AnyType : $cache);
  let results = (outs );
}

def InitOp : Enzyme_Op<"init"> {
  let summary = "Creat enzyme.gradient and enzyme.cache";
  let results = (outs AnyType:$result);

  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def Gradient : Enzyme_Type<"Gradient"> {
  let summary = "Stores gradient if it cant be stroed in a value.";
  let description = [{
    "Cache for reverse pass"
  }];
  let parameters = (ins "Type":$baseType);
  let mnemonic = "Gradient";
  let assemblyFormat = "`<` $baseType `>`";
}

def SetOp : Enzyme_Op<"set",
     [TypesMatchWith<"result type matches element type of 'gradient'",
                     "gradient", "value",
                     "$_self.cast<GradientType>().getBaseType()">]> {
  let summary = "Write to gradient";
  let arguments = (ins Gradient:$gradient, AnyType:$value);
  let assemblyFormat = [{ $value `,` $gradient attr-dict `:` type($gradient) }];
}

def GetOp : Enzyme_Op<"get",
     [TypesMatchWith<"result type matches element type of 'gradient'",
                     "gradient", "result",
                     "$_self.cast<GradientType>().getBaseType()">]> {
  let summary = "Load value of gradient";
  let arguments = (ins Gradient:$gradient);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{ $gradient attr-dict `:` type($gradient) }];
}

def AddToOp : Enzyme_Op<"addTo", [Pure, Terminator, ReturnLike]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Linalg add to operation";
  let description = [{
    TODO
  }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let hasCustomAssemblyFormat = 0;
  let hasVerifier = 0;
}

def GenericAdjointOp : Enzyme_Op<"genericAdjoint", [AttrSizedOperandSegments]> {
  let description = [{ }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       AffineMapArrayAttr:$indexing_maps,
                       ArrayAttr:$iterator_types,
                       OptionalAttr<StrAttr>:$doc,
                       OptionalAttr<StrAttr>:$library_call);
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  let regions = (region AnyRegion:$region);
  
}

#endif // ENZYME_OPS
