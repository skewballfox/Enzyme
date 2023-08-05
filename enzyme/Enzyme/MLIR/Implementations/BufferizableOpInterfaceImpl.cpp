//===- BufferizableOpInterfaceImpl.h - Impl registrations -* C++ -*--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of
// BufferizableOpInterface for enzyme dialect ops
//
//===----------------------------------------------------------------------===//

#include "BufferizableOpInterfaceImpl.h"
#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace mlir::bufferization;
using llvm::errs;

/// An !enzyme.Gradient<tensor<>> is effectively a memref. `enzyme.get` thus
/// bufferizes to a no-op/cast, while `enzyme.set` bufferizes to a copy.
struct GetOpInterface
    : public BufferizableOpInterface::ExternalModel<GetOpInterface,
                                                    enzyme::GetOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    auto getOp = cast<enzyme::GetOp>(op);
    return {getOp->getResult(opOperand.getOperandNumber())};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto getOp = cast<enzyme::GetOp>(op);
    Type resultType = getMemRefTypeWithStaticIdentityLayout(
        cast<TensorType>(getOp.getResult().getType()));
    Value memref = rewriter
                       .create<UnrealizedConversionCastOp>(
                           op->getLoc(), resultType, getOp.getGradient())
                       .getResult(0);
    rewriter.replaceOpWithNewOp<ToTensorOp>(op, memref);
    return success();
  }
};

struct SetOpInterface
    : public BufferizableOpInterface::ExternalModel<SetOpInterface,
                                                    enzyme::SetOp> {
  // Bufferizes to a copy, which both reads from and writes to memory.
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(Operation *op, OpOperand &opOperand,
                                            const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto setOp = cast<enzyme::SetOp>(op);
    auto bufferValue = getBuffer(rewriter, setOp.getValue(), options);
    if (failed(bufferValue)) {
      return failure();
    }

    Type baseType =
        cast<enzyme::GradientType>(setOp.getGradient().getType()).getBasetype();
    Type shadowType =
        getMemRefTypeWithStaticIdentityLayout(cast<TensorType>(baseType));
    Value shadow = rewriter
                       .create<UnrealizedConversionCastOp>(
                           op->getLoc(), shadowType, setOp.getGradient())
                       .getResult(0);
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, *bufferValue, shadow);
    return success();
  }
};

void mlir::enzyme::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, enzyme::EnzymeDialect *dialect) {
    GetOp::attachInterface<GetOpInterface>(*ctx);
    SetOp::attachInterface<SetOpInterface>(*ctx);
  });
}
