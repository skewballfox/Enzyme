//===- AddToOpToIndexAndLoad.cpp - Lower Shadowed Gradient ops
//------------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower custom ops generated by the Enzyme AD
// procedure to the MemRef dialect.
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Rewrite/PatternApplicator.h"
#include "llvm/Support/raw_ostream.h"

#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "Utils.h"

using namespace mlir;
using namespace enzyme;
using llvm::errs;
namespace {

SmallVector<Value> applyAffineMap(AffineMap aMap, SmallVector<Value> indices,
                                  OpBuilder &builder, Location loc) {
  SmallVector<Value> appliedAffineMap;
  for (unsigned int i = 0; i < aMap.getNumResults(); i++) {
    AffineMap subMap = aMap.getSubMap({i});
    auto mapApplied =
        builder.create<affine::AffineApplyOp>(loc, subMap, ValueRange(indices));
    appliedAffineMap.push_back(mapApplied);
  }
  return appliedAffineMap;
}

struct AddToOpToIndexAndLoadPass
    : public enzyme::AddToOpToIndexAndLoadPassBase<AddToOpToIndexAndLoadPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionPatternRewriter rewriter(context);

    getOperation()->walk([&](Operation *op) {
      auto loc = op->getLoc();
      auto enzymeAdjoint = dyn_cast<enzyme::GenericAdjointOp>(op);
      if (!enzymeAdjoint)
        return;

      OpBuilder cacheBuilder(enzymeAdjoint);
      auto adjoint = Utils::adjointToGeneric(enzymeAdjoint, cacheBuilder, loc);

      // check if adjoint contains a enzyme.addToOp
      Operation *addToOp = nullptr;
      adjoint.walk([&](Operation *op) {
        if (isa<enzyme::AddToOp>(op)) {
          addToOp = op;
        }
      });
      if (!addToOp)
        return;

      Operation *terminator = adjoint.getBodyRegion().front().getTerminator();
      SmallVector<Value> indices;
      SmallVector<Value> retargs;
      auto outs = adjoint.getOutputs();
      auto num_ins = adjoint.getInputs().size();
      for (auto val : addToOp->getOperands()) {
        retargs.push_back(val);
      }
      auto map = adjoint.getIndexingMapsArray();
      cacheBuilder.setInsertionPoint(terminator);

      // Is it a fine assumption that all indexing maps are the same?
      for (int i = 0; i < map[0].getNumDims(); i++) {
        indices.push_back(cacheBuilder.create<linalg::IndexOp>(loc, i));
      }

      SmallVector<Value> rets;
      for (int i = 0; i < retargs.size(); i++) {
        // auto load = cacheBuilder.create<AffineLoadOp>(loc, inputs[i], map[i],
        // indices); auto store = cacheBuilder.create<AffineStoreOp>(loc, load,
        // inputs[i], map[i], indices);
        ValueRange mapAppliedIndices =
            applyAffineMap(map[num_ins + i], indices, cacheBuilder, loc);
        auto load = cacheBuilder.create<memref::LoadOp>(loc, outs[i],
                                                        mapAppliedIndices);
        auto added = cast<enzyme::AutoDiffTypeInterface>(load.getType())
                         .createAddOp(cacheBuilder, loc, load, retargs[i]);
        cacheBuilder.create<memref::StoreOp>(loc, added, outs[i],
                                             mapAppliedIndices);
      }

      for (int i = 0; i < retargs.size(); i++) {
        ValueRange mapAppliedIndices =
            applyAffineMap(map[num_ins + i], indices, cacheBuilder, loc);
        auto load = cacheBuilder.create<memref::LoadOp>(loc, outs[i],
                                                        mapAppliedIndices);
        retargs[i] = load;
      }

      cacheBuilder.create<linalg::YieldOp>(loc, ValueRange{retargs});
      addToOp->erase();
    });
  };
};
} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createAddToOpToIndexAndLoadPass() {
  return std::make_unique<AddToOpToIndexAndLoadPass>();
}
} // namespace enzyme
} // namespace mlir
