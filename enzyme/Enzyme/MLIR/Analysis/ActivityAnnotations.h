#ifndef ENZYME_MLIR_ANALYSIS_ACTIVITYANNOTATIONS_H
#define ENZYME_MLIR_ANALYSIS_ACTIVITYANNOTATIONS_H

#include "AliasAnalysis.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir {
class FunctionOpInterface;

namespace enzyme {

//===----------------------------------------------------------------------===//
// ValueOriginsLattice
//===----------------------------------------------------------------------===//

class ValueOriginsLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;
  ValueOriginsLattice(Value value, AliasClassSet &&origins)
      : dataflow::AbstractSparseLattice(value), origins(std::move(origins)) {}

  static AliasClassLattice single(Value point, DistinctAttr value) {
    return AliasClassLattice(point, AliasClassSet(value));
  }

  void print(raw_ostream &os) const override;

  ChangeResult join(const AbstractSparseLattice &other) override;

  ChangeResult insert(const DenseSet<DistinctAttr> &classes) {
    return origins.insert(classes);
  }

  ChangeResult markUnknown() { return origins.markUnknown(); }

  bool isUnknown() const { return origins.isUnknown(); }

  bool isUndefined() const { return origins.isUndefined(); }

  const DenseSet<DistinctAttr> &getOrigins() const {
    return origins.getAliasClasses();
  }

  const AliasClassSet &getOriginsObject() const { return origins; }

private:
  // TODO: The AliasClassSet data structure is exactly what we want here, the
  // distinct attributes represent value origins instead of alias classes.
  AliasClassSet origins;
};

// TODO: do we need a backwards activity annotation analysis?
class ForwardActivityAnnotationAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<ValueOriginsLattice> {
public:
  ForwardActivityAnnotationAnalysis(DataFlowSolver &solver)
      : SparseForwardDataFlowAnalysis(solver) {
    assert(!solver.getConfig().isInterprocedural());
  }

  void setToEntryState(ValueOriginsLattice *lattice) override;

  void visitOperation(Operation *op,
                      ArrayRef<const ValueOriginsLattice *> operands,
                      ArrayRef<ValueOriginsLattice *> results) override;

  void visitExternalCall(CallOpInterface call,
                         ArrayRef<const ValueOriginsLattice *> operands,
                         ArrayRef<ValueOriginsLattice *> results) override;

private:
  OriginalClasses originalClasses;
};

//===----------------------------------------------------------------------===//
// ValueOriginsMap
//===----------------------------------------------------------------------===//

class ValueOriginsMap : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  void print(raw_ostream &os) const override;

  ChangeResult join(const AbstractDenseLattice &other) override;

  /// Mark the pointer stored in `dest` as originating from all of `origins`.
  ChangeResult insert(const AliasClassSet &destClasses,
                      const AliasClassSet &origins);

  ChangeResult markAllOriginsUnknown();

  ChangeResult joinPotentiallyMissing(DistinctAttr key,
                                      const AliasClassSet &value);

  const AliasClassSet &getOrigins(DistinctAttr id) const {
    auto it = valueOrigins.find(id);
    if (it == valueOrigins.end())
      return AliasClassSet::getUndefined();
    return it->getSecond();
  }

private:
  // Represents "this alias class has a differential dependency originating from
  // this value"
  // TODO: Don't get confused because they're both distinct attributes, the keys
  // are exclusively alias classes and the values are sets of value origins
  DenseMap<DistinctAttr, AliasClassSet> valueOrigins;
};

//===----------------------------------------------------------------------===//
// DenseActivityAnnotationAnalysis
//===----------------------------------------------------------------------===//

class DenseActivityAnnotationAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<ValueOriginsMap> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  void setToEntryState(ValueOriginsMap *lattice) override;

  void visitOperation(Operation *op, const ValueOriginsMap &before,
                      ValueOriginsMap *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const ValueOriginsMap &before,
                                    ValueOriginsMap *after) override;

private:
  OriginalClasses originalClasses;
};

void runActivityAnnotations(FunctionOpInterface callee);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_ACTIVITYANNOTATIONS_H