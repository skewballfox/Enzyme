//===- ActivityAnalysis.cpp - Implementation of Activity Analysis  -----------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning, Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file consists of two mutually recurive
// functions that compute this for values and instructions, respectively.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/IR/InstIterator.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/InlineAsm.h"

#include "ActivityAnalysis.h"
#include "Utils.h"

#include "TypeAnalysis/TBAA.h"

using namespace llvm;

cl::opt<bool> printconst("enzyme_printconst", cl::init(false), cl::Hidden,
                         cl::desc("Print constant detection algorithm"));

cl::opt<bool> nonmarkedglobals_inactive(
    "enzyme_nonmarkedglobals_inactive", cl::init(false), cl::Hidden,
    cl::desc("Consider all nonmarked globals to be inactive"));

cl::opt<bool> ipoconst("enzyme_ipoconst", cl::init(false), cl::Hidden,
                       cl::desc("Interprocedural constant detection"));

cl::opt<bool> emptyfnconst("enzyme_emptyfnconst", cl::init(false), cl::Hidden,
                           cl::desc("Empty functions are considered constant"));

#include "llvm/IR/InstIterator.h"
#include <map>
#include <set>
#include <unordered_map>


constexpr uint8_t UP = 1;
constexpr uint8_t DOWN = 2;

bool ActivityAnalyzer::isFunctionArgumentConstant(CallInst *CI, Value *val) {
  Function *F = CI->getCalledFunction();
  if (F == nullptr)
    return false;

  auto fn = F->getName();
  // todo realloc consider?
  // For known library functions, special case how derivatives flow to allow for
  // more aggressive active variable detection
  if (fn == "malloc" || fn == "free" || fn == "_Znwm" ||
      fn == "__cxa_guard_acquire" || fn == "__cxa_guard_release" ||
      fn == "__cxa_guard_abort")
    return true;
  if (F->getIntrinsicID() == Intrinsic::memset && CI->getArgOperand(0) != val &&
      CI->getArgOperand(1) != val)
    return true;
  if (F->getIntrinsicID() == Intrinsic::memcpy && CI->getArgOperand(0) != val &&
      CI->getArgOperand(1) != val)
    return true;
  if (F->getIntrinsicID() == Intrinsic::memmove &&
      CI->getArgOperand(0) != val && CI->getArgOperand(1) != val)
    return true;

  if (F->empty())
    return false;

  // return false;
  if (fn.startswith("augmented"))
    return false;
  if (fn.startswith("fakeaugmented"))
    return false;
  if (fn.startswith("diffe"))
    return false;
  // if (val->getType()->isPointerTy()) return false;
  if (!val->getType()->isIntOrIntVectorTy())
    return false;

  //assert(retvals.find(val) == retvals.end());

  // TODO need to fixup the below, it currently is incorrect, but didn't have
  // time to fix rn
  return false;

  #if 0
  // static std::unordered_map<std::tuple<Function*, Value*,
  // SmallPtrSet<Value*,20>, std::set<Value*> >, bool> metacache;
  static std::map<
      std::tuple<CallInst *, Value *, std::set<Value *>, std::set<Value *>,
                 std::set<Value *>, std::set<Value *>>,
      bool>
      metacache;
  // auto metatuple = std::make_tuple(F, val,
  // SmallPtrSet<Value*,20>(constants.begin(), constants.end()),
  // std::set<Value*>(nonconstant.begin(), nonconstant.end()));
  auto metatuple = std::make_tuple(
      CI, val, std::set<Value *>(constants.begin(), constants.end()),
      std::set<Value *>(nonconstant.begin(), nonconstant.end()),
      std::set<Value *>(constantvals.begin(), constantvals.end()),
      std::set<Value *>(retvals.begin(), retvals.end()));
  if (metacache.find(metatuple) != metacache.end()) {
    if (printconst)
      llvm::errs() << " < SUBFN metacache const " << F->getName()
                   << "> arg: " << *val << " ci:" << *CI << "\n";
    return metacache[metatuple];
  }
  if (printconst)
    llvm::errs() << " < METAINDUCTIVE SUBFN const " << F->getName()
                 << "> arg: " << *val << " ci:" << *CI << "\n";

  metacache[metatuple] = true;
  // Note that the base case of true broke the up/down variant so have to be
  // very conservative
  //  as a consequence we cannot detect const of recursive functions :'( [in
  //  that will be too conservative]
  // metacache[metatuple] = false;

  SmallPtrSet<Value *, 20> constants2;
  constants2.insert(constants.begin(), constants.end());
  SmallPtrSet<Value *, 20> nonconstant2;
  nonconstant2.insert(nonconstant.begin(), nonconstant.end());
  SmallPtrSet<Value *, 20> constantvals2;
  constantvals2.insert(constantvals.begin(), constantvals.end());
  SmallPtrSet<Value *, 20> retvals2;
  retvals2.insert(retvals.begin(), retvals.end());

  // Ask the question, even if is this is active, are all its uses inactive
  // (meaning this use does not impact its activity)
  nonconstant2.insert(val);
  // retvals2.insert(val);

  // constants2.insert(val);

  if (printconst) {
    llvm::errs() << " < SUBFN " << F->getName() << "> arg: " << *val
                 << " ci:" << *CI << "\n";
  }

  auto a = F->arg_begin();

  std::set<int> arg_constants;
  std::set<int> idx_findifactive;
  SmallPtrSet<Value *, 20> arg_findifactive;

  SmallPtrSet<Value *, 20> newconstants;
  SmallPtrSet<Value *, 20> newnonconstant;

  FnTypeInfo nextTypeInfo(F);
  int argnum = 0;
  for (auto &arg : F->args()) {
    nextTypeInfo.Arguments.insert(std::pair<Argument *, TypeTree>(
        &arg, TR.query(CI->getArgOperand(argnum))));
    ++argnum;
  }
  nextTypeInfo.Return = TR.query(CI);
  TypeResults TR2 = TR.analysis.analyzeFunction(nextTypeInfo);

  for (unsigned i = 0; i < CI->getNumArgOperands(); ++i) {
    if (CI->getArgOperand(i) == val) {
      arg_findifactive.insert(a);
      idx_findifactive.insert(i);
      newnonconstant.insert(a);
      ++a;
      continue;
    }

    if (isconstantValueM(TR, CI->getArgOperand(i), constants2, nonconstant2,
                         constantvals2, retvals2, AA),
        directions) {
      newconstants.insert(a);
      arg_constants.insert(i);
    } else {
      newnonconstant.insert(a);
    }
    ++a;
  }

  bool constret;

  // allow return index as valid entry as well
  if (CI != val) {
    constret = isconstantValueM(TR, CI, constants2, nonconstant2, constantvals2,
                                retvals2, AA, directions);
    if (constret)
      arg_constants.insert(-1);
  } else {
    constret = false;
    arg_findifactive.insert(a);
    idx_findifactive.insert(-1);
  }

  static std::map<std::tuple<std::set<int>, Function *, std::set<int>>, bool>
      cache;

  auto tuple = std::make_tuple(arg_constants, F, idx_findifactive);
  if (cache.find(tuple) != cache.end()) {
    if (printconst)
      llvm::errs() << " < SUBFN cache const " << F->getName()
                   << "> arg: " << *val << " ci:" << *CI << "\n";
    return cache[tuple];
  }

  //! inductively assume that it is constant, it should be deduced nonconstant
  //! elsewhere if this is not the case
  if (printconst)
    llvm::errs() << " < INDUCTIVE SUBFN const " << F->getName()
                 << "> arg: " << *val << " ci:" << *CI << "\n";

  cache[tuple] = true;
  // Note that the base case of true broke the up/down variant so have to be
  // very conservative
  //  as a consequence we cannot detect const of recursive functions :'( [in
  //  that will be too conservative]
  // cache[tuple] = false;

  SmallPtrSet<Value *, 4> newconstantvals;
  newconstantvals.insert(constantvals2.begin(), constantvals2.end());

  SmallPtrSet<Value *, 4> newretvals;
  newretvals.insert(retvals2.begin(), retvals2.end());

  for (llvm::inst_iterator I = llvm::inst_begin(F), E = llvm::inst_end(F);
       I != E; ++I) {
    if (auto ri = dyn_cast<ReturnInst>(&*I)) {
      if (!constret) {
        newretvals.insert(ri->getReturnValue());
        if (CI == val)
          arg_findifactive.insert(ri->getReturnValue());
      } else {
        // newconstantvals.insert(ri->getReturnValue());
      }
    }
  }

  for (auto specialarg : arg_findifactive) {
    for (auto user : specialarg->users()) {
      if (printconst)
        llvm::errs() << " going to consider user " << *user << "\n";
      if (!isconstantValueM(TR2, user, newconstants, newnonconstant,
                            newconstantvals, newretvals, AA, 3)) {
        if (printconst)
          llvm::errs() << " < SUBFN nonconst " << F->getName()
                       << "> arg: " << *val << " ci:" << *CI
                       << "  from sf: " << *user << "\n";
        metacache.erase(metatuple);
        return cache[tuple] = false;
      }
    }
  }

  constants.insert(constants2.begin(), constants2.end());
  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
  if (printconst) {
    llvm::errs() << " < SUBFN const " << F->getName() << "> arg: " << *val
                 << " ci:" << *CI << "\n";
  }
  metacache.erase(metatuple);
  return cache[tuple] = true;
  #endif
}

static bool
isGuaranteedConstantValue(TypeResults &TR, Value *val,
                          const SmallPtrSetImpl<Value *> *constantvals) {
  // This result of this instruction is certainly an integer (and only and
  // integer, not a pointer or float). Therefore its value is inactive
  // Note that this is correct, but not aggressive as it should be (we should
  // call isConstantValue(inst) here, but we need to be careful to not have an
  // infinite recursion)

  //  TODO: make this more aggressive
  if (TR.intType(val, /*errIfNotFound=*/false).isIntegral()) {
    if (printconst)
      llvm::errs() << " -> known integer " << *val << "\n";
    return true;
    // if we happen to have already deduced this instruction constant, we might
    // as well use the information
  } else if (constantvals && constantvals->find(val) != constantvals->end()) {
    if (printconst)
      llvm::errs() << " -> previous constant value " << *val << "\n";
    return true;
    // if we know the subtype contains no derivative information, we can assure
    // that this is a constant value
  } else if (val->getType()->isVoidTy() || val->getType()->isEmptyTy()) {
    return true;
  }
  return false;
}

// propagateFromOperand should consider the value that would make this call
// active and return true if it does make it active and thus
//   we don't need to consider any further operands
void propagateArgumentInformation(
    CallInst &CI, std::function<bool(Value *)> propagateFromOperand) {

  if (auto called = CI.getCalledFunction()) {
    auto n = called->getName();
    if (n == "lgamma" || n == "lgammaf" || n == "lgammal" || n == "lgamma_r" ||
        n == "lgammaf_r" || n == "lgammal_r" || n == "__lgamma_r_finite" ||
        n == "__lgammaf_r_finite" || n == "__lgammal_r_finite" || n == "tanh" ||
        n == "tanhf") {

      propagateFromOperand(CI.getArgOperand(0));
      return;
    }
  }

  for (auto &a : CI.arg_operands()) {
    if (propagateFromOperand(a))
      break;
  }
}

bool ActivityAnalyzer::isconstantM(TypeResults &TR, Instruction *inst) {
  assert(inst);
  assert(TR.info.Function == inst->getParent()->getParent());
  if (isa<ReturnInst>(inst))
    return true;

  if (isa<UnreachableInst>(inst) || isa<BranchInst>(inst) ||
      (constants.find(inst) != constants.end())) {
    return true;
  }

  if ((nonconstant.find(inst) != nonconstant.end())) {
    return false;
  }

  if (auto call = dyn_cast<CallInst>(inst)) {
    #if LLVM_VERSION_MAJOR >= 11
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand())) {
    #else
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledValue())) {
    #endif
      if (iasm->getAsmString() == "cpuid") {
        if (printconst)
          llvm::errs() << " constant instruction from known cpuid instruction "
                       << *inst << "\n";
        constants.insert(inst);
        return true;
      }
    }
  }

  // If this instruction does not write memory to memory that outlives itself
  // (therefore propagating derivative information), and the return value of
  // this instruction is known to be inactive this instruction is inactive as it
  // cannot propagate derivative information
  if (isGuaranteedConstantValue(TR, inst,
                                directions == 3 ? &constantvals : nullptr)) {
    if (!inst->mayWriteToMemory() ||
        (isa<CallInst>(inst) && AA.onlyReadsMemory(cast<CallInst>(inst)))) {
      if (printconst)
        llvm::errs() << " constant instruction from known constant non-writing "
                        "instruction "
                     << *inst << "\n";
      constants.insert(inst);
      return true;
    } else {
      if (printconst)
        llvm::errs() << " may be active inst as could write to memory " << *inst
                     << "\n";
    }
  }

  if (auto op = dyn_cast<CallInst>(inst)) {
    if (auto called = op->getCalledFunction()) {
      if (called->getName() == "printf" || called->getName() == "puts" ||
          called->getName() == "__assert_fail" || called->getName() == "free" ||
          called->getName() == "_ZdlPv" || called->getName() == "_ZdlPvm" ||
          called->getName() == "__cxa_guard_acquire" ||
          called->getName() == "__cxa_guard_release" ||
          called->getName() == "__cxa_guard_abort") {
        constants.insert(inst);
        return true;
      }
      if (!isCertainPrintMallocOrFree(called) && called->empty() &&
          !hasMetadata(called, "enzyme_gradient") && !isa<IntrinsicInst>(op) &&
          emptyfnconst) {
        constants.insert(inst);
        return true;
      }
    }
  }

  if (auto op = dyn_cast<IntrinsicInst>(inst)) {
    switch (op->getIntrinsicID()) {
    case Intrinsic::assume:
    case Intrinsic::stacksave:
    case Intrinsic::stackrestore:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::dbg_addr:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::var_annotation:
    case Intrinsic::ptr_annotation:
    case Intrinsic::annotation:
    case Intrinsic::codeview_annotation:
    case Intrinsic::expect:
    case Intrinsic::type_test:
    case Intrinsic::donothing:
      // case Intrinsic::is_constant:
      constants.insert(inst);
      return true;
    default:
      break;
    }
  }

  if (isa<SIToFPInst>(inst) || isa<UIToFPInst>(inst) || isa<FPToSIInst>(inst) ||
      isa<FPToUIInst>(inst)) {
    constants.insert(inst);
    return true;
  }

  if (auto storeinst = dyn_cast<StoreInst>(inst)) {
    auto storeSize =
        storeinst->getParent()
            ->getParent()
            ->getParent()
            ->getDataLayout()
            .getTypeSizeInBits(storeinst->getValueOperand()->getType()) /
        8;

    bool allIntegral = true;
    bool anIntegral = false;
    auto q = TR.query(storeinst->getPointerOperand()).Data0();
    for (int i = -1; i < (int)storeSize; ++i) {
      auto dt = q[{i}];
      if (dt.isIntegral() || dt == BaseType::Anything) {
        anIntegral = true;
      } else if (dt.isKnown()) {
        allIntegral = false;
        break;
      }
    }

    if (allIntegral && anIntegral) {
      if (printconst)
        llvm::errs() << " constant instruction from TBAA " << *inst << "\n";
      constants.insert(inst);
      return true;
    }
  }

  if (directions & UP) {
    if (auto li = dyn_cast<LoadInst>(inst)) {
      if (constantvals.find(li->getPointerOperand()) != constantvals.end() ||
          constants.find(li->getPointerOperand()) != constants.end()) {
        constants.insert(li);
        constantvals.insert(li);
        return true;
      }
    }
    if (auto rmw = dyn_cast<AtomicRMWInst>(inst)) {
      if (constantvals.find(rmw->getPointerOperand()) != constantvals.end() ||
          constants.find(rmw->getPointerOperand()) != constants.end()) {
        constants.insert(rmw);
        constantvals.insert(rmw);
        return true;
      }
    }
    if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
      if (constantvals.find(gep->getPointerOperand()) != constantvals.end() ||
          constants.find(gep->getPointerOperand()) != constants.end()) {
        constants.insert(gep);
        constantvals.insert(gep);
        return true;
      }
    }
    if (auto cst = dyn_cast<CastInst>(inst)) {
      if (constantvals.find(cst->getOperand(0)) != constantvals.end() ||
          constants.find(cst->getOperand(0)) != constants.end()) {
        constants.insert(cst);
        constantvals.insert(cst);
        return true;
      }
    }
  }

  if (printconst)
    llvm::errs() << "checking if is constant[" << (int)directions << "] "
                 << *inst << "\n";

  // For specific instructions, if for some reason or another we know that the
  // value is a constant, this pointer instruction must be constant, indepedent
  // of whether it could be memory or not
  if (constantvals.find(inst) != constantvals.end() && UP) {
    if (isa<CastInst>(inst) || isa<PHINode>(inst) || isa<SelectInst>(inst)) {
      if (printconst)
        llvm::errs() << "constant value becomes constant instruction " << *inst
                     << "\n";
      constants.insert(inst);
      return true;
    }
  }

  // Handle types that could contain pointers
  //  Consider all types except
  //   * floating point types (since those are assumed not pointers)
  //   * integers that we know are not pointers
  bool containsPointer = true;
  if (inst->getType()->isFPOrFPVectorTy())
    containsPointer = false;
  if (!TR.intType(inst, /*errIfNotFound*/ false).isPossiblePointer())
    containsPointer = false;

  if (containsPointer) {

    // If the value returned here is a pointer and active, we will fallback
    // and always assume this instruction is active even if it has either no
    // active operands (say because it's a malloc).
    // If this has no active uses, the value itself should already be proven
    // inactive
    if (!isconstantValueM(TR, inst)) {
      nonconstant.insert(inst);
      if (printconst)
        llvm::errs() << "memory(" << (int)directions << ")  ret: " << *inst
                     << "\n";
      return false;
    }
  }

  // TODO consider propagating tmps
  if (directions & UP) {
    if (printconst)
      llvm::errs() << " < UPSEARCH" << (int)directions << ">" << *inst
                    << "\n";

    ActivityAnalyzer Hypothesis(*this, UP);
    Hypothesis.constants.insert(inst);

    if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
      if (Hypothesis.isconstantValueM(TR, gep->getPointerOperand())) {
        insertConstantsFrom(Hypothesis);
        if (printconst)
          llvm::errs() << "constant(" << (int)directions << ") up-gep "
                        << *inst << "\n";
        return true;
      }

    } else if (auto ci = dyn_cast<CallInst>(inst)) {
      bool seenuse = false;

      propagateArgumentInformation(*ci, [&](Value *a) {
        if (!Hypothesis.isconstantValueM(TR, a)) {
          seenuse = true;
          if (printconst)
            llvm::errs() << "nonconstant(" << (int)directions << ")  up-call "
                          << *inst << " op " << *a << "\n";
          return true;
          /*
          if (directions == 3)
            nonconstant.insert(inst);
          if (printconst)
            llvm::errs() << "nonconstant(" << (int)directions << ")  call " <<
          *inst << " op " << *a << "\n";
          //return false;
          break;
          */
        }
        return false;
      });

      //! TODO consider calling interprocedural here
      //! TODO: Really need an attribute that determines whether a function
      //! can access a global (not even necessarily read)
      // if (ci->hasFnAttr(Attribute::ReadNone) ||
      // ci->hasFnAttr(Attribute::ArgMemOnly))
      if (!seenuse) {
        insertConstantsFrom(Hypothesis);
        // constants.insert(constants_tmp.begin(), constants_tmp.end());
        // if (directions == 3)
        //  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
        if (printconst)
          llvm::errs() << "constant(" << (int)directions
                        << ")  up-call:" << *inst << "\n";
        return true;
      }
    } else if (auto si = dyn_cast<StoreInst>(inst)) {

      if (Hypothesis.isconstantValueM(TR, si->getPointerOperand())) {
        // Note: not adding nonconstant here since if had full updown might
        // not have been nonconstant
        insertConstantsFrom(Hypothesis);

        if (printconst)
          llvm::errs() << "constant(" << (int)directions
                        << ") up-store:" << *inst << "\n";
        return true;
      }

      /* TODO consider stores of constant values
      if (isconstantValueM(TR, (TR, si->getValueOperand(), constants2,
      nonconstant2, constantvals2, retvals2, originalInstructions,
      directions)) { constants.insert(inst);
          constants.insert(constants2.begin(), constants2.end());
          constants.insert(constants_tmp.begin(), constants_tmp.end());

          // not here since if had full updown might not have been nonconstant
          //nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
          if (printconst)
            llvm::errs() << "constant(" << (int)directions << ") store:" <<
      *inst << "\n"; return true;
      }
      */
    } else if (auto si = dyn_cast<SelectInst>(inst)) {

      if ( Hypothesis.isconstantValueM(TR, si->getTrueValue()) &&
            Hypothesis.isconstantValueM(TR, si->getFalseValue())) {
        // Note: not adding nonconstant here since if had full updown might
        // not have been nonconstant
        insertConstantsFrom(Hypothesis);

        if (printconst)
          llvm::errs() << "constant(" << (int)directions
                        << ") up-sel:" << *inst << "\n";
        return true;
      }

    } else {
      bool seenuse = false;

      for (auto &a : inst->operands()) {
        bool hypval = Hypothesis.isconstantValueM(TR, a);
        //llvm::errs() << " checking op a=" << *a << " of inst " << inst << " constant=" << hypval << " hypot.dir=" << (size_t)Hypothesis.directions << " mydir=" << (size_t)directions << "\n";
        if (!hypval) {
          // if (directions == 3)
          //  nonconstant.insert(inst);
          if (printconst)
            llvm::errs() << "nonconstant(" << (int)directions << ")  up-inst "
                          << *inst << " op " << *a << "\n";
          seenuse = true;
          break;
          // return false;
        }
      }

      if (!seenuse) {
        // Note: not adding nonconstant here since if had full updown might
        // not have been nonconstant
        insertConstantsFrom(Hypothesis);

        if (printconst)
          llvm::errs() << "constant(" << (int)directions
                        << ")  up-inst:" << *inst << "\n";
        return true;
      }
    }
    if (printconst)
      llvm::errs() << " </UPSEARCH" << (int)directions << ">" << *inst
                    << "\n";
  }

  // TODO use typeInfo for more aggressive activity analysis
  if (!containsPointer && (!inst->mayWriteToMemory()) && (directions & DOWN) &&
      (retvals.find(inst) == retvals.end())) {

    // Proceed assuming this is constant, can we prove this should be constant
    // otherwise
    ActivityAnalyzer Hypothesis(*this, DOWN);
    Hypothesis.constants.insert(inst);

    if (printconst)
      llvm::errs() << " < USESEARCH" << (int)directions << ">" << *inst << "\n";

    assert(!inst->mayWriteToMemory());
    assert(!isa<StoreInst>(inst));
    bool seenuse = false;
    for (const auto a : inst->users()) {
      if (auto gep = dyn_cast<GetElementPtrInst>(a)) {
        assert(inst != gep->getPointerOperand());
        continue;
      }
      if (isa<AllocaInst>(a)) {
        if (printconst)
          llvm::errs() << "found constant(" << (int)directions
                       << ")  allocainst use:" << *inst << " user " << *a
                       << "\n";
        continue;
      }

      if (auto call = dyn_cast<CallInst>(a)) {
        if (Hypothesis.isFunctionArgumentConstant(call, inst)) {
          if (printconst)
            llvm::errs() << "found constant(" << (int)directions
                         << ")  callinst use:" << *inst << " user " << *a
                         << "\n";
          continue;
        } else {
          if (printconst)
            llvm::errs() << "found seminonconstant(" << (int)directions
                         << ")  callinst use:" << *inst << " user " << *a
                         << "\n";
          // seenuse = true;
          // break;
        }
      }

      if (!Hypothesis.isconstantM(TR, cast<Instruction>(a))) {
        if (printconst)
          llvm::errs() << "nonconstant(" << (int)directions
                       << ") inst (uses):" << *inst << " user " << *a << " "
                       << &seenuse << "\n";
        seenuse = true;
        break;
      } else {
        if (printconst)
          llvm::errs() << "found constant(" << (int)directions
                       << ")  inst use:" << *inst << " user " << *a << "\n";
      }
    }

    if (!seenuse) {
      insertConstantsFrom(Hypothesis);
      if (printconst)
        llvm::errs() << "constant(" << (int)directions
                     << ") inst (uses):" << *inst << "   seenuse:" << &seenuse
                     << "\n";
      return true;
    }

    if (printconst)
      llvm::errs() << " </USESEARCH" << (int)directions << ">" << *inst << "\n";
  }

  if (directions == 3)
    nonconstant.insert(inst);
  if (printconst)
    llvm::errs() << "couldnt decide nonconstants(" << (int)directions
                 << "):" << *inst << "\n";
  return false;
}

bool ActivityAnalyzer::isconstantValueM(TypeResults &TR, Value *val) {
  assert(val);
  if (auto inst = dyn_cast<Instruction>(val)) {
    assert(TR.info.Function == inst->getParent()->getParent());
  }
  if (auto arg = dyn_cast<Argument>(val)) {
    assert(TR.info.Function == arg->getParent());
  }
  // assert(directions >= 0);
  assert(directions <= 3);

  // llvm::errs() << "looking for: " << *val << "\n";
  if (val->getType()->isVoidTy())
    return true;

  //! False so we can replace function with augmentation
  if (isa<Function>(val)) {
    return false;
  }

  if (isa<UndefValue>(val) || isa<MetadataAsValue>(val))
    return true;

  if (isa<ConstantData>(val) || isa<ConstantAggregate>(val)) {
    if (printconst)
      llvm::errs() << " VALUE const as constdata: " << *val << "\n";
    return true;
  }
  if (isa<BasicBlock>(val))
    return true;
  assert(!isa<InlineAsm>(val));

  if (constants.find(val) != constants.end()) {
    return true;
  }

  if (constantvals.find(val) != constantvals.end()) {
    // if (printconst)
    //    llvm::errs() << " VALUE const from precomputation " << *val << "\n";
    return true;
  }
  if (retvals.find(val) != retvals.end()) {
    return false;
  }

  // All arguments should be marked constant/nonconstant ahead of time
  if (isa<Argument>(val)) {
    if ((nonconstant.find(val) != nonconstant.end())) {
      if (printconst)
        llvm::errs() << " VALUE nonconst from arg nonconst " << *val << "\n";
      return false;
    }
    llvm::errs() << *(cast<Argument>(val)->getParent()) << "\n";
    llvm::errs() << *val << "\n";
    assert(0 && "must've put arguments in constant/nonconstant");
  }

  // This value is certainly an integer (and only and integer, not a pointer or
  // float). Therefore its value is constant
  //llvm::errs() << "VC: " << *val << " TR: " << TR.intType(val, false).str() << "\n";
  if (TR.intType(val, /*errIfNotFound*/ false).isIntegral()) {
    if (printconst)
      llvm::errs() << " Value const as integral " << (int)directions << " "
                   << *val << " "
                   << TR.intType(val, /*errIfNotFound*/ false).str() << "\n";
    constantvals.insert(val);
    return true;
  }

  // This value is certainly a pointer to an integer (and only and integer, not
  // a pointer or float). Therefore its value is constant
  // TODO use typeInfo for more aggressive activity analysis
  if (val->getType()->isPointerTy() &&
      cast<PointerType>(val->getType())->isIntOrIntVectorTy() &&
      TR.firstPointer(1, val, /*errifnotfound*/ false).isIntegral()) {
    if (printconst)
      llvm::errs() << " Value const as integral pointer" << (int)directions
                   << " " << *val << "\n";
    constantvals.insert(val);
    return true;
  }

  if (auto gi = dyn_cast<GlobalVariable>(val)) {
    if (!hasMetadata(gi, "enzyme_shadow") && nonmarkedglobals_inactive) {
      constantvals.insert(val);
      gi->setMetadata("enzyme_activity_value",
                      MDNode::get(gi->getContext(),
                                  MDString::get(gi->getContext(), "const")));
      return true;
    }
    // TODO consider this more
    if (gi->isConstant() &&
        isconstantValueM(TR, gi->getInitializer())) {
      constantvals.insert(val);
      gi->setMetadata("enzyme_activity_value",
                      MDNode::get(gi->getContext(),
                                  MDString::get(gi->getContext(), "const")));
      if (printconst)
        llvm::errs() << " VALUE const global " << *val << "\n";
      return true;
    }
    auto res = TR.query(gi).Data0();
    auto dt = res[{-1}];
    dt |= res[{0}];
    if (dt.isIntegral()) {
      if (printconst)
        llvm::errs() << " VALUE const as global int pointer " << *val
                     << " type - " << res.str() << "\n";
      return true;
    }
    if (printconst)
      llvm::errs() << " VALUE nonconst unknown global " << *val << " type - "
                   << res.str() << "\n";
    return false;
  }

  if (auto ce = dyn_cast<ConstantExpr>(val)) {
    if (ce->isCast()) {
      if (isconstantValueM(TR, ce->getOperand(0))) {
        if (printconst)
          llvm::errs() << " VALUE const cast from from operand " << *val
                       << "\n";
        constantvals.insert(val);
        return true;
      }
    }
    if (ce->isGEPWithNoNotionalOverIndexing()) {
      if (isconstantValueM(TR, ce->getOperand(0))) {
        if (printconst)
          llvm::errs() << " VALUE const cast from gep operand " << *val << "\n";
        constantvals.insert(val);
        return true;
      }
    }
    if (printconst)
      llvm::errs() << " VALUE nonconst unknown expr " << *val << "\n";
    return false;
  }


  // Handle types that could contain pointers
  //  Consider all types except
  //   * floating point types (since those are assumed not pointers)
  //   * integers that we know are not pointers
  bool containsPointer = true;
  if (val->getType()->isFPOrFPVectorTy())
    containsPointer = false;
  if (!TR.intType(val, /*errIfNotFound*/ false).isPossiblePointer())
    containsPointer = false;

  if (containsPointer) {

    if (printconst)
      llvm::errs() << " < MEMSEARCH" << (int)directions << ">" << *val << "\n";
    // A pointer value is active if two things hold:
    //   an potentially active value is stored into the memory
    //   memory loaded from the value is used in an active way
    bool potentiallyActiveStore = false;
    bool potentiallyActiveLoad = false;

    // Assume the value (not instruction) is itself active
    // In spite of that can we show that there are either no active stores
    // or no active loads
    ActivityAnalyzer Hypothesis(*this, directions);
    Hypothesis.retvals.insert(val);

    llvm::Function* ParentF = nullptr;
    if(auto inst = dyn_cast<Instruction>(val))
      ParentF = inst->getParent()->getParent();
    else if(auto arg = dyn_cast<Argument>(val))
      ParentF = arg->getParent();
    else {
      llvm::errs() << "unknown pointer value type: " << *val << "\n";
      assert(0 && "unknown pointer value type");
      llvm_unreachable("unknown pointer value type");
    }

    /*
    for(Argument &A : ParentF.args()) {
      // If the value could alias an argument that is known inactive, this
      // must be considered an active pointer
      auto Loc = MemoryLocation::getOrNone(&A);
      if (Loc) {
        if (!AA.isNoAlias(Loc.get(), MemoryLocation(val, LocationSize::unknown()))) {
          llvm::errs() << " </MEMSEARCH" << (int)directions << ">" << *val << " activeFromArgument=" << A << "\n";
          retvals.insert(val);
          return false;
        }
      }
    }
    */

    for(BasicBlock& BB : *ParentF) {
      if (potentiallyActiveStore && potentiallyActiveLoad) break;
      for(Instruction& I : BB) {
        if (potentiallyActiveStore && potentiallyActiveLoad) break;

        #if LLVM_VERSION_MAJOR >= 9
        auto AARes = AA.getModRefInfo(&I, MemoryLocation(val, LocationSize::unknown()));
        #else
        auto AARes = AA.getModRefInfo(&I, MemoryLocation(val, MemoryLocation::UnknownSize));
        #endif

        // If we haven't already shown a potentially active load
        // check if this loads the given value and is active
        if (!potentiallyActiveLoad && isRefSet(AARes)) {
          //llvm::errs() << "potential active load: " << I << "\n";
          if (auto LI = dyn_cast<LoadInst>(&I)) {
            // If the ref'ing value is a load check if the loaded value is active
            potentiallyActiveLoad = !Hypothesis.isconstantValueM(TR, LI);
          } else {
            // Otherwise fallback and check any part of the instruction is active
            // TODO: note that this can be optimized (especially for function calls)
            potentiallyActiveLoad = !Hypothesis.isconstantM(TR, &I);
          }
        }
        if (!potentiallyActiveStore && isModSet(AARes)) {
          //llvm::errs() << "potential active store: " << I << "\n";
          if (auto SI = dyn_cast<StoreInst>(&I)) {
            // If this is a store, simply check if the stored value is active
            potentiallyActiveStore = !Hypothesis.isconstantValueM(TR, SI->getValueOperand());
          } else {
            // Otherwise fallback and check if the instruction is active
            // TODO: note that this can be optimized (especially for function calls)
            potentiallyActiveStore = !Hypothesis.isconstantM(TR, &I);
          }
        }
      }
    }

    if (printconst)
      llvm::errs() << " </MEMSEARCH" << (int)directions << ">" << *val << " potentiallyActiveLoad=" << potentiallyActiveLoad << " potentiallyActiveStore=" << potentiallyActiveStore << "\n";
    if (potentiallyActiveLoad && potentiallyActiveStore) {
      insertAllFrom(Hypothesis);
      return false;
    } 
  }


    
  if (directions & UP) {
    if (auto inst = dyn_cast<Instruction>(val)) {
      // If the instruction is inactive in spite of this value being active,
      // this value is therefore inactive
      ActivityAnalyzer Hypothesis(*this, UP);
      Hypothesis.retvals.insert(val);

      if (Hypothesis.isconstantM(TR, inst)) {
        insertConstantsFrom(Hypothesis);
        constantvals.insert(val);
        return true;
      }
    }
  }

  if (directions & DOWN) {
    // Assume this value is active. If we can prove that all its uses
    // are inactive, we have proven this value inactive.
    ActivityAnalyzer Hypothesis(*this, DOWN);
    Hypothesis.retvals.insert(val);

    if (printconst)
      llvm::errs() << " <Value USESEARCH" << (int)directions << ">" << *val
                   << "\n";

    bool seenuse = false;

    for (const auto a : val->users()) {

      if (printconst)
        llvm::errs() << "      considering use of " << *val << " - " << *a
                     << "\n";

      if (auto call = dyn_cast<CallInst>(a)) {
        bool ConstantArg = Hypothesis.isFunctionArgumentConstant(call, val);
        if (ConstantArg) {
          if (printconst) {
            llvm::errs() << "Value found constant callinst use:" << *val
                         << " user " << *call << "\n";
          }
          continue;
        }
      }

      bool ConstantInst = Hypothesis.isconstantM(TR, cast<Instruction>(a));

      if (!ConstantInst) {
        if (printconst)
          llvm::errs() << "Value nonconstant inst (uses):" << *val << " user "
                       << *a << "\n";
        seenuse = true;
        break;
      } else {
        if (printconst)
          llvm::errs() << "Value found constant inst use:" << *val << " user "
                       << *a << "\n";
      }
    }

    if (!seenuse) {
      insertConstantsFrom(Hypothesis);
      if (printconst)
        llvm::errs() << "Value constant inst (uses):" << *val << "\n";
      return true;
    }

    if (printconst)
      llvm::errs() << " </Value USESEARCH" << (int)directions << ">" << *val
                   << "\n";
  }

  if (printconst)
    llvm::errs() << " Value nonconstant (couldn't disprove)[" << (int)directions
                 << "]" << *val << "\n";

  retvals.insert(val);
  return false;
}