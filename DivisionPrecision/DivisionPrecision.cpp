//===- DivisionPrecision.cpp - Convert fdiv float to builtin calls --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file converts fdiv float to builtin function calls to avoid bugs in
// AMD CL compiler for SPIR inputs.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
using namespace llvm;

#define DEBUG_TYPE "dp"

STATISTIC(KernelCounter, "Counts number of kernel processed");
STATISTIC(FDivFloatCounter, "Counts number of fdiv float converted");

namespace {
  typedef SmallVector<Function*, 3> FunctionVect;

  // The implementation with getAnalysisUsage implemented.
  struct DivisionPrecision : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    DivisionPrecision() : FunctionPass(ID), found_kernels(), NewFDiv(nullptr) {}

    bool doInitialization(Module &M) override {

      // walk through all SPIR kernels and log them
      NamedMDNode* root = M.getNamedMetadata("opencl.kernels");
      if (!root || (root->getNumOperands() == 0))
        return false;

      for (unsigned operand = 0, end = root->getNumOperands(); operand < end; ++operand) {
        MDNode* N = root->getOperand(operand);

#if LLVM_VERSION_MAJOR == 3
  #if (LLVM_VERSION_MINOR >= 3) && (LLVM_VERSION_MINOR <= 5)
        // logic which is compatible from LLVM 3.3 till LLVM 3.5
        Value* Op = N->getOperand(0);
        if (!Op) {
          return false;
        }
        if (Function* F = dyn_cast<Function>(Op)) {
          found_kernels.push_back(F);
        }
  #elif LLVM_VERSION_MINOR > 5
        // support new metadata data structure introduced in LLVM 3.6+
        const MDOperand& Op = N->getOperand(0);
        if (Function* F = mdconst::dyn_extract<Function>(Op)) {
          found_kernels.push_back(F);
        }
  #else
    #error Unsupported LLVM MINOR VERSION
  #endif
#else
  #error Unsupported LLVM MAJOR VERSION
#endif

      }

      // declare __precise_fp32_div_f32 which would be used to replace all fdiv float
      LLVMContext& CTX = M.getContext();
      std::vector<Type*> argTypes;
      argTypes.push_back(Type::getFloatTy(CTX));
      argTypes.push_back(Type::getFloatTy(CTX));
      FunctionType* FT = FunctionType::get(Type::getFloatTy(CTX), ArrayRef<Type*>(argTypes), false);
      NewFDiv = M.getOrInsertFunction("__precise_fp32_div_f32", FT);

      return true;
    }

    bool convertFDivFloatInstruction(Function& F) {
      bool modifiedFunction = false;

      // for each basic block
      for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
        BasicBlock& B = *BI;

        // for each instruction
        for (BasicBlock::iterator II = B.begin(), IE = B.end(); II != IE; ++II) {
          // check if it's fdiv float
          Instruction& I = *II;
          if ((I.getOpcode() == Instruction::FDiv) && (I.getType()->isFloatTy())) {
            // create a CallInst to __precise_fp32_div_f32
            std::vector<Value*> args;
            args.push_back(I.getOperand(0));
            args.push_back(I.getOperand(1));
            CallInst* CI = CallInst::Create(NewFDiv, ArrayRef<Value*>(args), "", &I);

            // replace instruction
            ReplaceInstWithInst(I.getParent()->getInstList(), II, CI);

            // update statistic counter
            ++FDivFloatCounter;

            // mark the function has been changed
            modifiedFunction = true;
          }
        } // for each instruciton
      } // for each bb

      return modifiedFunction;
    }

    bool runOnFunction(Function &F) override {
      bool modified = false;

      //errs().write_escaped(F.getName()) << "\n";
      for (FunctionVect::iterator KI = found_kernels.begin(), KE = found_kernels.end(); KI != KE; ++KI) {
        if (*KI == &F) {
          ++KernelCounter;
          modified = convertFDivFloatInstruction(F);
          break;
        }
      }

      return modified;
    }

    // We don't modify the program, so we preserve all analyses.
    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesAll();
    }

    FunctionVect found_kernels;

    Value* NewFDiv;
  };
}

char DivisionPrecision::ID = 0;
static RegisterPass<DivisionPrecision>
Y("divprecise", "Division Precision Pass (with getAnalysisUsage implemented)", false, false);
