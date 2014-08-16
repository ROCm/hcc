//===- TileUniform.cpp - Tile Uniform analysis ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detects whether all active control flow expressions leading to a tile barrier 
// to be tile-uniform.
//
//===----------------------------------------------------------------------===//

#include "llvm/InstVisitor.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Debug.h"

#include "llvm/Support/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <map>
#include <set>

using namespace llvm;

namespace {

//#define HANDLE_LOAD_PRIVATE 0
//#define TILE_UNIFORM_DEBUG  0

/// TileUniform Class - Used to ensure tile uniform.
///
class PromotePrivates : public FunctionPass {
public:
  static char ID;

public:        
  PromotePrivates() : FunctionPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage& AU) const {
    AU.setPreservesAll();
    // FIXME: use AnalysisUsage class 
  }

  virtual bool runOnFunction(Function &F);
};

/// NewedMemoryAnalyzer Class - Used to compute the thread dependency.
///
class NewedMemoryAnalyzer : public InstVisitor<NewedMemoryAnalyzer> {
protected:
  Function *NewScalar;
  Function *NewArray;
  //SmallPtrSet<Instruction *, 8> Visited;
  bool IsNewed;

public:
  NewedMemoryAnalyzer(Function &F) : IsNewed(false) {
    Module *M = F.getParent();
    NewScalar = M->getFunction("_Znwj");
    NewArray = M->getFunction("_Znaj"); 
  }

  bool isNewed() { return IsNewed; }

  /// Entry point of analysis
  void analyze(Instruction &I) { 
    if (!isa<StoreInst>(&I) && !isa<LoadInst>(&I)) {
      IsNewed = false;
      return;
    }
    visit(I); 
  }

  /// Opcode Implementations
  void visitLoadInst(LoadInst &I) {
    unsigned AS = I.getPointerAddressSpace();
    if (AS == 1 || AS == 2 || AS == 3) {
      IsNewed = false;
      return;
    }
    if (Instruction *Operand = dyn_cast<Instruction>(I.getPointerOperand()))
      visit(*Operand);
  }
  
  void visitStoreInst(StoreInst &I) {
    unsigned AS = I.getPointerAddressSpace();
    if (AS == 1 || AS == 2 || AS == 3) {
      IsNewed = false;
      return;
    }
    if (Instruction *Operand = dyn_cast<Instruction>(I.getPointerOperand()))
      visit(*Operand);
  }

  void visitBitCastInst(BitCastInst &I) {
    if (Instruction *Operand = dyn_cast<Instruction>(I.getOperand(0)))
      visit(*Operand);
  }

  void visitGetElementPtrInst(GetElementPtrInst  &I) {
    if (Instruction *Operand = dyn_cast<Instruction>(I.getPointerOperand()))
      visit(*Operand);
  }

  void visitCallInst(CallInst &I) {
    Function *Callee = I.getCalledFunction();
    if ((NewScalar && Callee == NewScalar) || (NewArray && Callee == NewArray))
      IsNewed =true;
  }

#if 0
  void visitInstruction(Instruction &I);
#endif

};

} // ::<unnamed> namespace

/// PromotePrivates Implementation - Used to ensure tile uniform.
///
bool PromotePrivates::runOnFunction(Function &F) {
  if (F.getName().find("cxxamp_trampoline") == StringRef::npos)
    return false;

  //errs() << "Execute PromotePrivates::runOnFunction: " << F.getName() << "\n";

  LLVMContext& C = F.getContext();
  std::vector<Instruction*> NeedPromoted;

  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    NewedMemoryAnalyzer NMA(F);
    NMA.analyze(*I);
    
    if (NMA.isNewed())
      NeedPromoted.push_back(&*I);
  }

#if 0
  for (unsigned i = 0; i < NeedPromoted.size(); i++)
    errs () << *NeedPromoted[i] << "\n";
#endif

  while (!NeedPromoted.empty()) {
    Instruction *I = NeedPromoted.back();

    if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      IRBuilder<> Builder(SI);
      Value *StoreAddr = Builder.CreatePointerCast(SI->getPointerOperand(), Type::getInt32PtrTy(C, 1));
      StoreInst* nSI = new StoreInst(SI->getValueOperand(), StoreAddr);

      ReplaceInstWithInst(SI, nSI);
    }

    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      IRBuilder<> Builder(LI);
      Value *LoadAddr = Builder.CreatePointerCast(LI->getPointerOperand(), Type::getInt32PtrTy(C, 1));
      LoadInst* nLI = new LoadInst(LoadAddr);

      ReplaceInstWithInst(LI, nLI);
    }

#if 0
    if (StoreInst *SI = dyn_cast<StoreInst>(NeedPromoted.back()) ) {
    IRBuilder<> Builder(SI);
    Value *StoreAddr = Builder.CreatePointerCast(SI->getPointerOperand(), Type::getInt32PtrTy(C, 1));
    StoreInst* nSI = new StoreInst(SI->getValueOperand(), StoreAddr);
    ReplaceInstWithInst(SI, nSI);
    }
    else if (LoadInst *LI = dyn_cast<LoadInst>(NeedPromoted.back()) ) {
    IRBuilder<> Builder(LI);
    Value *StoreAddr = Builder.CreatePointerCast(LI->getPointerOperand(), Type::getInt32PtrTy(C, 1));
    LoadInst* nLI = new LoadInst(/*SI->getValueOperand(), */StoreAddr);
    ReplaceInstWithInst(LI, nLI);
    }
#endif

    NeedPromoted.pop_back();
  }

  //F.dump();
  //errs() << "Finished PromotePrivates::runOnFunction\n";

  return true;
}

char PromotePrivates::ID = 0;
static RegisterPass<PromotePrivates>
Y("promote-privates", "Promote st_private to st_global.");

