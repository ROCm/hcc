//===- PromotePrivate.cpp - Private pointer promotion analysis -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Promote pointers obtained from new/delete operations from private address space
// to global segment (addrspace(1)).  This pass is only applied on HSA build path.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/InstVisitor.h"
#include "llvm/Analysis/PostDominators.h"
//#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Debug.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <map>
#include <set>

using namespace llvm;

namespace {

//#define HANDLE_LOAD_PRIVATE 0
//#define TILE_UNIFORM_DEBUG  0

/// PromotePrivates Class - Used to promote pointers from new/delete operations to global segment in HSA
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

/// NewedMemoryAnalyzer Class - Used to compute if an instruction will use operand from new/delete
///
class NewedMemoryAnalyzer : public InstVisitor<NewedMemoryAnalyzer> {
protected:
  Function *NewScalar;
  Function *NewArray;

  Function *Memset;
  //SmallPtrSet<Instruction *, 8> Visited;
  bool IsNewed;

public:
  NewedMemoryAnalyzer(Function &F) : IsNewed(false) {
    Module *M = F.getParent();
    NewScalar = M->getFunction(/*"_Znwj"*/ "_Znwm");
    NewArray = M->getFunction(/*"_Znaj"*/ "_Znam");

    Memset = M->getFunction("llvm.memset.p0i8.i64");
  }

  bool isNewed() { return IsNewed; }

  bool NotMemIntrinsic(Instruction &I) {
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      Function *Callee = CI->getCalledFunction();
      if (Memset && Callee == Memset) 
        return false;
    }
    return true;
  }

  /// Entry point of analysis
  void analyze(Instruction &I) { 
    if (!isa<StoreInst>(&I) && !isa<LoadInst>(&I) && NotMemIntrinsic(I)) {
    //if (!I.mayReadOrWriteMemory()) {
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

    if ((NewScalar && Callee == NewScalar) || (NewArray && Callee == NewArray)) {
      IsNewed =true;
      return;
    }

    if (Memset && Callee == Memset) {
      if (Instruction *Operand = dyn_cast<Instruction>(I.getArgOperand(0)))
      {
        //llvm::errs() << "I: " << I << "\n";
        //llvm::errs() << "Operand" << *Operand << "\n";
        visit(*Operand);
      }
    }
  }

#if 0
  void visitInstruction(Instruction &I);
#endif

};

} // ::<unnamed> namespace

/// PromotePrivates Implementation - Used to promote pointers from new/delete operations to global segment in HSA
///
bool PromotePrivates::runOnFunction(Function &F) {
  if (F.getName().find("cxxamp_trampoline") == StringRef::npos)
    return false;

  // Need refactor!
  Module *M = F.getParent();
  if ((M->getFunction(/*"_Znwj"*/ "_Znwm") == NULL) && (M->getFunction(/*"_Znaj"*/ "_Znam") == NULL))
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

#if 0
    llvm::errs() << "NeedPromoted:" << *I << "\n";
#endif

    if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      IRBuilder<> Builder(SI);
      //Value *StoreAddr = Builder.CreatePointerCast(SI->getPointerOperand(), Type::getInt64PtrTy(C, 1));
      PointerType *DestTy = SI->getPointerOperand()->getType()->getPointerElementType()->getPointerTo(1);
      Value *StoreAddr = Builder.CreatePointerCast(SI->getPointerOperand(), DestTy);
      StoreInst* nSI = new StoreInst(SI->getValueOperand(), StoreAddr);

      ReplaceInstWithInst(SI, nSI);
    }

    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      IRBuilder<> Builder(LI);
      Value *LoadAddr = Builder.CreatePointerCast(LI->getPointerOperand(), Type::getInt32PtrTy(C, 1));
      LoadInst* nLI = new LoadInst(LoadAddr);

      ReplaceInstWithInst(LI, nLI);
    }

    if (CallInst *CI = dyn_cast<CallInst>(I)) {
      IRBuilder<> Builder(CI);

      PointerType *DestTy = CI->getArgOperand(0)->getType()->getPointerElementType()->getPointerTo(1);
      Value *MemsetAddr = Builder.CreatePointerCast(CI->getArgOperand(0), DestTy);
      std::vector<Value*> ArgsVec;
      ArgsVec.push_back(MemsetAddr);
      for (int i = 1, e = CI->getNumArgOperands(); i < e; i++) {
        ArgsVec.push_back(CI->getArgOperand(i));
      }
      ArrayRef<Value*> Args(ArgsVec);

      FunctionType *MemsetFuncType = CI->getCalledFunction()->getFunctionType();
      Type *MemsetRetType = MemsetFuncType->getReturnType();
      std::vector<Type*> ArgsTypeVec;
      ArgsTypeVec.push_back(DestTy);
      for (int i = 1, e = MemsetFuncType->getNumParams(); i < e; i++) {
        ArgsTypeVec.push_back(MemsetFuncType->getParamType(i));
      }
      ArrayRef<Type*> ArgsType(ArgsTypeVec);  
      FunctionType *nMemsetFuncType = FunctionType::get(MemsetRetType, ArgsType, false);
      M->getOrInsertFunction("llvm.memset.p1i8.i64", nMemsetFuncType);
      Function *MemsetFunc = M->getFunction("llvm.memset.p1i8.i64");

      CallInst* nCI = CallInst::Create(MemsetFunc, Args);

      ReplaceInstWithInst(CI, nCI);
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

