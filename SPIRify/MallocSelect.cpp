//===- MallocSelect.cpp - Malloc Selection Transformation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Transform control divergent Xmalloc to malloc
//
//===----------------------------------------------------------------------===//

// TODO: Refactor Tile Uniform/Malloc Selection/Promote Privates

#include "llvm/IR/InstVisitor.h"
#include "llvm/Analysis/PostDominators.h"
//#include "llvm/Assembly/Writer.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"

#include <map>
#include <set>

using namespace llvm;

static cl::opt<bool>
AlwaysMalloc("always-malloc", cl::init(false), cl::Hidden,
  cl::desc("always transform Xmalloc to malloc"));

namespace {

/// ControlDependences Class - Used to compute the control dependences.
///
class ControlDependences : public FunctionPass {
public:
  typedef std::set<BasicBlock*>                 CtrlDepSetType;
  typedef std::map<BasicBlock*, CtrlDepSetType> CtrlDepSetMapType;
  static char ID; // Pass ID, replacement for typeid

protected:
  CtrlDepSetMapType CtrlDeps;

public:
  ControlDependences() : FunctionPass(ID) {}

  virtual void releaseMemory() { CtrlDeps.clear(); }

  /// Accessor interface: 
  typedef CtrlDepSetMapType::iterator iterator; 
  typedef CtrlDepSetMapType::const_iterator const_iterator; 
  iterator       begin()       { return CtrlDeps.begin(); } 
  const_iterator begin() const { return CtrlDeps.begin(); } 
  iterator       end()         { return CtrlDeps.end(); } 
  const_iterator end()   const { return CtrlDeps.end(); } 
  iterator       find(BasicBlock *B)       { return CtrlDeps.find(B); } 
  const_iterator find(BasicBlock *B) const { return CtrlDeps.find(B); }

  /// print - Convert to human readable form
  virtual void print(raw_ostream &OS, const Module* = 0) const;

  /// dump - Dump the control dependences to dbgs().
  void dump() const;

  virtual bool runOnFunction(Function &);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
  }
};

/// ThreadDependencyAnalyzer Class - Used to compute the thread dependency.
///
class ThreadDependencyAnalyzer : public InstVisitor<ThreadDependencyAnalyzer> {
protected:
  Function *get_global_id;
  Function *get_local_id;
  SmallPtrSet<Instruction *, 8> Visited;

  bool indep;

public:
  ThreadDependencyAnalyzer(Module &M);

  bool isIndep() { return indep; }

  /// Entry point of analysis
  void analyze(Instruction &I) { visit(I); }

  /// Opcode Implementations
  void visitCallInst(CallInst &I);
  void visitInstruction(Instruction &I);
};

/// MallocSelect Class - Used to transform control divergent Xmalloc to malloc
///
class MallocSelect : public ModulePass {
public:
  static char ID;

protected:
  Function *NewScalar;
  Function *NewArray;

public:        
  MallocSelect() : ModulePass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage& AU) const {
    AU.setPreservesAll();
    // FIXME: use AnalysisUsage class 
  }

  virtual bool runOnModule(Module& M);
};

} // ::<unnamed> namespace

/// ControlDependences Implementation - Used to compute the control 
/// dependences.
///
char ControlDependences::ID = 0;
static RegisterPass<ControlDependences>
X("ctrl-deps-redun", "Control Dependences Construction.");

bool ControlDependences::runOnFunction(Function &F) {
  PostDominatorTree *PDT = new PostDominatorTree();
  PDT->runOnFunction(F);

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    for (succ_iterator SI = succ_begin(I), SE = succ_end(I); SI != SE; ++SI) {
      BasicBlock *BB = dyn_cast<BasicBlock>(I);
      BasicBlock *SBB = *SI;

      if(PDT->dominates(SBB, BB))
        continue;
      
      BasicBlock *PBB = PDT->getNode(BB)->getIDom()->getBlock();
      while (SBB != PBB) {
        CtrlDepSetType &CtrlDep = CtrlDeps[SBB];
        CtrlDep.insert(BB);
        SBB = PDT->getNode(SBB)->getIDom()->getBlock();
      }
    }
  }

  return false;
}

void ControlDependences::print(raw_ostream &OS, const Module*) const {
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    OS << "  BB ";
    if (I->first)
      ; //WriteAsOperand(OS, I->first, false);
    else
      OS << " <<exit node>>";
    OS << " is Control Dependent on:\t";
    
    const std::set<BasicBlock*> &BBs = I->second;
    
    for (std::set<BasicBlock*>::const_iterator I = BBs.begin(), E = BBs.end();
         I != E; ++I) {
      OS << ' ';
      if (*I)
        ; //WriteAsOperand(OS, *I, false);
      else
        OS << "<<exit node>>";
    }
    OS << "\n";
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void ControlDependences::dump() const {
  print(dbgs());
}
#endif

/// ThreadDependencyAnalyzer Implementation - Used to compute the thread 
/// dependency.
///
ThreadDependencyAnalyzer::ThreadDependencyAnalyzer(Module &M) {
  get_global_id = M.getFunction("amp_get_global_id");
  get_local_id = M.getFunction("amp_get_local_id");
  indep = true;
}

void ThreadDependencyAnalyzer::visitCallInst(CallInst &I) {
#if LLVM_VERSION_MAJOR == 3
  #if (LLVM_VERSION_MINOR >= 3) && (LLVM_VERSION_MINOR <= 5)
  // logic which is compatible from LLVM 3.3 till LLVM 3.5
  if (!Visited.insert(&I)) return;
  #elif LLVM_VERSION_MINOR > 5
  // support new SmallPtrSet interface
  auto pair = Visited.insert(&I);
  if (!pair.second) return;
  #else
    #error Unsupported LLVM MINOR VERSION
  #endif
#else
  #error Unsupported LLVM MAJOR VERSION
#endif

#if DEBUG
  errs() << I << "\n";
#endif
  Function *callee = I.getCalledFunction();
  if (callee == get_local_id || callee == get_global_id)
    indep = false;
}

void ThreadDependencyAnalyzer::visitInstruction(Instruction &I) {
#if LLVM_VERSION_MAJOR == 3
  #if (LLVM_VERSION_MINOR >= 3) && (LLVM_VERSION_MINOR <= 5)
  // logic which is compatible from LLVM 3.3 till LLVM 3.5
  if (!Visited.insert(&I)) return;
  #elif LLVM_VERSION_MINOR > 5
  // support new SmallPtrSet interface
  auto pair = Visited.insert(&I);
  if (!pair.second) return;
  #else
    #error Unsupported LLVM MINOR VERSION
  #endif
#else
  #error Unsupported LLVM MAJOR VERSION
#endif

#if DEBUG
  errs() << I << "\n";
#endif
  for (User::op_iterator oi = I.op_begin(), e = I.op_end(); oi != e; ++oi) {
    if (Instruction *Inst = dyn_cast<Instruction>(*oi))
      visit(*Inst);
  }
} 

/// MallocSelect Implementation
///
bool MallocSelect::runOnModule(Module &M) {

  std::vector<Instruction*> Needmalloc;

  if ((NewScalar = M.getFunction("_Znwm"))) { // new

    for (Value::user_iterator UI = NewScalar->user_begin(), UE = NewScalar->user_end();
          UI != UE; ++UI) {

      if (Instruction *I = dyn_cast<Instruction>(*UI)) {

        if (AlwaysMalloc) {
          Needmalloc.push_back(I);
          continue;
        }

        BasicBlock *BB = I->getParent();
        Function *F = BB->getParent();

        ControlDependences *CtrlDeps = new ControlDependences();
        CtrlDeps->runOnFunction(*F);

        if (CtrlDeps->find(BB) == CtrlDeps->end())
          continue;

        typedef ControlDependences::CtrlDepSetType CDST;
        CDST *CtrlDep = &CtrlDeps->find(BB)->second;      

        for (CDST::iterator i = CtrlDep->begin(), e = CtrlDep->end(); i != e;
              ++i) {
          BasicBlock *CtrlDepBB = *i;

          TerminatorInst *TI = CtrlDepBB->getTerminator();
          ThreadDependencyAnalyzer TDA(M);
          TDA.analyze(*TI);

          if (!TDA.isIndep())
            Needmalloc.push_back(I); 
        }

        CtrlDeps->releaseMemory();
        delete CtrlDeps;
      }
    }
  } // end of new

  if ((NewArray = M.getFunction("_Znam"))) { // new[]

    for (Value::user_iterator UI = NewArray->user_begin(), UE = NewArray->user_end();
          UI != UE; ++UI) {

      if (Instruction *I = dyn_cast<Instruction>(*UI)) {

        if (AlwaysMalloc) {
          Needmalloc.push_back(I);
          continue;
        }

        BasicBlock *BB = I->getParent();
        Function *F = BB->getParent();

        ControlDependences *CtrlDeps = new ControlDependences();
        CtrlDeps->runOnFunction(*F);

        if (CtrlDeps->find(BB) == CtrlDeps->end())
          continue;

        typedef ControlDependences::CtrlDepSetType CDST;
        CDST *CtrlDep = &CtrlDeps->find(BB)->second;

        for (CDST::iterator i = CtrlDep->begin(), e = CtrlDep->end(); i != e;
              ++i) {
          BasicBlock *CtrlDepBB = *i;

          TerminatorInst *TI = CtrlDepBB->getTerminator();
          ThreadDependencyAnalyzer TDA(M);
          TDA.analyze(*TI);

          if (!TDA.isIndep())
            Needmalloc.push_back(I);
        }

        CtrlDeps->releaseMemory();
        delete CtrlDeps;
      }
    }
  } // end of new[]

  while (!Needmalloc.empty()) {
    Instruction *I = Needmalloc.back();

#if DEBUG
    llvm::errs() << "Needmalloc:" << *I << "\n";
#endif
    if (CallInst *CI = dyn_cast<CallInst>(I)) {
      std::vector<Value*> ArgsVec;
      for (int i = 0, e = CI->getNumArgOperands(); i < e; i++) {
        ArgsVec.push_back(CI->getArgOperand(i));
      }
      ArrayRef<Value*> Args(ArgsVec);

      FunctionType *MemsetFuncType = CI->getCalledFunction()->getFunctionType();
      Type *MemsetRetType = MemsetFuncType->getReturnType();
      std::vector<Type*> ArgsTypeVec;
      for (int i = 0, e = MemsetFuncType->getNumParams(); i < e; i++) {
        ArgsTypeVec.push_back(MemsetFuncType->getParamType(i));
      }
      ArrayRef<Type*> ArgsType(ArgsTypeVec);
      FunctionType *nMemsetFuncType = FunctionType::get(MemsetRetType, ArgsType, false);

      Function *MemsetFunc;
      if (CI->getCalledFunction() == NewScalar) {
        M.getOrInsertFunction("_Znwm_malloc", nMemsetFuncType);
        MemsetFunc = M.getFunction("_Znwm_malloc");
      } else if (CI->getCalledFunction() == NewArray) {
        M.getOrInsertFunction("_Znam_malloc", nMemsetFuncType);
        MemsetFunc = M.getFunction("_Znam_malloc");
      }

      CallInst* nCI = CallInst::Create(MemsetFunc, Args);

      ReplaceInstWithInst(CI, nCI);
    }

    Needmalloc.pop_back();
  }

  return true;
}

char MallocSelect::ID = 0;
static RegisterPass<MallocSelect>
Y("malloc-select", "Transform control divergent Xmalloc to malloc.");
