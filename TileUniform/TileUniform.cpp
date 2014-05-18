//===- Hello.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements two versions of the LLVM "Hello World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//

//#define DEBUG_TYPE "PromoteGlobals"

#include "llvm/InstVisitor.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Debug.h"
#include <map>
#include <set>

using namespace llvm;

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
  ControlDependences() : FunctionPass(ID) {
    //initializeControlDependencesPass(*PassRegistry::getPassRegistry());
  }

  virtual void releaseMemory() { CtrlDeps.clear(); }

  // Accessor interface: 
  typedef CtrlDepSetMapType::iterator iterator; 
  typedef CtrlDepSetMapType::const_iterator const_iterator; 
  iterator       begin()       { return CtrlDeps.begin(); } 
  const_iterator begin() const { return CtrlDeps.begin(); } 
  iterator       end()         { return CtrlDeps.end(); } 
  const_iterator end()   const { return CtrlDeps.end(); } 
  iterator       find(BasicBlock *B)       { return CtrlDeps.find(B); } 
  const_iterator find(BasicBlock *B) const { return CtrlDeps.find(B); }

  /// print - Convert to human readable form
  ///
  virtual void print(raw_ostream &OS, const Module* = 0) const;

  /// dump - Dump the control dependences to dbgs().
  void dump() const;

  virtual bool runOnFunction(Function &);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
//    AU.addRequired<PostDominatorTree>();
  }
};

/// ThreadDependencyAnalyzer Class - Used to compute the thread dependency.
///
class ThreadDependencyAnalyzer : public InstVisitor<ThreadDependencyAnalyzer> {
protected:
  Function *get_global_id;
  Function *get_local_id;

public:
  ThreadDependencyAnalyzer(Module &M);
  void analyze(Instruction &I) { visit(I); }
  // Opcode Implementations
  void visitLoadInst(LoadInst &I);
  void visitCallInst(CallInst &I);
  void visitInstruction(Instruction &I);
};

/// TileUniform Class - Used to ensure tile uniform.
///
class TileUniform : public ModulePass {
public:
  static char ID;
protected:
  Function *barrier;

public:        
  TileUniform() : ModulePass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage& AU) const {
    AU.setPreservesAll();
//    AU.addRequired<ControlDependences>();
  }

  virtual bool runOnModule(Module& M);
};

} // ::<unnamed> namespace

/// ControlDependences Implementation - Used to compute the control 
/// dependences.
///
char ControlDependences::ID = 0;
static RegisterPass<ControlDependences>
X("ctrl-deps", "Control Dependences Construction.");

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
      WriteAsOperand(OS, I->first, false);
    else
      OS << " <<exit node>>";
    OS << " is Control Dependent on:\t";
    
    const std::set<BasicBlock*> &BBs = I->second;
    
    for (std::set<BasicBlock*>::const_iterator I = BBs.begin(), E = BBs.end();
         I != E; ++I) {
      OS << ' ';
      if (*I)
        WriteAsOperand(OS, *I, false);
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
  get_global_id = M.getFunction("get_global_id");
  get_local_id = M.getFunction("get_local_id");
}

void ThreadDependencyAnalyzer::visitLoadInst(LoadInst &I) { 
  report_fatal_error("violated tile uniform\n");
}

void ThreadDependencyAnalyzer::visitCallInst(CallInst &I) { 
  Function *callee = I.getCalledFunction();
  if (callee == get_local_id || callee == get_global_id)
    report_fatal_error("violated tile uniform\n");
}

void ThreadDependencyAnalyzer::visitInstruction(Instruction &I) {
  for (User::op_iterator oi = I.op_begin(), e = I.op_end(); oi != e; ++oi) {
    if (Instruction *Inst = dyn_cast<Instruction>(*oi))
      visit(*Inst);
  }
} 

/// TileUniform Implementation - Used to ensure tile uniform.
///
bool TileUniform::runOnModule(Module &M) {

#if 0
  /// The following code is used to test ControlDependences class
  ///
  Module::FunctionListType &funcs = M.getFunctionList();
  ControlDependences *CDs = new ControlDependences();
  
  for (Module::iterator I = funcs.begin(), E = funcs.end(); I != E; ++I) {
    Function &func = *I;
    if (!func.isDeclaration()) {
      CDs->runOnFunction(func);
      CDs->print(errs());
      CDs->releaseMemory();      
    }
  }
  
  return false;
#endif

  if(!(barrier = M.getFunction("barrier")))
    return false;

  for (Value::use_iterator UI = barrier->use_begin(), UE = barrier->use_end();
        UI != UE; ++UI) {
    if (Instruction *I = dyn_cast<Instruction>(*UI)) {
      BasicBlock *BB = I->getParent();
      Function *F = BB->getParent();
      ControlDependences *CtrlDeps = new ControlDependences();
      ThreadDependencyAnalyzer *TDA = new ThreadDependencyAnalyzer(M);

      CtrlDeps->runOnFunction(*F);

      typedef ControlDependences::CtrlDepSetType CDST;

      CDST *CtrlDep = &CtrlDeps->find(BB)->second;

      for (CDST::iterator i = CtrlDep->begin(), e = CtrlDep->end(); i != e;
            ++i) {
        BasicBlock *CtrlDepBB = *i;
        TerminatorInst *TI = CtrlDepBB->getTerminator();
        TDA->analyze(*TI);
      }

      CtrlDeps->releaseMemory();
      delete CtrlDeps;
      delete TDA;
    }
  }
  return false;
}

char TileUniform::ID = 0;
static RegisterPass<TileUniform>
Y("tile-uniform", "Ensure tile uniform.");
