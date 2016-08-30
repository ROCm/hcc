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

#include "llvm/IR/InstVisitor.h"
#include "llvm/Analysis/PostDominators.h"
//#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Debug.h"

#include <map>
#include <set>

using namespace llvm;

namespace {

#define HANDLE_LOAD_PRIVATE 0
#define TILE_UNIFORM_DEBUG  0

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

public:
  ThreadDependencyAnalyzer(Module &M);

  /// Entry point of analysis
  void analyze(Instruction &I) { visit(I); }

  /// Opcode Implementations
#if HANDLE_LOAD_PRIVATE
  void visitLoadInst(LoadInst &I);
#endif
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
X("ctrl-deps", "Control Dependences Construction.");

bool ControlDependences::runOnFunction(Function &F) {
  PostDominatorTreeWrapperPass *PDT = new PostDominatorTreeWrapperPass();
  PDT->runOnFunction(F);

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    for (succ_iterator SI = succ_begin(I.operator->()), SE = succ_end(I.operator->()); SI != SE; ++SI) {
      BasicBlock *BB = dyn_cast<BasicBlock>(I);
      BasicBlock *SBB = *SI;

      if(PDT->getPostDomTree().dominates(SBB, BB))
        continue;
      
      BasicBlock *PBB = PDT->getPostDomTree().getNode(BB)->getIDom()->getBlock();
      while (SBB != PBB) {
        CtrlDepSetType &CtrlDep = CtrlDeps[SBB];
        CtrlDep.insert(BB);
        SBB = PDT->getPostDomTree().getNode(SBB)->getIDom()->getBlock();
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
}

#if HANDLE_LOAD_PRIVATE
void ThreadDependencyAnalyzer::visitLoadInst(LoadInst &I) {
  auto pair = Visited.insert(&I);
  if (!pair.second) return;
#if TILE_UNIFORM_DEBUG
  errs() << I << "\n";
#endif
  // FIXME: load from private memory is still thread dependent
}
#endif

void ThreadDependencyAnalyzer::visitCallInst(CallInst &I) {
  auto pair = Visited.insert(&I);
  if (!pair.second) return;
#if TILE_UNIFORM_DEBUG
  errs() << I << "\n";
#endif
  Function *callee = I.getCalledFunction();
  if (callee == get_local_id || callee == get_global_id)
    report_fatal_error("violated tile uniform\n");
}

void ThreadDependencyAnalyzer::visitInstruction(Instruction &I) {
  // support new SmallPtrSet interface
  auto pair = Visited.insert(&I);
  if (!pair.second) return;
#if TILE_UNIFORM_DEBUG
  errs() << I << "\n";
#endif
  for (User::op_iterator oi = I.op_begin(), e = I.op_end(); oi != e; ++oi) {
    if (Instruction *Inst = dyn_cast<Instruction>(*oi))
      visit(*Inst);
  }
} 

/// TileUniform Implementation - Used to ensure tile uniform.
///
bool TileUniform::runOnModule(Module &M) {
  // FIXME: TileUniform should be implement as a FunctionPass
  if(!(barrier = M.getFunction("amp_barrier")))
    return false;

  for (Value::user_iterator UI = barrier->user_begin(), UE = barrier->user_end();
        UI != UE; ++UI) {
    if (Instruction *I = dyn_cast<Instruction>(*UI)) {
      BasicBlock *BB = I->getParent();
      Function *F = BB->getParent();

#if TILE_UNIFORM_DEBUG
      errs() << "Decide whether Instruction " << *I << "\n"
               << " of Basic Block " << BB->getName() << "\n"
               << " of Function " << F->getName() << "\n"
               << " is tile uniform or not\n";
#endif

      ControlDependences *CtrlDeps = new ControlDependences();
      CtrlDeps->runOnFunction(*F);

#if TILE_UNIFORM_DEBUG
      CtrlDeps->print(errs());
#endif

      if (CtrlDeps->find(BB) == CtrlDeps->end())
        continue;

      typedef ControlDependences::CtrlDepSetType CDST;
      CDST *CtrlDep = &CtrlDeps->find(BB)->second;      

      for (CDST::iterator i = CtrlDep->begin(), e = CtrlDep->end(); i != e;
            ++i) {
        BasicBlock *CtrlDepBB = *i;

#if TILE_UNIFORM_DEBUG
        errs() << "Analyze the Thread Dependency of Terminator Instruction of "
                 << CtrlDepBB->getName() << " which is " << BB->getName()
                 << " control dependent on \n";
#endif

        TerminatorInst *TI = CtrlDepBB->getTerminator();
        ThreadDependencyAnalyzer TDA(M);
        TDA.analyze(*TI);
      }

      CtrlDeps->releaseMemory();
      delete CtrlDeps;
    }
  }
  return false;
}

char TileUniform::ID = 0;
static RegisterPass<TileUniform>
Y("tile-uniform", "Ensure tile uniform.");
