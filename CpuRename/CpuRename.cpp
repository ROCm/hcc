//===- CpuRename.cpp - Remove non-GPU codes from LLVM IR -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass which add suffix for function used in AMP kernel
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h" 
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
using namespace llvm;


namespace {
    typedef SmallVector<Function *, 4> FunctionVect;

    void findKernels(Module& M, FunctionVect& found_kernels)
    {
        NamedMDNode * root = M.getNamedMetadata("opencl.kernels");
        if (!root || (root->getNumOperands() == 0))
            return;

        for (unsigned operand = 0, end = root->getNumOperands();
             operand < end; ++operand) {
            MDNode *M = root->getOperand(operand);
            if ( M->getNumOperands() < 1) return;
            Value *Op = M->getOperand(0);
            if ( Function *F = dyn_cast<Function>(Op)) {
                found_kernels.push_back(F);
            }
        }
    }

    class CpuRename : public ModulePass {
        FunctionVect foundKernels;
    public:
        static char ID;
        CpuRename() : ModulePass(ID) {}
        virtual ~CpuRename() {};
        bool runOnModule(Module& M);
    };

}

using std::queue;
using std::vector;
using std::map;

class Traverse
{
public:
    Traverse(queue<Function *>& WorkList, vector<Function *>& RenameList,
             map<Function *, bool>& visited)
        : WorkList(WorkList), RenameList(RenameList), visited(visited) {}

    bool operator()(Function *F) {
        bool update = false;
        for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
            for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
                CallSite CS(cast<Value>(I));
                if (!CS || isa<IntrinsicInst>(I))
                    continue;
                if (Function *save = CS.getCalledFunction()) {
                    if (save->isDeclaration())
                        continue;
                    if (!visited[save]) {
                        update = true;
                        visited[save] = true;
                        WorkList.push(save);
                        RenameList.push_back(save);
                    }
                }
            }
        return update;
    }
private:
    queue<Function *>& WorkList;
    vector<Function *>& RenameList;
    map<Function *, bool>& visited;
};

bool CpuRename::runOnModule(Module &M)
{
    findKernels(M, foundKernels);
    if (foundKernels.empty())
        return true;
    std::queue<Function *> WorkList;
    std::vector<Function *> RenameList;
    std::map<Function *, bool> visited;
    Traverse trav(WorkList, RenameList, visited);
    typedef FunctionVect::const_iterator kernel_iterator;
    typedef std::vector<Function *>::iterator fun_iterator;
    for (kernel_iterator KernFunc = foundKernels.begin(), KernFuncEnd = foundKernels.end();
         KernFunc != KernFuncEnd; ++KernFunc) {
        Function *F = *KernFunc;
        visited[F] = true;
        // The last function call in trampoline function is operator() of kernel
        if (trav(F))
            RenameList.pop_back();
    }
    while (!WorkList.empty()) {
        Function *F = WorkList.front();
        WorkList.pop();
        trav(F);
    }
    if (RenameList.size() == 0)
        return true;
    for (fun_iterator Fun = RenameList.begin(), E = RenameList.end();
         Fun != E; ++Fun) {
        Function *F = *Fun;
        F->setName(F->getName().str() + "_amp");
    }
    return true;
}

char CpuRename::ID = 0;
static RegisterPass<CpuRename>
Y("cpu-rename", "Suffix functions used in kernel with _amp to avoid name conflict in cpu path.");
