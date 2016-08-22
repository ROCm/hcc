//===- CpuRename.cpp - Rename functions used in cpu kernel -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pss which renames functions used in cpu kernel
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h" 
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
using namespace llvm;


namespace {
    typedef SmallVector<Function *, 4> FunctionVect;

    void findKernels(Module& M, FunctionVect& found_kernels)
    {
        NamedMDNode * root = M.getNamedMetadata("hcc.kernels");
        if (!root || (root->getNumOperands() == 0))
            return;

        for (unsigned operand = 0, end = root->getNumOperands();
             operand < end; ++operand) {
            MDNode *M = root->getOperand(operand);
            if ( M->getNumOperands() < 1) return;

            const MDOperand& Op = M->getOperand(0);
            if ( Function *F = mdconst::dyn_extract<Function>(Op)) {
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
    Traverse(queue<Function *>& WorkList, map<Function *, bool>& visited)
        : WorkList(WorkList), visited(visited) {}

    void operator()(Function *F) {
        for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
            for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
                CallSite CS(cast<Value>(I));
                if (!CS || isa<IntrinsicInst>(I))
                    continue;
                if (Function *save = CS.getCalledFunction()) {
                    if (save->isDeclaration() || save->isIntrinsic())
                        continue;
                    if (!visited[save]) {
                        visited[save] = true;
                        WorkList.push(save);
                    }
                }
            }
    }
private:
    queue<Function *>& WorkList;
    map<Function *, bool>& visited;
};

bool CpuRename::runOnModule(Module &M)
{
    findKernels(M, foundKernels);
    std::queue<Function *> WorkList;
    std::map<Function *, bool> visited;
    Traverse trav(WorkList, visited);
    typedef FunctionVect::const_iterator kernel_iterator;
    for (kernel_iterator KernFunc = foundKernels.begin(), KernFuncEnd = foundKernels.end();
         KernFunc != KernFuncEnd; ++KernFunc)
        trav(*KernFunc);
    while (!WorkList.empty()) {
        Function *F = WorkList.front();
        WorkList.pop();
        trav(F);
    }
    Module::FunctionListType &funcs = M.getFunctionList();
    for (Module::iterator I = funcs.begin(), E = funcs.end(); I != E; ) {
        Function *F = (I++).operator->();
        if (F->getName().str().find("$_") != std::string::npos ||
                F->getName().str().find("_cl") != std::string::npos)
            F->setLinkage(GlobalValue::ExternalLinkage);
        else if (visited[F])
            F->setName(F->getName().str() + "_amp");
        else if (!F->isDeclaration() && !F->isIntrinsic())
            F->setLinkage(GlobalValue::PrivateLinkage);
    }
    return true;
}

char CpuRename::ID = 0;
static RegisterPass<CpuRename>
Y("cpu-rename", "Suffix functions used in kernel with _amp to avoid name conflict in cpu path.");

