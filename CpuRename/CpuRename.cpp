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
    class CpuRename : public ModulePass {
    public:
        static char ID;
        CpuRename() : ModulePass(ID) {}
        virtual ~CpuRename() {};
        bool runOnModule(Module& M);
    };
}

bool CpuRename::runOnModule(Module &M)
{
    Module::FunctionListType &funcs = M.getFunctionList();
	for (Module::iterator I = funcs.begin(), E = funcs.end(); I != E; ) {
        Function *F = I++;
        if (F->getName().str().find("$_") != std::string::npos ||
            F->getName().str().find("_cl") != std::string::npos)
            F->setLinkage(GlobalValue::ExternalLinkage);
        else if (!F->isDeclaration() && !F->isIntrinsic()) {
            F->setName(F->getName().str() + "_amp");
            F->setLinkage(GlobalValue::InternalLinkage);
        }
    }
    return true;
}

char CpuRename::ID = 0;
static RegisterPass<CpuRename>
Y("cpu-rename", "Suffix functions used in kernel with _amp to avoid name conflict in cpu path.");
