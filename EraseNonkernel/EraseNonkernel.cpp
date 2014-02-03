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

#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;


namespace {

/* Data type container into which the list of LLVM functions
   that are OpenCL kernels will be stored */
typedef SmallVector<Function *, 3> FunctionVect;

/* The name of the MDNode into which the list of
   MD nodes referencing each OpenCL kernel is stored. */
static Twine KernelListMDNodeName = "opencl.kernels";

/* Find the MDNode which reference the list of opencl kernels.
   NULL if it does not exists. */

NamedMDNode * getKernelListMDNode (Module & M)
{
        return M.getNamedMetadata(KernelListMDNodeName);
}

/* A visitor function which is called for each MDNode
   located into another MDNode */
class KernelNodeVisitor {
        public:
        KernelNodeVisitor(FunctionVect& FV);
        void operator()(MDNode * N);

        private:
        FunctionVect& found_kernels;
};

KernelNodeVisitor::KernelNodeVisitor(FunctionVect& FV)
        : found_kernels(FV)
{}

void KernelNodeVisitor::operator()(MDNode *N)
{
        if ( N->getNumOperands() < 1) return;
        Value * Op = N->getOperand(0);
        if ( Function * F = dyn_cast<Function>(Op)) {
                found_kernels.push_back(F); 
        }
}

/* Call functor for each MDNode located within the Named MDNode */
void visitMDNodeOperands(NamedMDNode * N, KernelNodeVisitor& visitor)
{
        for (unsigned operand = 0, end = N->getNumOperands();
             operand < end; ++operand) {
                visitor(N->getOperand(operand));
        }
}

/* Accumulate LLVM functions that are kernels within the
   found_kernels vector. Return true if kernels are found.
   False otherwise. */
bool findKernels(Module& M, FunctionVect& found_kernels)
{
        NamedMDNode * root = getKernelListMDNode(M);
        if (!root || (root->getNumOperands() == 0)) return false;

        KernelNodeVisitor visitor(found_kernels);
        visitMDNodeOperands(root, visitor);

        return found_kernels.size() != 0;
}

class EraseNonkernels : public ModulePass {
        FunctionVect foundKernels;
        public:
        static char ID;
        EraseNonkernels();
        virtual ~EraseNonkernels();
        void getAnalysisUsage(AnalysisUsage& AU) const;
        bool runOnModule(Module& M);
};
} // ::<unnamed> namespace

EraseNonkernels::EraseNonkernels() : ModulePass(ID)
{}

EraseNonkernels::~EraseNonkernels()
{}

void EraseNonkernels::getAnalysisUsage(AnalysisUsage& AU) const
{
        AU.addRequired<CallGraph>();
}

bool EraseNonkernels::runOnModule(Module &M)
{
        findKernels(M, foundKernels);

        typedef Module::iterator func_iterator;
        for (func_iterator F = M.begin(), Fend = M.end();
             F != Fend; ++F) {
                typedef FunctionVect::const_iterator kernel_iterator;
                bool isKernel = false;
                if (!foundKernels.empty()) {
                        for (kernel_iterator KernFunc = foundKernels.begin(), KernFuncEnd = foundKernels.end();
                             KernFunc != KernFuncEnd; ++KernFunc) {
                                if (*KernFunc == &*F) {
                                        isKernel = true;
                                }
                        }
                }
                        
                if (isKernel) continue;

                F->deleteBody();
        }

        foundKernels.clear();
        return true;
}

char EraseNonkernels::ID = 0;
static RegisterPass<EraseNonkernels>
Y("erase-nonkernels", "Erase body of all functions not marked kernel in metadata.");
