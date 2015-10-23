//===- RemoveSpecialSection.cpp - Remove special section information --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file removes special sections in global variables. It is tailored for
// AMDGPU backend as of now.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define _DEBUG (0)

namespace {
  struct RemoveSpecialSection : public ModulePass {
    static char ID;
    RemoveSpecialSection() : ModulePass(ID) {}
    bool runOnModule(Module& M) override {
      bool modified = false;
      for(Module::global_iterator GI = M.global_begin(), GE = M.global_end(); GI != GE; ++GI) {
#if _DEBUG
        llvm::errs() << "Global variable:\n";
        GI->dump();
#endif

        // remove all section information associated with global variables in a module
        if (GI->hasSection()) {
#if _DEBUG
          llvm::errs() << "Has section information: ";
          llvm::errs() << GI->getSection() << "\n";
#endif

          // remove section
          GI->setSection("");
          modified = true;

#if _DEBUG
          llvm::errs() << "After modification:\n";
          GI->dump();
#endif
        }
      }

      return modified;
    }
  };
}

char RemoveSpecialSection::ID = 0;
static RegisterPass<RemoveSpecialSection>
Z("remove-special-section", "Special Section Removal Pass", false, false);

