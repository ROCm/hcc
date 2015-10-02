#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
using namespace llvm;
using namespace std;

namespace {

  struct DirectFuncCall : public ModulePass
  {
    static char ID;
    DirectFuncCall() : ModulePass(ID) {
    }
    bool runOnModule(Module &M) override {

      const char* const HCGridLaunchAttr = "hc_grid_launch";

      // Find functions with attribute: grid_launch
      for(Module::iterator F = M.begin(), F_end = M.end(); F != F_end; ++F)
      {
        if(F->hasFnAttribute(HCGridLaunchAttr))
        {
          // Attribute::NoInline is used to find the user of the function.
          // If inline is used, either forced or through optimziation, then this
          // pass will not be able to find a user to replace.
          // Whether or not users were found, this pass will reinstate inlining
          F->removeFnAttr(Attribute::NoInline);
          F->addFnAttr(Attribute::AlwaysInline);

          if(!F->hasNUses(0))
          {
            string funcName = F->getName().str();
            string wrapperName = "__hcLaunchKernel_" + funcName;

            ValueToValueMapTy VMap; // unused
            Function* wrapperFunc = CloneFunction(F, VMap, true);
            wrapperFunc->setName(wrapperName);
            wrapperFunc->deleteBody();
            // AttributeSet does not have a direct way of removing string attributes
            // Using AttrBuilder to do so
            AttrBuilder B(wrapperFunc->getAttributes(), AttributeSet::FunctionIndex);
            B.removeAttribute(HCGridLaunchAttr);
            wrapperFunc->setAttributes(AttributeSet::get(F->getContext(), AttributeSet::FunctionIndex, B));
            M.getFunctionList().push_back(wrapperFunc);

            // find uses of kernel proper
            for(Value::user_iterator U = F->user_begin(), U_end = F->user_end(); U != U_end; ++U)
            {
              if(CallInst* ci = dyn_cast<CallInst>(*U))
                ci->setCalledFunction(wrapperFunc);
            }
          } // !F->hasNUses > 0
        } // F->hasFnAttribute(HCGridLaunchAttr)
      } // Module::iterator

      errs() << M;
      return false;
    }
  };

}

char DirectFuncCall::ID = 0;
static RegisterPass<DirectFuncCall> X("redirect", "Redirect kernel function call to wrapper.", false, false);
