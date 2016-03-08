#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
using namespace llvm;
using namespace std;

namespace {

  cl::opt<bool> HostSpecific("host", cl::desc("Mark the operation as host specific"));

  struct DirectFuncCall : public ModulePass
  {
    static char ID;
    DirectFuncCall() : ModulePass(ID) {
    }
    bool runOnModule(Module &M) override {

      bool hasHCGridLaunchAttr = false;

      const char* const HCGridLaunchAttr = "hc_grid_launch";

      // Find functions with attribute: grid_launch
      for(Module::iterator F = M.begin(), F_end = M.end(); F != F_end; ++F)
      {
        if(F->hasFnAttribute(HCGridLaunchAttr))
        {
          hasHCGridLaunchAttr = true;

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

            Function* wrapperFunc = Function::Create(F->getFunctionType(), GlobalValue::ExternalLinkage, wrapperName, &M);

            // Simply replace all uses with new wrapper function
            F->replaceAllUsesWith(wrapperFunc);
          } // !F->hasNUses > 0

          // host side does not need the function definition of a grid_launch kernel
          // deleting the function body can help resolve linking errors
          if(HostSpecific) {
            if(F->size() > 0)
              F->deleteBody();
          }
        } // F->hasFnAttribute(HCGridLaunchAttr)
      } // Module::iterator

      if(hasHCGridLaunchAttr)
        errs() << M;
      return false;
    }
  };

}

char DirectFuncCall::ID = 0;
static RegisterPass<DirectFuncCall> X("redirect", "Redirect kernel function call to wrapper.", false, false);

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  return 0;
}
