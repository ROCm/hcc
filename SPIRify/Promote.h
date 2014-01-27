namespace llvm {
  class PassRegistry;
  class ModulePass;
  void initializePromoteGlobalsPass(llvm::PassRegistry&);
  ModulePass * createPromoteGlobalsPass ();
}
