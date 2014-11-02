//===---- PromoteStmt.cpp - Replacing statements with promoted counterparts-- ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file searchs STL calls in input files and replaces them with AMD Bolt calls by using
// utilites from RewriteHelper. The implementation will remove 'std::' text if any right before 
// call's name and add nested namespace"bolt::amp::" instead. All needed bolt headers are also
// rewritten in the begining place of the source files. Other statments, like arguments in
// CallExpr or their delcarations also will be rewritten if any changes to them need to happen
// by using Bolt and its data structures.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/AST.h"
#include "clang/AST/Attr.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/CommentDiagnostic.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/EvaluatedExprVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"

#include <fcntl.h>
#include <errno.h>
#include "ParallelRewriter.h"

using namespace clang;

#define RW_STL_2_BOLT_CALL           (1<<20)
// For debug purpose
#define RW_BOLT_2_STL_CALL           (1<<21)
#define RW_EXPAND_ARG_MACROS (1<<21)

class CallsVisitor: public RecursiveASTVisitor<CallsVisitor> {
  // This maps an original source AST to it's rewritten form. This allows
  // us to avoid rewriting the same node twice (which is very uncommon).
  // This is needed to support some of the exotic property rewriting.
  llvm::DenseMap<Stmt *, Stmt *> ReplacedNodes;

public:
  ASTContext* Context;

private:
    std::vector<StringRef> CallVec;
    StringRef SourceFile;
    unsigned RWOpts;

    // FIXME: when using clang::Rewriter in PluginASTAction, there might be potential 
    // multi-threading issues. The root cause is that std::map inside RewiterBuffers fails
    // in such occasions. We just copy codes from clang/lib/Rewrite/Core/Rewriter.cpp
    // and compile with clang isolately here
    parallel::Rewriter Rw;

public:
  CallsVisitor(ASTContext* Ctx, StringRef InFile, unsigned Options) 
    : Context(Ctx), SourceFile(InFile), RWOpts(Options) {
    Rw.setSourceMgr(Ctx->getSourceManager(),Ctx->getLangOpts());
    if (CallVec.size() == 0) {
      #define ELIGIBLE_STL_CALL(Name) CallVec.push_back(#Name);
      #include "EligibleSTLCallName.def"
    }
  }

  void AddBoltInclude(FileID FID) {
    SourceLocation LocFileStart = Rw.getSourceMgr().getLocForStartOfFile(FID);

    // FIXME: not sure why LocFileStart can't be replaced or be inserted.
    // Use LocOffset temporarily. This needs user codes has a blank "first line".
    SourceLocation LocOffset = LocFileStart.getLocWithOffset(1);
    
    std::string SearchPaths = StringRef("//#include <bolt/amp/binary_search.h>\n"
    "#include <bolt/amp/transform.h>\n"
    "#include <bolt/amp/device_vector.h>\n"
    "#include <bolt/amp/copy.h>\n"
    "#include <bolt/amp/fill.h>\n"
    "#include <bolt/amp/functional.h>\n"
    "#include <bolt/amp/gather.h>\n"
    "#include <bolt/amp/generate.h>\n"
    "//#include <bolt/amp/inner_product.h>\n"
    "#include <bolt/amp/max_element.h>\n"
    "#include <bolt/amp/merge.h>\n"
    "#include <bolt/amp/min_element.h>\n"
    "#include <bolt/amp/pair.h>\n"
    "//#include <bolt/amp/reduce.h>\n"
    "#include <bolt/amp/scatter.h>\n"
    "//#include <bolt/amp/sorts.h>\n"
    "//#include <bolt/amp/sort_by_key.h>\n"
    "#include <bolt/amp/transform_reduce.h>\n"
    "#include <bolt/amp/transform_scan.h>\n\n");
    
    // TODO: avoid duplicated
    bool OkWrite = !Rw.InsertText(LocOffset, StringRef(SearchPaths));
    if (!OkWrite) {
      llvm::errs() << "Adding headers fails at: \n";
      LocFileStart.dump(Context->getSourceManager());
      llvm::errs()<<"\n";
    }

  }

  int SaveRewrittenFiles() {
    for (parallel::Rewriter::buffer_iterator I = Rw.buffer_begin(),
                                 E = Rw.buffer_end(); I != E; ++I) {
      const FileEntry *Entry = Rw.getSourceMgr().getFileEntryForID(I->first);
      if (!Entry )
        continue;

      // Add Bolt headers
      if (I->second.size())
        AddBoltInclude(I->first);
      
      // Here we can't directly use raw_fd_ostream(const char *Filename, std::string &ErrorInfo,
      //                 unsigned Flags = 0) from raw_fd_ostream.cpp, for the following reasons:
      //
      // (1) raw_fd_ostream is built in libLLVMSupport by g++ with libstdc++
      //  the mangled name: _ZN4llvm14raw_fd_ostreamC1EPKcRSsj
      //  which is 
      //    llvm::raw_fd_ostream::raw_fd_ostream(char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> >&, unsigned int)
      //
      // (2) while the caller of it will be compiled by clang with libc++
      // the mangled name: _ZN4llvm14raw_fd_ostreamC1EPKcRNSt3__112basic_stringIcNS3_11char_traitsIcEENS3_9allocatorIcEEEEj
      // which is 
      //     llvm::raw_fd_ostream::raw_fd_ostream(char const*, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >&, unsigned int)
      // 
      // FIXME: No applicalbe way to solve the linking issue for now. Just use its other overloaded APIs 

      int fd;
      bool ShouldClose = false;
       int OpenFlags = O_WRONLY|O_CREAT;
       #ifdef O_BINARY
       OpenFlags |= O_BINARY;
       #endif
      while ((fd = open(Entry->getName(), OpenFlags, 0664)) < 0) {
        if (errno != EINTR) {
          ShouldClose = false;
          break;
        }
      }

      std::string ErrorInfo;
      llvm::raw_fd_ostream FileStream(fd, ShouldClose);
      if (!ErrorInfo.empty())
        return 1;
      I->second.write(FileStream);
      FileStream.flush();
    }

    return 0;
  }

  bool IsEligibleSTLCall(StringRef CallName) const {
     return (std::find(CallVec.begin(), CallVec.end(), CallName) != CallVec.end());
  }

  // FIXME: getQualifiedNameAsString does not work even by giving PringtingPolicy 
  // Borrow implementation of getQualifiedNameAsString from class NameDecl in here
  StringRef getFirstNamespace(FunctionDecl* FD) const {
    const DeclContext *Ctx = FD->getDeclContext();
    if (Ctx->isFunctionOrMethod()) {
      return StringRef();
    }

    typedef SmallVector<const DeclContext *, 8> ContextsTy;
    ContextsTy Contexts;

    // Collect contexts.
    while (Ctx && isa<NamedDecl>(Ctx)) {
      Contexts.push_back(Ctx);
      Ctx = Ctx->getParent();
    }

    for (ContextsTy::reverse_iterator I = Contexts.rbegin(), E = Contexts.rend();
         I != E; ++I) {
      // The outest namespace
      if (const NamespaceDecl *ND = dyn_cast<NamespaceDecl>(*I)) {
        if (ND->isAnonymousNamespace())
          return StringRef("<anonymous namespace>");
        else
          return ND->getName();
      }
    }

    return StringRef();
  }

  // FIXME: Since poisoned libstdc++ linked in libLLVM*, we can't use 
  // Rewriter::ReplaceStmt and other std::string, raw_os_stream related 
  // in our applications with libc++ linked
  void ReplaceStatement(Stmt *Old, Stmt* New) {
      Stmt *ReplacingStmt = ReplacedNodes[Old];

      if (ReplacingStmt)
        return; // We can't rewrite the same node twice.

      //if (DisableReplaceStmt)
      //  return;

      // Prepare new source string
      #if 0 
      // If replacement succeeded or warning disabled return with no warning.
      // Measaure the old text.
      int Size = Rw.getRangeSize(Old->getSourceRange());
      if (Size == -1)
        return;
      
      // If getPretty implementation is checked in
      // FIXME: StringRef has unexpected codes
      StringRef NewPretty = New->getPretty(*Context);
      Rw.ReplaceText(Old->getLocStart(), Size, NewPretty);
      ReplacedNodes[Old] = New;
      #else
      // If replacement succeeded or warning disabled return with no warning.
      if (!Rw.ReplaceStmt(Old, New)) {
        ReplacedNodes[Old] = New;
        return;
      }
      #endif
      
    }

  CallExpr * SynthesizeCallToFunctionDecl(
    FunctionDecl *FD, Expr **args, unsigned nargs, SourceLocation StartLoc,
                                                      SourceLocation EndLoc) {
    // Get the type, we will need to reference it in a couple spots.
    QualType FDTy = FD->getType();

    // Create a reference to the FD
    DeclRefExpr *DRE = new (Context) DeclRefExpr(FD, false, FDTy,
                                                 VK_LValue, SourceLocation());

    // Now, we cast the reference to a pointer to FDTy
    QualType pToFunc = Context->getPointerType(FDTy);
    ImplicitCastExpr *ICE = 
      ImplicitCastExpr::Create(*Context, pToFunc, CK_FunctionToPointerDecay,
                               DRE, 0, VK_RValue);

    const FunctionType *FT = FDTy->getAs<FunctionType>();

    CallExpr *Exp =  
      new (Context) CallExpr(*Context, ICE, llvm::makeArrayRef(args, nargs),
                             FT->getCallResultType(*Context),
                             VK_RValue, EndLoc);
    return Exp;
  }

  FunctionDecl * SynthesizeFunctionDecl(FunctionDecl* STLDecl) {
    DeclarationNameInfo NameInfo = STLDecl->getNameInfo();
    QualType FType = STLDecl->getType();

    // DeclContext, SourceLocation, StorageType, TypeSourceInfo are not necessarily 
    // meaningful during rewriting
    return FunctionDecl::Create(*Context, Context->getTranslationUnitDecl(),
                              SourceLocation(), SourceLocation(), NameInfo.getName(), FType, 
                              /*TInfo*/0, SC_Extern,
                              false, false);
  }
  
  // Handle directive: 'using namespace std;'
  bool VisitUsingDirectDecl(UsingDirectiveDecl* UD) {
    return true;
  }

  // Search potential STL calls in CallExpr and replace with "bolt::amp::" calls statement
  bool VisitCallExpr(CallExpr* E) {
    assert(E && "CallExpr is null!");
    if (E->getCallee()) {

      if (FunctionDecl* FD = E->getDirectCallee()) {
        StringRef FuncName = FD->getName();
        if ((RWOpts & RW_STL_2_BOLT_CALL) && IsEligibleSTLCall(FuncName) &&
          getFirstNamespace(FD) == StringRef("std")) {
          FunctionDecl* BoltFunc = SynthesizeFunctionDecl(FD);
          CallExpr* NewExpr = SynthesizeCallToFunctionDecl(
             BoltFunc, E->getArgs(),E->getNumArgs(),SourceLocation(), SourceLocation());
          if (RWOpts & RW_EXPAND_ARG_MACROS ) {
            // Syntheize Bolt call by creating new FunctionDecl and rewrite CallExpr
            ReplaceStatement(E, NewExpr);
          } else {
            // Syntheize Bolt call straightforward
            SourceRange ArgBodyWithRParen(E->getLocStart(), E->getRParenLoc());
            if(E->getNumArgs()) {
              SourceLocation Loc = E->getArg(0)->getLocStart();
              SourceLocation LocUserView = Context->getSourceManager().getExpansionLoc(Loc);
              ArgBodyWithRParen.setBegin(LocUserView);
            } else {
              ArgBodyWithRParen.setBegin(E->getRParenLoc());
            }
            std::string NewCall;
            NewCall += StringRef("bolt::amp::");
            NewCall += FuncName;
            NewCall += StringRef("(");
            NewCall += Rw.getRewrittenText(ArgBodyWithRParen);
            bool OkWrite = !Rw.ReplaceText( E->getSourceRange(), StringRef(NewCall));
            if (!OkWrite) {
              llvm::errs() << "Replacement fails at: \n";
              E->getLocStart().dump(Context->getSourceManager());
              llvm::errs()<<"\n";
           }
         }
       }
     }
   }

   return true;
  }

};

class FindStmt : public RecursiveASTVisitor<FindStmt> {
private:
    CompilerInstance& Compiler;
    StringRef SourceFile;
    CallsVisitor Visitor;

public:
  FindStmt(CompilerInstance& CI, StringRef InFile, unsigned Options)
    : Compiler(CI), SourceFile(InFile), Visitor(&CI.getASTContext(), InFile, Options) { }

   void Terminate() { Visitor.SaveRewrittenFiles(); }

  bool AreUserCodes( const SourceRange SR) const {
    FullSourceLoc SpellingBegin = Compiler.getASTContext().getFullLoc(SR.getBegin());
    StringRef FileName  = Compiler.getSourceManager().getFilename(SpellingBegin);
    return (FileName.equals(SourceFile));
  }

  // TODO: use RWOpts to control implemented virtual functions behavior
  // If Rewrite STL calls, this implementation is engough
  bool VisitFunctionDecl(FunctionDecl *FD) {
    // Only visit user codes
     if (AreUserCodes(FD->getSourceRange())) {
      Visitor.TraverseDecl(FD);
    }
    return true;
  }
};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.
class FindConsumer : public ASTConsumer {
private:
  FindStmt FindMe;

public:
  FindConsumer(CompilerInstance& CI, StringRef InFile, unsigned Options)
    : FindMe(CI, InFile, Options) {}

  virtual void HandleTranslationUnit(ASTContext &Context) {
    FindMe.TraverseDecl(Context.getTranslationUnitDecl());

    // Retwrite changes to physical source file on the disk as it ends
    FindMe.Terminate();
  }
};

// Implement action that can be invoked by the framework
class StmtRewriterAction : public PluginASTAction {
private:
  unsigned RWOpts;

public:
  virtual clang::ASTConsumer *CreateASTConsumer(
    CompilerInstance& CI, StringRef InFile) {

    // FIXME: Compiler instance shall be valid at this time
    assert(CI.hasFileManager() && "File Manager is invalid!");
    assert(CI.hasASTContext() && "Context is invalid!");

    // Enable to rewrite STL calls with Bolt calls by default
    RWOpts |= RW_STL_2_BOLT_CALL;

    // Once the consumer is returned, our own implementation will be executed automatically
    // Also the consumer will be deleted automatically in the end by framework
    return new FindConsumer(CI, InFile, RWOpts);
  }

  // Parse plugin's arguments if any
  bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string>& args) {
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      // TODO: Add specific arguments here to control plugin behavior
    }
    
    return true;
  }
};

static FrontendPluginRegistry::Add<StmtRewriterAction>
X("StmtRewriter", "Rewrite statements(AST-based)");

