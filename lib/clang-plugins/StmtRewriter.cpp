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
// routines from ParallelRewriter. The implementation will remove 'std::' text if any preceding
// call's name and add nested namespace"bolt::BKN::" instead. The BKN reprensets 
// Bolt's amp or cl code path. The amp code path is by default. All needed bolt headers 
// are also rewritten in the begining place of the source files. Other statments, like arguments in
// CallExpr or their delcarations also will be rewritten if any changes to them need to happen
// by using Bolt and its data structures.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/AST.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"

#include <fcntl.h>
#include <errno.h>
#include "ParallelRewriter.h"

using namespace clang;

// Default: using bolt's amp backend
#define RW_STL_2_BOLT_CALL           (1<<10)
#define RW_EXPAND_ARG_MACROS (1<<11)
// For extensions
#define RW_BOLT_CL_BACKEND        (1<<12) // N.A. for now
// For debug purpose
#define RW_BOLT_2_STL_CALL           (1<<20)

class CallsVisitor: public RecursiveASTVisitor<CallsVisitor> {
  // This maps an original source AST to it's rewritten form. This allows
  // us to avoid rewriting the same node twice (which is very uncommon).
  // This is needed to support some of the exotic property rewriting.
  llvm::DenseMap<Stmt *, Stmt *> ReplacedNodes;
  std::map<StringRef, bool> HeadersToAdd;
  ASTContext* Context;
  std::vector<StringRef> CallVec;
  std::map<StringRef, StringRef>Name2HeaderMap;
  CompilerInstance& Compiler;
  StringRef SourceFile;
  unsigned RWOpts;
  std::string BoltBKN;            // backend name as: "amp"
  std::string BoltNS;               // namespace as: "bolt::amp::"
  std::string BoltDirname;  // dirname with trailing '/' as: "bolt/amp/"

  // FIXME: when using clang::Rewriter in PluginASTAction, there might be potential 
  // multi-threading issues. The root cause is that std::map inside RewiterBuffers fails
  // in such occasions. We just copy codes from clang/lib/Rewrite/Core/Rewriter.cpp
  // and compile with clang isolately here
  parallel::Rewriter Rw;

public:
  CallsVisitor(CompilerInstance& CI, StringRef InFile, unsigned Options) 
    : Compiler(CI), SourceFile(InFile), RWOpts(Options), BoltBKN("amp") {
    Context = &CI.getASTContext();
    Rw.setSourceMgr(Context->getSourceManager(),Context->getLangOpts());
    if (RWOpts & RW_BOLT_CL_BACKEND)
      BoltBKN = "cl";
    BoltNS = "bolt::" + BoltBKN + "::";
    BoltDirname = "bolt/"+BoltBKN+ "/";
    if (CallVec.size() == 0) {
      #define ELIGIBLE_STL_CALL(Name, Header) CallVec.push_back(#Name); \
        Name2HeaderMap[#Name] = #Header;
      #include "EligibleSTLCallName.def"
    }
  }

  inline std::string FormBoltHeader( std::string HeaderName) {
    return "#include <" + BoltDirname + HeaderName + ">\n";
  }

  void AddBoltInclude(FileID FID) {
    SourceLocation LocFileStart = Rw.getSourceMgr().getLocForStartOfFile(FID);

    // Retrieve headers of this file based on AST and check if there is any "expected"
    // Bolt header is already included. This is to avoid adding duplicated Bolt headers
    Preprocessor& PP = Compiler.getPreprocessor();
    HeaderSearch& HS = PP.getHeaderSearchInfo();
    SmallVector<const FileEntry *, 16> Headers;
    // We shall cache Headers for better performance
    HS.getFileMgr().GetUniqueIDMapping(Headers);
    if (Headers.size() > HS.header_file_size())
      Headers.resize(HS.header_file_size());
    for (unsigned UID = 0, E = Headers.size(); UID != E; ++UID) {
      const FileEntry *File = Headers[UID];
      if (!File)
        continue;
      StringRef FullName = File->getName();
      StringRef FileName = llvm::sys::path::filename(FullName);
      std::map<StringRef, bool>::iterator It = HeadersToAdd.find(FileName);
      if ( It != HeadersToAdd.end() && It->second) {
        // FIMXE: check if contain BoltDirname+FileName
        if ( FullName.find(BoltDirname) != StringRef::npos)
          It->second = false;
        }
    }

    for (std::map<StringRef, bool>::iterator It = HeadersToAdd.begin(), 
      E = HeadersToAdd.end(); It!=E; It++) {
      if (It->second) {
        std::string Header = FormBoltHeader(It->first);
        bool OkWrite = !Rw.InsertText(LocFileStart, Header);
        if (!OkWrite) {
          llvm::errs() << "Adding headers fails at: \n";
          LocFileStart.dump(Context->getSourceManager());
          llvm::errs()<<"\n";
        }
      }
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
         
          // Indicate that we need to include this function's header
          HeadersToAdd[Name2HeaderMap[FuncName]] = true;

          FunctionDecl* BoltFunc = SynthesizeFunctionDecl(FD);
          CallExpr* NewExpr = SynthesizeCallToFunctionDecl(
             BoltFunc, E->getArgs(),E->getNumArgs(),SourceLocation(), SourceLocation());
          assert (NewExpr && "Null created CallExpr!");
          if (RWOpts & RW_EXPAND_ARG_MACROS ) {
            // Syntheize Bolt call by creating new FunctionDecl and rewrite CallExpr
            ReplaceStatement(E, NewExpr);
            return true;
          } else {
            // Ensure we only replace once
            if ( ReplacedNodes[E] )
              return true;
            ReplacedNodes[E] = NewExpr;

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
            NewCall += BoltNS;
            NewCall += FuncName;
            NewCall += StringRef("(");
            NewCall += Rw.getRewrittenText(ArgBodyWithRParen);
            bool OkWrite = !Rw.ReplaceText(E->getSourceRange(), StringRef(NewCall));
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
    CompilerInstance& Compiler;
    StringRef SourceFile;
    CallsVisitor Visitor;

public:
  FindStmt(CompilerInstance& CI, StringRef InFile, unsigned Options)
    : Compiler(CI), SourceFile(InFile), Visitor(CI, InFile, Options) { }

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
  unsigned RWOpts;

public:
  virtual clang::ASTConsumer *CreateASTConsumer(
    CompilerInstance& CI, StringRef InFile) {

    // FIXME: Compiler instance shall be valid at this time
    assert(CI.hasFileManager() && "File Manager is invalid!");
    assert(CI.hasASTContext() && "Context is invalid!");

    // Once the consumer is returned, our own implementation will be executed automatically
    // Also the consumer will be deleted automatically in the end by framework
    return new FindConsumer(CI, InFile, RWOpts);
  }

  // Load in plugin's arguments by specifying the following in Clang's cc1
  // for example,
  //   -Xclang -plugin-arg-StmtRewriter -Xclang -expandarg
  bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string>& args) {
    // Enable to rewrite STL calls with Bolt amp calls by default
    RWOpts |= RW_STL_2_BOLT_CALL;
    
    // TODO: Due to poisonous libstdc++ from cfe, args are not meaningful for now
    #if 0
    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      // TODO: Add specific arguments here to control plugin behavior
      if (args[i] == "-expandarg")
        RWOpts |= RW_EXPAND_ARG_MACROS;
      if (args[i] == "-clbkn")
        RWOpts |= RW_BOLT_CL_BACKEND;
       if (args[i] == "-bolt2stl")
        RWOpts |= RW_BOLT_2_STL_CALL;
    }
    #endif
    
    return true;
  }
};

static FrontendPluginRegistry::Add<StmtRewriterAction>
X("StmtRewriter", "Rewrite statements(AST-based)");

