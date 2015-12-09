#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace
{

  class WrapperType {
    public:
      WrapperType(llvm::Type *Ty, bool isConst=false, bool isByVal=false) {
        // ConvertTypeToString may change these values
        mIsConst = isConst;
        mIsByVal = isByVal;
        mPointerCount = 0;
        mIsCustomType = false;

        mTypeName = ConvertTypeToString(Ty);
      }

      std::string getTypeName() const {
        return mTypeName;
      }

      std::string getTypeNameWithQual() const {
        std::string str(mTypeName);
        // FIXME: Fix const correctness
        if(mIsConst && (mTypeName != "grid_launch_parm"))
          str.insert(0, "const ");
        if(mPointerCount && !mIsByVal) {
          for(unsigned int i = 0; i < mPointerCount; i++)
            str.append(" * ");
        }
        return str;
      }

      bool isCustomType() {
        return mIsCustomType;
      }

      std::vector<WrapperType* > getStructContainerList() {
        return mStructContainer;
      }

    private:
      void FindUnderlyingPointerType(llvm::Type *Ty, llvm::Type *&T);
      void GetCustomTypeElements(llvm::StructType *sTy);
      std::string ConvertTypeToString(llvm::Type *Ty);
      std::string mTypeName;
      bool mIsConst;
      bool mIsByVal;
      unsigned int mPointerCount;
      bool mIsCustomType;

      std::vector<WrapperType* > mStructContainer;
  };

  class WrapperArgument {
    public:
      WrapperArgument(const llvm::Argument &A) {
        mType = new WrapperType(A.getType(), A.onlyReadsMemory(), A.hasByValAttr());

        mArgName = A.getName();
        auto st = mArgName.find(".coerce");
        if(st != std::string::npos)
          mArgName.erase(st, std::strlen(".coerce"));

      }

      std::string getArgName() const {
        return mArgName;
      }

      WrapperType *getType() const {
        return mType;
      }

      std::string getTypeName() const {
        return mType->getTypeName();
      }

      std::string getTypeNameWithQual() const {
        return mType->getTypeNameWithQual();
      }

      bool isCustomType() {
        return mType->isCustomType();
      }

      std::vector<WrapperType* > getStructContainerList() {
        return mType->getStructContainerList();
      }

    private:
      WrapperType *mType;
      std::string mArgName;

  };

  class WrapperFunction {
    public:
      WrapperFunction(llvm::Function *F) {
        mFunctionName = F->getName().str();
        mFunctorName = mFunctionName + "_functor";
        mWrapperName = "__hcLaunchKernel_" + mFunctionName;
      }

      void insertArgument(WrapperArgument* A) {
        mArgs.push_back(A);
      }

      void printArgsAsParameters(llvm::raw_ostream &out) {
        printRange(out, PARAMETERS);
      }

      void printArgsAsParametersInConstructor(llvm::raw_ostream &out) {
        printRange(out, PARAMETERSCONSTR);
      }

      void printArgsAsInitializers(llvm::raw_ostream &out) {
        printRange(out, INITIALIZE);
      }

      void printArgsAsArguments(llvm::raw_ostream &out) {
        printRange(out, ARGUMENTS);
      }

      void printArgsAsDeclarations(llvm::raw_ostream &out) {
        printRange(out, DECLARE);
      }

      void getUnderlyingStructDefs(std::vector<WrapperType* > structList) {
        for(auto sTy: structList){

          if(sTy->isCustomType())
          {
            getUnderlyingStructDefs(sTy->getStructContainerList());
            mCustomTypes.push_back(sTy);
          }
        }
      }

      std::vector<WrapperType* > getListUniqueTypesInFunction() {
        for(auto i: mArgs) {
          if(i->isCustomType()) {
            auto structList = i->getStructContainerList();
            getUnderlyingStructDefs(structList);
            mCustomTypes.push_back(i->getType());
          }
        }
        return mCustomTypes;
      }

      std::string getFunctionName() const {
        return mFunctionName;
      }

      std::string getFunctorName() const {
        return mFunctorName;
      }

      std::string getWrapperName() const {
        return mWrapperName;
      }

   private:
      enum RangeOptions {
        PARAMETERS,
        PARAMETERSCONSTR,
        INITIALIZE,
        ARGUMENTS,
        DECLARE
      };

      void printRange(llvm::raw_ostream &out, RangeOptions rop);

      std::vector<WrapperArgument* > mArgs;
      std::vector<WrapperType* > mCustomTypes;
      std::string mFunctionName;
      std::string mFunctorName;
      std::string mWrapperName;
  };

  class WrapperModule {
    public:
      void insertFuntion(WrapperFunction* F) {
        locateUniqueTypes(F);
        mFuncs.push_back(F);
      }
      std::vector<WrapperFunction* > getFunctionList() {
        return mFuncs;
      }
      void printCustomTypeDefinition(llvm::raw_ostream &out) {
        for(auto t: mCustomTypes) {
          auto structList = t->getStructContainerList();
          out << "struct " << t->getTypeName() << " { ";
          unsigned int membCount = 0;
          for(auto sTy: structList) {
            out << sTy->getTypeNameWithQual() << " m" << membCount << "; ";
            membCount++;
          }
          out << "};\n";
        }
      }
    private:
      void locateUniqueTypes(WrapperFunction* F);
      std::vector<WrapperFunction* > mFuncs;
      std::vector<WrapperType* > mCustomTypes;

  };

// Helper classes =========================================================== //

struct StringFinder
{
  StringFinder(const std::string &st) : s(st) { }
  bool operator()(const WrapperType *lhs) const {
    return lhs->getTypeName() == s;
  }

  std::string s;

};

// Class member function definitions ======================================== //

  void WrapperModule::locateUniqueTypes(WrapperFunction* F) {
    auto wt = F->getListUniqueTypesInFunction();
    for(auto t: wt) {
      if(std::find_if(mCustomTypes.begin(), mCustomTypes.end(), StringFinder(t->getTypeName())) == mCustomTypes.end())
        mCustomTypes.push_back(t);
    }
  }

  void WrapperFunction::printRange(llvm::raw_ostream &out, RangeOptions rop) {
    auto firstArg = mArgs.begin();
    auto lastArg = mArgs.end();

    std::string delim("");
    if(rop == DECLARE)
      delim.append("; ");
    else delim.append(", ");

    // Used for DECLARE only. Checks for last element on the list to print a delimiter
    auto secondLastArg = lastArg;
    secondLastArg--;

    // Used for INITIALIZE only. Omits printing delimiter for first and second arg
    auto secondArg = firstArg;
    secondArg++;

    for(auto i: mArgs) {
      if(rop == INITIALIZE && i != *firstArg && i != *secondArg)
        out << delim;
      else if(rop != INITIALIZE && i != *firstArg)
        out << delim;

      switch(rop) {
        // type function(type1 arg1, type1 arg2, type2 arg3)
        case PARAMETERS:
          out << i->getTypeNameWithQual() << " " << i->getArgName();
          break;
        // Class::Class(grid_launch_parm _lp, type1 arg2, type2 arg3)
        case PARAMETERSCONSTR:
          out << i->getTypeNameWithQual() << " ";
          if(i->getTypeName() == "grid_launch_parm")
            out << "_";
          out << i->getArgName();
          break;
        // : arg2(arg2), arg3(arg3) {}
        case INITIALIZE:
          if(i->getTypeName() != "grid_launch_parm")
            out << i->getArgName() << "(" << i->getArgName() << ")";
          break;
        // function(arg1, arg2, arg3);
        case ARGUMENTS:
          out << i->getArgName();
          break;
        // type1 arg1; type2 arg2; type1 arg3;
        case DECLARE:
          out << i->getTypeNameWithQual() << " " << i->getArgName();
          if(i == *secondLastArg)
            out << delim;
          break;
        default:
          break;
      }
    }
  }

  void WrapperType::FindUnderlyingPointerType(llvm::Type *Ty, llvm::Type *&T) {
    if(Ty->isPointerTy()) {
      mPointerCount++;
      llvm::Type *nextTy = Ty->getSequentialElementType();
      if(!mIsByVal) {
        FindUnderlyingPointerType(nextTy, T);
      }
      else T = nextTy;
    }
    else T = Ty;
  }

  std::string WrapperType::ConvertTypeToString(llvm::Type *Ty) {

    std::string str("");
    llvm::Type *T = NULL;

    FindUnderlyingPointerType(Ty, T);

    if(llvm::IntegerType *intTy = llvm::dyn_cast<llvm::IntegerType>(T)) {
      unsigned bitwidth = intTy->getBitWidth();
      switch(bitwidth) {
        case 1:
          str.insert(0, "bool");
          break;
        case 8:
          str.insert(0, "char");
          break;
        case 16:
          str.insert(0, "short");
          break;
        case 32:
          str.insert(0, "int");
          break;
        case 64:
          str.insert(0, "long");
          break;
        default:
          break;
      };
    }

    if(T->isFloatingPointTy()) {
      if(T->isFloatTy())
        str.insert(0, "float");
      if(T->isDoubleTy())
        str.insert(0, "double");
    }

    if(llvm::StructType *sTy = llvm::dyn_cast<llvm::StructType>(T)) {
      str.insert(0, sTy->getName());

      auto st = sTy->getName().find("class.");
      if(st != std::string::npos)
        str.erase(st, std::strlen("class."));
      else {
        st = sTy->getName().find("struct.");
        if(st != std::string::npos)
          str.erase(st, std::strlen("struct."));
      }

      // Rename struct so there won't be name conflicts during compilation
      // Linking should still resolve correctly as long as struct has POD members
      if(str != "grid_launch_parm") {
        str.append("_gl");

        mIsCustomType = true;
        GetCustomTypeElements(sTy);
      }
    }

    if(str == "")
      str.append("!UNKNOWN_TYPE_PLEASE_FIX!");

    return str;

  }

  void WrapperType::GetCustomTypeElements(llvm::StructType *sTy) {

    if(mIsCustomType) {

      for(auto e = sTy->element_begin(), e_end = sTy->element_end(); e != e_end; ++e) {
        llvm::Type *T = *e;

        mStructContainer.push_back(new WrapperType(T));
      }
    }
  }

  struct WrapperGen : public llvm::ModulePass
  {
    static char ID;
    WrapperGen() : llvm::ModulePass(ID) {
    }

    bool runOnModule(llvm::Module &M) override
    {
      // Write to stderr
      // To save to a file, redirect to stdout, discard old stdout and write with tee
      // 2>&1 >/dev/null | tee output.cpp
      llvm::raw_ostream & out = llvm::errs();

      // headers and namespace uses
      out << "#include \"hc.hpp\"\n";
      out << "#include \"grid_launch.h\"\n";

      out << "using namespace hc;\n";

      WrapperModule *Mod = new WrapperModule;

      // Find functions with attribute: grid_launch
      // Collect information
      for(auto F = M.begin(), F_end = M.end(); F != F_end; ++F) {
        if(F->hasFnAttribute("hc_grid_launch")) {
          WrapperFunction* func = new WrapperFunction(F);

          const llvm::Function::ArgumentListType &Args(F->getArgumentList());
          for (auto i = Args.begin(), e = Args.end(); i != e; ++i) {
            func->insertArgument(new WrapperArgument(*i));
          }

          Mod->insertFuntion(func);
        }
      }

      Mod->printCustomTypeDefinition(out);

      for(auto func: Mod->getFunctionList()) {
          // extern kernel definition
          out << "\nextern \"C\"" << "\n"
              << "__attribute__((always_inline)) void "
              << func->getFunctionName()
              << "(";
          func->printArgsAsParameters(out);
          out << ");\n";

          // functor
          out << "namespace\n{\nstruct " << func->getFunctorName() << "\n{\n";
          out << func->getFunctorName() << "(";
          func->printArgsAsParametersInConstructor(out);
          out << ") :\n";
          func->printArgsAsInitializers(out);
          out << "{\n";
          out << "lp.gridDim.x = _lp.gridDim.x;\n";
          out << "lp.gridDim.y = _lp.gridDim.y;\n";
          out << "lp.gridDim.z = _lp.gridDim.z;\n";
          out << "lp.groupDim.x = _lp.groupDim.x;\n";
          out << "lp.groupDim.y = _lp.groupDim.y;\n";
          out << "lp.groupDim.z = _lp.groupDim.z;\n";
          out << "}\n";

          out << "void operator()(tiled_index<3>& i) __attribute((hc))\n{\n";
          out << "lp.groupId.x = i.tile[0];\n";
          out << "lp.groupId.y = i.tile[1];\n";
          out << "lp.groupId.z = i.tile[2];\n";
          out << "lp.threadId.x = i.local[0];\n";
          out << "lp.threadId.y = i.local[1];\n";
          out << "lp.threadId.z = i.local[2];\n";
          out << func->getFunctionName() << "(";
          func->printArgsAsArguments(out);
          out << ");\n}\n";

          func->printArgsAsDeclarations(out);
          out << "\n};\n}\n";

          // wrapper
          out << "extern \"C\"\n";
          out << "void " << func->getWrapperName() << "(";
          func->printArgsAsParameters(out);
          out << ")\n{\n";
          out << "completion_future cf = parallel_for_each(*(lp.av),extent<3>(lp.gridDim.x*lp.groupDim.x,lp.gridDim.y*lp.groupDim.y,lp.gridDim.z*lp.groupDim.z).tile(lp.groupDim.x, lp.groupDim.y, lp.groupDim.z), "
              << func->getFunctorName()
              << "(";
          func->printArgsAsArguments(out);
          out << "));\n\n"
              << "if(lp.cf)\n"
              << "  *(lp.cf) = cf;\n"
              << "else\n"
              << "  cf.wait();\n"
              << "}\n";
      }
        return false;
    }
  };
}

char WrapperGen::ID = 0;
static llvm::RegisterPass<WrapperGen> X("gensrc", "Generate a wrapper and functor source file from input source.", false, false);

