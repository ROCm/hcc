#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"


// Check the new launch-parm arguments to make sure they have sane default values.
#define CHECK_LP_ARGS 0


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
        mArraySize = 0;

        mTypeName = ConvertTypeToString(Ty);
      }

      std::string getTypeName() const {
        return mTypeName;
      }

      std::string getTypeNameWithQual() const {
        std::string str(mTypeName);
        // FIXME: Fix const correctness
        if(mIsConst && (mTypeName != "grid_launch_parm_cxx"))
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

      unsigned int getArraySize() {
        return mArraySize;
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
      unsigned int mArraySize;

      std::vector<WrapperType* > mStructContainer;
  };

  class WrapperArgument {
    public:
      WrapperArgument(const llvm::Argument &A) {
        mType = new WrapperType(A.getType(), A.onlyReadsMemory(), A.hasByValAttr());

        mArgName = "_" + A.getName().str();
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

      void printArgsAsParametersKernel(llvm::raw_ostream &out) {
        printRange(out, PARAMETERSKERNEL);
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

      unsigned int getNumArgs() const {
        return mArgs.size();
      }

   private:
      enum RangeOptions {
        PARAMETERS,
        PARAMETERSKERNEL, // parameters to kernel
        PARAMETERSCONSTR, // parameters in constructor
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
            std::string strArray("");
            if(sTy->getArraySize() > 0) {
              strArray.append("[");
              strArray.append(std::to_string(sTy->getArraySize()));
              strArray.append("]");
            }
            // FIXME: Currently there seems to be a bug with char
            // Current workaround is to use bool since they both have the same size
            // https://bitbucket.org/snippets/wukevin/L8nbK
            std::string typeNameWithQual(sTy->getTypeNameWithQual());
            size_t start_pos = typeNameWithQual.find("char");
            if(start_pos != std::string::npos) {
              typeNameWithQual.replace(start_pos, strlen("char"), "bool");
            }

            out << typeNameWithQual << " m" << membCount << strArray << "; ";
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
    auto pastEnd = mArgs.end();

    std::string delim("");
    if(rop == DECLARE)
      delim.append("; ");
    else delim.append(", ");

    // Checks for last element on the list to print a delimiter
    auto lastArg = pastEnd;
    lastArg--;

    for(auto i: mArgs) {
      if(rop != INITIALIZE && i != *firstArg)
        out << delim;

      switch(rop) {
        // type function(type1 arg1, type1 arg2, type2 arg3)
        case PARAMETERS:
          if(i->getTypeName() == "grid_launch_parm_cxx") {
              //out << "const ";
          }
          out << i->getTypeNameWithQual() << " " << i->getArgName();
          break;
        // type function(grid_launch_parm lp, type1 arg2, type2 arg3)
        // retain grid_launch_parm for compatibility
        case PARAMETERSKERNEL:
          if(i->getTypeName() == "grid_launch_parm_cxx")
            out << "const grid_launch_parm" << " " << i->getArgName();
          else
            out << i->getTypeNameWithQual() << " " << i->getArgName();
          break;
        // Class::Class(grid_launch_parm _lp, type1 arg2, type2 arg3)
        case PARAMETERSCONSTR:
          if(i->getTypeName() == "grid_launch_parm_cxx") {
            out << "const " ;
            out << i->getTypeNameWithQual() << " ";
            out << "x";
          } else {
            out << i->getTypeNameWithQual() << " ";
          }
          out << i->getArgName();
          break;
        // : arg2(arg2), arg3(arg3) {}
        case INITIALIZE:
          if(i->getTypeName() != "grid_launch_parm_cxx") {
            out << i->getArgName() << "(" << i->getArgName() << ")";
            if (i != *lastArg)
              out << delim;
          }
          break;
        // function(arg1, arg2, arg3);
        case ARGUMENTS:
          out << i->getArgName();
          break;
        // type1 arg1; type2 arg2; type1 arg3;
        case DECLARE:
          out << i->getTypeNameWithQual() << " " << i->getArgName();
          if(i == *lastArg)
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

      // strip out 'struct." for grid_launch_parm
      if(sTy->getName() == "struct.grid_launch_parm") {
        auto st = sTy->getName().find("struct.");
        if(st != std::string::npos)
          str.erase(st, std::strlen("struct."));
        // use subclass which contains custom serializer
        str.append("_cxx");
      }

      // Remove other periods in the type name.
      std::replace(str.begin(), str.end(), '.', '_');
      std::replace(str.begin(), str.end(), ':', '_');
      std::replace(str.begin(), str.end(), '(', '_');
      std::replace(str.begin(), str.end(), ')', '_');
      std::replace(str.begin(), str.end(), ' ', '_');

      // Rename struct so there won't be name conflicts during compilation
      // Linking should still resolve correctly as long as struct has POD members
      if(str != "grid_launch_parm_cxx") {
        str.append("_gl");

        mIsCustomType = true;
        GetCustomTypeElements(sTy);
      }
    }

    if(T->isArrayTy()) {
      str.append(ConvertTypeToString(T->getArrayElementType()));
      mArraySize = T->getArrayNumElements();
    }

    if(str == "") {
      str.append("\'!UNKNOWN_TYPE: ");
      llvm::raw_string_ostream rso(str);
      T->print(rso);
      rso.flush();
      str.append("\'");
    }

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
      out << "#include \"grid_launch.hpp\"\n";

      out << "using namespace hc;\n";

      WrapperModule *Mod = new WrapperModule;

      // Find functions with attribute: grid_launch
      // Collect information
      for(auto F = M.begin(), F_end = M.end(); F != F_end; ++F) {
        if(F->hasFnAttribute("hc_grid_launch") && (F->size() > 0)) {
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
          func->printArgsAsParametersKernel(out);
          out << ");\n";

          // functor
          // This code is executed on the host :
          out << "namespace\n{\nstruct " << func->getFunctorName() << "\n{\n";
          out << func->getFunctorName() << "(";
          func->printArgsAsParametersInConstructor(out);
          out << ") ";
          if(func->getNumArgs() > 1) {
            out << ":\n";
            func->printArgsAsInitializers(out);
          }
          out << "{\n";
          out << "_lp.grid_dim.x = x_lp.grid_dim.x;\n";
          out << "_lp.grid_dim.y = x_lp.grid_dim.y;\n";
          out << "_lp.grid_dim.z = x_lp.grid_dim.z;\n";
          out << "_lp.group_dim.x = x_lp.group_dim.x;\n";
          out << "_lp.group_dim.y = x_lp.group_dim.y;\n";
          out << "_lp.group_dim.z = x_lp.group_dim.z;\n";
          out << "_lp.dynamic_group_mem_bytes = x_lp.dynamic_group_mem_bytes;\n";
#if CHECK_LP_ARGS
          out << "assert(x_lp.barrier_bit   ==  barrier_bit_queue_default);\n";  // remove when barrier-bit supported
          out << "assert(x_lp.launch_fence  == -1);\n";  // remove when launch_fence supported
#endif
          out << "}\n";


          // This code is in the compute kernel that is executed on the accelerator :
          out << "void operator()(tiled_index<3>& i) __attribute((hc))\n{\n";
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
          out << "if(!_lp.av) {\n";
          out << "  static accelerator_view av = accelerator().get_default_view();\n";
          out << "  _lp.av = &av;\n";
          out << "}\n\n";
          out << "completion_future cf;\n";
          out << "completion_future* cf_ptr = _lp.cf ? _lp.cf : &cf;\n";

          out << "*cf_ptr = parallel_for_each(*(_lp.av),extent<3>(_lp.grid_dim.z*_lp.group_dim.z,_lp.grid_dim.y*_lp.group_dim.y,_lp.grid_dim.x*_lp.group_dim.x).tile_with_dynamic(_lp.group_dim.z, _lp.group_dim.y, _lp.group_dim.x, _lp.dynamic_group_mem_bytes), \n"
          //out << "cf = parallel_for_each(*(_lp.av),extent<3>(_lp.grid_dim.z*_lp.group_dim.z,_lp.grid_dim.y*_lp.group_dim.y,_lp.grid_dim.x*_lp.group_dim.x).tile_with_dynamic(_lp.group_dim.z, _lp.group_dim.y, _lp.group_dim.x, _lp.dynamic_group_mem_bytes), \n"
              << func->getFunctorName()
              << "(";
          func->printArgsAsArguments(out);
          out << "));\n\n";

          //out << "printf (\"==_lp.cf=%p USE_COUNT=%d isReady=%d%c\", _lp.cf, cf_ptr->use_count(), cf_ptr->is_ready(), 0xA);\n";
          //out << "cf_ptr->wait();\n"; // bozo
          

          out << "}\n";
      }
        return false;
    }
  };
}

char WrapperGen::ID = 0;
static llvm::RegisterPass<WrapperGen> X("gensrc", "Generate a wrapper and functor source file from input source.", false, false);

