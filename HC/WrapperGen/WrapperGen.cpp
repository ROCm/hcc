#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace std;
#define EOL << "\n"

namespace
{
  enum RangeOptions
  {
    PARAMETERS,
    INITIALIZE,
    ARGUMENTS,
    DECLARE
  };

  void printRange(raw_ostream &out,
                  vector<pair<string,string>> v,
                  vector<pair<string,string>>::const_iterator begin,
                  vector<pair<string,string>>::const_iterator end,
                  RangeOptions rop=PARAMETERS)
  {
    string delim("");
    if(rop == DECLARE)
      delim.append("; ");
    else delim.append(", ");

    // DECLARE checks for the last element on the list to print a delimiter
    vector<pair<string,string>>::const_iterator last = end;
    last--;

    for(vector<pair<string,string>>::const_iterator i = begin, e = end; i != e; ++i)
    {
      if(i != begin)
        out << delim;
      switch(rop)
      {
        case PARAMETERS:
          out << get<0>(*i) << " " << get<1>(*i);
          break;
        case INITIALIZE:
        {
          string arg(get<1>(*i));
          out << arg << "(" << arg << ")";
          break;
        }
        case ARGUMENTS:
          out << get<1>(*i);
          break;
        case DECLARE:
          out << get<0>(*i) << " " << get<1>(*i);
          if(i == last)
            out << delim;
          break;
        default:
          break;
      }
    }
  }

  void pointerAsterix(string &str, Type* Ty, Type*& T)
  {
    if(Ty->isPointerTy())
    {
      Type* nextTy = Ty->getSequentialElementType();
      if(!nextTy->isStructTy())
      {
        str.append("*");
        pointerAsterix(str, nextTy, T);
      }
      else T = nextTy;
    }
    else T = Ty;
  }

   // Alternative to the hash table?
  //
  string typeToString(Type* Ty)
  {
    string str("");
    Type* T = NULL;

    pointerAsterix(str, Ty, T);
    assert(T && "T is not NULL");

    if(IntegerType * intTy = dyn_cast<IntegerType>(T))
    {
      unsigned bitwidth = intTy->getBitWidth();
      switch(bitwidth)
      {
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
        default:
          break;
      };
    }

    if(T->isFloatingPointTy())
    {
      if(T->isFloatTy())
        str.insert(0, "float");
      if (T->isDoubleTy())
        str.insert(0, "double");
    }

    if(StructType * sTy = dyn_cast<StructType>(T))
      str.insert(0, sTy->getName().substr(7));

    return str;
  }

  struct WrapperGen : public ModulePass
  {
    static char ID;
    WrapperGen() : ModulePass(ID) {
    }

    bool runOnModule(Module &M) override
    {
      // Write to stderr
      // To save to a file, redirect to stdout, discard old stdout and write with tee
      // 2>&1 >/dev/null | tee output.cpp
      raw_ostream & out = errs();

      // headers and namespace uses
      out << "#include \"amp.h\"" EOL;
      out << "extern \"C\" {" EOL
             << "#include \"hip.h\"" EOL
             << "}" EOL;
      out << "using namespace concurrency;" EOL;

      // Find functions with attribute: grid_launch
      for(Module::iterator F = M.begin(), F_end = M.end(); F != F_end; ++F)
      {
        if(F->hasFnAttribute(Attribute::HCGridLaunch))
        {
          string funcName = F->getName().str();
          string wrapperStr = "__hcLaunchKernel_" + funcName;
          string functorName = F->getName().str() + "_functor";

          // get arguments from kernel
          vector<pair<string,string>> argList;
          const Function::ArgumentListType &Args(F->getArgumentList());
          for (Function::ArgumentListType::const_iterator i = Args.begin(), e = Args.end(); i != e; ++i)
          {
            Type* Ty = i->getType();
            string argType("");

            // Get type as string
            string tyName = typeToString(Ty);
            argType.append(tyName);

            // check if const
            if(i->onlyReadsMemory() && (tyName != "grid_launch_parm"))
              argType.insert(0, "const ");

            pair<string,string> arg = make_pair(argType, i->getName());
            argList.push_back(arg);
          }

          // Let's assume first argument has to be a grid_launch_parm Type
          assert(get<0>(argList[0]).compare("%struct.grid_launch_parm*") &&
                 "First argument of kernel must be of grid_launch_parm type");
          vector<pair<string,string>>::const_iterator i2 = argList.begin();
          ++i2;

          // extern kernel definition
          out << "extern \"C\"" EOL
                 << "__attribute__((always_inline)) void "
                 << funcName
                 << "(";
                    printRange(out, argList, argList.begin(), argList.end());
          out << ");"
                 EOL;

          // functor
          out << "namespace\n{\nstruct " << functorName << "\n{" EOL;
          out << functorName << "(";
          out << "grid_launch_parm _lp, ";
          printRange(out, argList, i2, argList.end());
          out << ") :" EOL;
          printRange(out, argList, i2, argList.end(), INITIALIZE);
          out << "{" EOL;
          out << "lp.gridDim.x = _lp.gridDim.x;" EOL;
          out << "lp.gridDim.y = _lp.gridDim.y;" EOL;
          out << "lp.groupDim.x = _lp.groupDim.x;" EOL;
          out << "lp.groupDim.y = _lp.groupDim.y;" EOL;
          out << "}" EOL;

          out << "void operator()(index<1> i) restrict(amp)\n{" EOL;
          out << "lp.groupId.x = (i[0] / lp.groupDim.x) % lp.gridDim.x;" EOL;
          out << "lp.groupId.y = i[0] / (lp.gridDim.x*lp.groupDim.x * lp.groupDim.y);" EOL;
          out << "lp.threadId.x = i[0] % lp.groupDim.x;" EOL;
          out << "lp.threadId.y = (i[0] / (lp.gridDim.x*lp.groupDim.x)) % lp.groupDim.y;" EOL;
          out << funcName << "(lp, ";
          printRange(out, argList, i2, argList.end(), ARGUMENTS);
          out << ");\n}" EOL;

          printRange(out, argList, argList.begin(), argList.end(), DECLARE);
          out << "\n};\n}" EOL;

          // wrapper
          out << "extern \"C\"" EOL;
          out << "void " << wrapperStr << "(";
          printRange(out, argList, argList.begin(), argList.end(), PARAMETERS);
          out << ")\n{" EOL;
          out << "parallel_for_each(extent<1>(lp.gridDim.x*lp.groupDim.y * lp.gridDim.y*lp.groupDim.y), " << functorName << "(";
          printRange(out, argList, argList.begin(), argList.end(), ARGUMENTS);
          out << "));\n}" EOL;
        }
      }
        return false;
    }
  };
}

char WrapperGen::ID = 0;
static RegisterPass<WrapperGen> X("gensrc", "Generate a wrapper and functor source file from input source.", false, false);

