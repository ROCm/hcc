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

  void pointerAsterix(string &str, Type* Ty, Type*& T, bool isByVal=false)
  {
    if(Ty->isPointerTy())
    {
      Type* nextTy = Ty->getSequentialElementType();
      if(!isByVal)
      {
        str.append("*");
        pointerAsterix(str, nextTy, T);
      }
      else T = nextTy;
    }
    else T = Ty;
  }

  // Returns string converted from type. Also returns bool if type is a struct pointer
  string typeToString(Type* Ty, bool& isStruct, bool isByVal=false)
  {
    string str("");
    Type* T = NULL;

    pointerAsterix(str, Ty, T, isByVal);
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
    {
      str.insert(0, sTy->getName().substr(7));
      isStruct = true;
    }

    return str;
  }

  void removeString(string& srcStr, const string str)
  {
    string::size_type strSzTy = srcStr.find(str);
    if(strSzTy != string::npos)
      srcStr.erase(strSzTy, str.size());
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
      out << "#include \"hc.hpp\"" EOL;
      out << "#include \"hip.h\"" EOL;

      out << "using namespace hc;" EOL;

      // Find functions with attribute: grid_launch
      for(Module::iterator F = M.begin(), F_end = M.end(); F != F_end; ++F)
      {
        if(F->hasFnAttribute("hc_grid_launch"))
        {
          string funcName = F->getName().str();
          string wrapperStr = "__hcLaunchKernel_" + funcName;
          string functorName = F->getName().str() + "_functor";

          vector<string> customTypes;
          // get arguments from kernel
          vector<pair<string,string>> argList;
          const Function::ArgumentListType &Args(F->getArgumentList());
          for (Function::ArgumentListType::const_iterator i = Args.begin(), e = Args.end(); i != e; ++i)
          {
            bool isStruct = false;

            Type* Ty = i->getType();
            string argType("");

            // Get type as string and check if type is a struct pointer
            bool hasByVal = i->hasByValAttr();
            string tyName = typeToString(Ty, isStruct, hasByVal);
            argType.append(tyName);

            // check if const
            if(i->onlyReadsMemory() && (tyName != "grid_launch_parm"))
              argType.insert(0, "const ");

            pair<string,string> arg = make_pair(argType, i->getName());
            argList.push_back(arg);

            // Only support struct pointers for now
            if(isStruct && !hasByVal)
              customTypes.push_back(argType);
          }

          // Let's assume first argument has to be a grid_launch_parm Type
          assert(get<0>(argList[0]).compare("%struct.grid_launch_parm*") &&
                 "First argument of kernel must be of grid_launch_parm type");
          vector<pair<string,string>>::const_iterator i2 = argList.begin();
          ++i2;

          // print forward declaration of custom types
          // only support pointers for now
          for(auto i : customTypes)
          {
            removeString(i, "*");
            removeString(i, "const ");
            out << "struct " << i << ";" EOL;
          }

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
          out << "lp.gridDim.z = _lp.gridDim.z;" EOL;
          out << "lp.groupDim.x = _lp.groupDim.x;" EOL;
          out << "lp.groupDim.y = _lp.groupDim.y;" EOL;
          out << "lp.groupDim.z = _lp.groupDim.z;" EOL;
          out << "}" EOL;

          out << "void operator()(tiled_index<3>& i) __attribute((hc))\n{" EOL;
          out << "lp.groupId.x = i.tile[0];" EOL;
          out << "lp.groupId.y = i.tile[1];" EOL;
          out << "lp.groupId.z = i.tile[2];" EOL;
          out << "lp.threadId.x = i.local[0];" EOL;
          out << "lp.threadId.y = i.local[1];" EOL;
          out << "lp.threadId.z = i.local[2];" EOL;
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
          out << "parallel_for_each(extent<3>(lp.gridDim.x*lp.groupDim.x,lp.gridDim.y*lp.groupDim.y,lp.gridDim.z*lp.groupDim.z).tile(lp.groupDim.x, lp.groupDim.y, lp.groupDim.z), " << functorName << "(";
          printRange(out, argList, argList.begin(), argList.end(), ARGUMENTS);
          out << ")).wait();\n}" EOL;
        }
      }
        return false;
    }
  };
}

char WrapperGen::ID = 0;
static RegisterPass<WrapperGen> X("gensrc", "Generate a wrapper and functor source file from input source.", false, false);

