//===- Promote.cpp - Lift LLVM IR to SPIR-compatible ----- ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to promote variables to correct addrspace to
// adhere SPIR specification.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "PromoteGlobals"

#include "Promote.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <map>
#include <set>
using namespace llvm;
#ifdef __APPLE__
#define TILE_STATIC_NAME "clamp,opencl_local"
#else
#define TILE_STATIC_NAME "clamp_opencl_local"
#endif

namespace {

  /* Data type container into which the list of LLVM functions
     that are OpenCL kernels will be stored */
  typedef SmallVector<Function *, 3> FunctionVect;
  typedef SmallVector<std::pair<Function *, Function *>, 3> FunctionPairVect;
  typedef std::map <Function *, Function *> FunctionMap;

  /* The name of the MDNode into which the list of
     MD nodes referencing each OpenCL kernel is stored. */
  static Twine KernelListMDNodeName = "opencl.kernels";

  enum {
    PrivateAddressSpace = 0,
    GlobalAddressSpace = 1,
    ConstantAddressSpace = 2,
    LocalAddressSpace = 3
  };

  class InstUpdateWorkList;

  class InstUpdate {
  public:
    virtual ~InstUpdate() {}
    virtual void operator()(InstUpdateWorkList*) =0;
  };

  Instruction * updateInstructionWithNewOperand(Instruction * I,
                                                Value * oldOperand,
                                                Value * newOperand,
                                                InstUpdateWorkList * updatesNeeded);

  void updateListWithUsers ( Value::user_iterator U,
                             const Value::user_iterator& Ue,
                             Value * oldOperand, Value * newOperand,
                             InstUpdateWorkList * updates );
  /* This structure hold information on which instruction
     should be updated with a new operand. The subject is
     the instruction that need to be updated. OldOperand
     is the current operand within the subject that needs
     to be updated. The newOperand is the value that needs
     to be substituted for the oldOperand value. The Old
     operand value is provided to identify which operand
     needs to be updated, as it may not be trivial to
     identify which operand is affected. In addition, the same
     value may be used multiple times by an instruction.
     Rather than allocating this structure multiple time,
     only one is used. */
  class ForwardUpdate : public InstUpdate {
  public:
    ForwardUpdate(Instruction * s, Value * oldOperand, Value * newOperand);
    void operator()(InstUpdateWorkList *);
    static void UpdateUsersWithNewOperand(Instruction *s, Value * oldOperand, Value * newOperand, InstUpdateWorkList * workList);
  private:
    Instruction * subject;
    Value * oldOperand;
    Value * newOperand;
  };

  ForwardUpdate::ForwardUpdate(Instruction *s,
                               Value *oldOp, Value *newOp)
    : subject(s), oldOperand(oldOp), newOperand(newOp)
  {}

  void ForwardUpdate::operator()(InstUpdateWorkList * workList)
  {
    DEBUG(llvm::errs() << "F: "; subject->dump(););
    updateInstructionWithNewOperand (subject,
                                     oldOperand,
                                     newOperand,
                                     workList);
    // Instruction *oldInst = dyn_cast<Instruction>(oldOperand);
    // if ( oldInst->use_empty () ) {
    //oldInst->eraseFromParent ();
    // }
  }

  void ForwardUpdate::UpdateUsersWithNewOperand (Instruction * Insn,
                                                 Value * oldOperand,
                                                 Value * newOperand,
                                                 InstUpdateWorkList * workList)
  {
    updateListWithUsers ( Insn->user_begin (), Insn->user_end(),
                          oldOperand, newOperand,
                          workList );
  }

  class BackwardUpdate : public InstUpdate {
  public:
    BackwardUpdate(Instruction * upstream, Type* expected);
    void operator()(InstUpdateWorkList *updater);
    static void setExpectedType(Instruction * Insn, Type * expected,
                                InstUpdateWorkList * updater);
  private:
    Instruction * upstream;
    Type * expected;
  };

  class InstUpdateWorkList {
  public:
    ~InstUpdateWorkList();
    bool run();
    bool empty() const;
    void addUpdate(InstUpdate * update);
  private:
    typedef std::vector<InstUpdate *> WorkListTy;
    WorkListTy workList;
  };

  BackwardUpdate::BackwardUpdate (Instruction * insn, Type *type)
    : upstream (insn), expected (type)
  {}

  void updateBackAllocaInst(AllocaInst * AI, Type * expected,
                            InstUpdateWorkList * updater)
  {
    PointerType * ptrType = dyn_cast<PointerType> (expected);
    if ( !ptrType ) {
      DEBUG(llvm::errs() << "Was expecting a pointer type. Got ";
            expected->dump(););
    }

    AllocaInst * newInst = new AllocaInst (ptrType->getElementType(),
                                           AI->getArraySize (), "",
                                           AI);

    ForwardUpdate::UpdateUsersWithNewOperand (AI, AI, newInst,
                                              updater);
  }

  Type * patchType ( Type * baseType, Type* patch, User::op_iterator idx, User::op_iterator idx_end);

  Type * patchType ( Type * baseType, Type* patch, User::op_iterator idx, User::op_iterator idx_end)
  {
    if ( idx == idx_end ) {
      return patch;
    }

    bool isIndexLiteral = false;
    uint64_t literalIndex = 0;
    if ( ConstantInt * CI = dyn_cast<ConstantInt>(*idx) ) {
      literalIndex = CI->getZExtValue ();
      isIndexLiteral = true;
    }

    if ( StructType * ST = dyn_cast<StructType>(baseType ) ) {
      if ( !isIndexLiteral ) {
        llvm::errs() << "Expecting literal index for struct type\n";
        return NULL;
      }
      std::vector<Type *> newElements;
      for (unsigned elem = 0, last_elem = ST->getNumElements();
           elem != last_elem; ++elem) {
        Type * elementType = ST->getElementType (elem);
        if ( elem != literalIndex ) {
          newElements.push_back (elementType);
          continue;
        }
        Type * transformed = patchType (elementType, patch,
                                        ++idx, idx_end);
        newElements.push_back (transformed);
      }
      return StructType::get (ST->getContext(),
                              ArrayRef<Type *>(newElements),
                              ST->isPacked());
    } else if ( ArrayType *AT = dyn_cast<ArrayType>(baseType) ) {
      if (!isIndexLiteral) {
        llvm::errs() << "Expecting literal index for array type\n";
        return NULL;
      }
      Type *transformed = patchType(AT->getElementType(), patch, ++idx, idx_end);
      return ArrayType::get(transformed, AT->getNumElements());
    }
    DEBUG(llvm::errs() << "Patch type not handling ";
          baseType->dump(););
    return NULL;
  }

  void updateBackGEP (GetElementPtrInst * GEP, Type* expected,
                      InstUpdateWorkList * updater)
  {
    DEBUG(llvm::errs() << "=== BEFORE UPDATE BACK GEP ===\n";
          llvm::errs() << "EXPECTED TYPE: "; expected->dump(); llvm::errs() << "\n";
          llvm::errs() << "Source operand type: "; GEP->getPointerOperandType()->dump(); llvm::errs() << "\n";);

    PointerType * ptrExpected = dyn_cast<PointerType> (expected);
    if ( !ptrExpected ) {
      llvm::errs() << "Expected type for GEP is not a pointer!\n";
      return;
    }
    PointerType * ptrSource =
      dyn_cast<PointerType> (GEP->getPointerOperand()->getType());
    if ( !ptrSource ) {
      llvm::errs() << "Source operand type is not a pointer!\n";
      return;
    }

    DEBUG(llvm::errs() << "Element type: "; ptrSource->getElementType()->dump(); llvm::errs() << "\n";
          llvm::errs() << "Expected element type: "; ptrExpected->getElementType()->dump(); llvm::errs() << "\n";
          for (User::op_iterator it = GEP->idx_begin(), ie = GEP->idx_end(); it != ie; ++it) {
          if ( ConstantInt * CI = dyn_cast<ConstantInt>(*it) ) {
          llvm::errs() << " idx " << CI->getZExtValue();
          }
          }
          llvm::errs() << "\n";);

    User::op_iterator first_idx = GEP->idx_begin();
    ++first_idx;
    Type * newElementType = patchType (ptrSource->getElementType(),
                                       ptrExpected->getElementType(),
                                       first_idx, GEP->idx_end());


    // be aware that newElementType might be null
    if (newElementType) {
      DEBUG(llvm::errs() << "NEW ELEMENT TYPE: "; newElementType->dump(); llvm::errs() << "\n";);
      PointerType  * newUpstreamType =
        PointerType::get(newElementType,
                         ptrExpected->getAddressSpace());
      Instruction * ptrProducer =
        dyn_cast<Instruction>(GEP->getPointerOperand());
      // sometimes the pointer operand is an argument and does not need update
      if (ptrProducer)
        BackwardUpdate::setExpectedType (ptrProducer,
                                         newUpstreamType, updater);
    } else {
      DEBUG(llvm::errs() << "newElementType is null\n";);
    }
  }

  void updateBackLoad (LoadInst * L, Type * expected,
                       InstUpdateWorkList * updater)
  {
    Value * ptrOperand = L->getPointerOperand();
    Instruction * ptrSource = dyn_cast<Instruction>(ptrOperand);

    // sometimes the pointer operand is an argument and does not need further processing
    if (!ptrSource) return;

    PointerType * sourceType =
      dyn_cast<PointerType> (ptrOperand->getType());
    assert(sourceType
           && "Load ptr operand's type is not a pointer type");

    PointerType * newPtrType =
      PointerType::get(expected,
                       sourceType->getAddressSpace());

    BackwardUpdate::setExpectedType(ptrSource, newPtrType, updater);
  }

  void updateBackBitCast (BitCastInst * BCI, Type * expected,
                          InstUpdateWorkList * updater)
  {
    DEBUG(BCI->dump(););
    Type * srcType = BCI->getSrcTy();
    PointerType * ptrSrcType = dyn_cast<PointerType> (srcType);
    assert (ptrSrcType
            && "Unexpected non-pointer type as source operand of bitcast");

    Type * destType = BCI->getDestTy();
    PointerType * ptrDestType = dyn_cast<PointerType> (destType);
    assert (ptrDestType
            && "Unexpected non-pointer type as dest operand of bitcast");

    Type * srcElement = ptrSrcType->getElementType ();
    StructType *srcElementStructType = dyn_cast<StructType> (srcElement);
    if ( !srcElementStructType ) {
      DEBUG(llvm::errs () << "Do not know how handle bitcast\n";);
      return;
    }

    Type * dstElement = ptrDestType->getElementType ();
    StructType *dstElementStructType = dyn_cast<StructType> (dstElement);
    if ( !dstElementStructType ) {
      DEBUG(llvm::errs () << "Do not know how handle bitcast\n";);
      return;
    }

    bool sameLayout =
      srcElementStructType->isLayoutIdentical(dstElementStructType);
    if ( !sameLayout ) {
      DEBUG(llvm::errs() << "Different layout in bitcast!\n";);
      return;
    }

    Instruction *sourceOperand = dyn_cast<Instruction>(BCI->getOperand(0));
    if ( !sourceOperand ) {
      DEBUG(llvm::errs() << "Do not know how to handle"
            " non-instruction source operand\n";);
    }
    BitCastInst * newBitCast =
      new BitCastInst(sourceOperand,
                      expected, "", BCI);

    ForwardUpdate::UpdateUsersWithNewOperand (BCI, BCI, newBitCast,
                                              updater);
    BackwardUpdate::setExpectedType(sourceOperand, expected, updater);
    return;


  }

  void BackwardUpdate::operator ()(InstUpdateWorkList *updater)
  {
    DEBUG(llvm::errs() << "B: "; upstream->dump(););
    if (!upstream) return;

    if ( AllocaInst * AI = dyn_cast<AllocaInst> (upstream) ) {
      updateBackAllocaInst (AI, expected, updater);
      return;
    }
    if ( GetElementPtrInst * GEP = dyn_cast<GetElementPtrInst>(upstream) ) {
      updateBackGEP (GEP, expected, updater);
      return;
    }
    if ( LoadInst * LI = dyn_cast<LoadInst>(upstream) ) {
      updateBackLoad (LI, expected, updater);
      return;
    }
    if ( BitCastInst * BCI = dyn_cast<BitCastInst>(upstream) ) {
      updateBackBitCast (BCI, expected, updater);
      return;
    }
    if ( isa<PHINode>(upstream)) {
      DEBUG(llvm::errs() << "[BackwardUpdate::operator()] TBD PHINode\n";);
      return;
    }

    DEBUG(llvm::errs() << "Do not know how to update ";
          upstream->dump(); llvm::errs() << " with "; expected->dump();
          llvm::errs() << "\n";);
    return;
  }

  void BackwardUpdate::setExpectedType (Instruction * Insn, Type * expected,
                                        InstUpdateWorkList * update)
  {
    update->addUpdate (new BackwardUpdate (Insn, expected));
  }



  bool InstUpdateWorkList::run()
  {
    bool didSomething = false;
    while( !workList.empty() ) {
      InstUpdate * update = workList.back();
      workList.pop_back();
      (*update) (this);
      delete update;
      didSomething = true;
    }
    return didSomething;
  }

  bool InstUpdateWorkList::empty() const
  {
    return workList.empty ();
  }

  void InstUpdateWorkList::addUpdate(InstUpdate * update)
  {
    workList.push_back(update);
  }

  InstUpdateWorkList::~InstUpdateWorkList ()
  {
    for ( WorkListTy::iterator U = workList.begin(), Ue = workList.end();
          U != Ue; ++U ) {
      delete *U;
    }
  }
  Function * createPromotedFunctionToType ( Function * F, FunctionType * promoteType);

  /* Find the MDNode which reference the list of opencl kernels.
     NULL if it does not exists. */

  NamedMDNode * getKernelListMDNode (Module & M)
  {
    return M.getNamedMetadata(KernelListMDNodeName);
  }

  NamedMDNode * getNewKernelListMDNode (Module & M)
  {
    NamedMDNode * current = getKernelListMDNode (M);
    if ( current ) {
      M.eraseNamedMetadata (current);
    }
    return M.getOrInsertNamedMetadata (KernelListMDNodeName.str());
  }

  Type * mapTypeToGlobal ( Type *);

  namespace {
    std::map<StructType*, StructType*> structTypeMap;

    // a mapping of old types to newly translated types
    std::map<Type *, Type *> translatedTypeMap;
  }

  StructType* mapTypeToGlobal(StructType* T) {
    // create a new, empty StructType
    StructType* newST = nullptr;
    if (T->hasName()) {
      newST = StructType::create(T->getContext(), T->getName());
    } else {
      newST = StructType::create(T->getContext(), "");
    }

    // mark the original StructType as translated to the new StructType
    structTypeMap[T] = newST;

    // walk through each field in the old StructType, and translate them
    // for types which are already translated, directly lookup the result in structTypeMap
    // this way we can support recursive types, ex:
    // %T1 = type { i32, %T1* }
    std::vector<Type*> translatedTypes;
    for (unsigned elem = 0, last_elem = T->getNumElements();
         elem != last_elem; ++elem) {
      Type* baseType = T->getElementType(elem);

      Type* translatedType = nullptr;
      // test if baseType points to a type which has already been translated
      if (PointerType* PT = dyn_cast<PointerType>(baseType)) {
        if (StructType* pointedStructType = dyn_cast<StructType>(PT->getElementType())) {
          if (structTypeMap.find(pointedStructType) != structTypeMap.end()) {
            translatedType = PointerType::get(structTypeMap[pointedStructType], GlobalAddressSpace);
          }
        }
      }

      // use normal type translation logic if otherwise
      if (translatedType == nullptr) {
        translatedType = mapTypeToGlobal(baseType);

        // associate the newly created translated type with the old type
        translatedTypeMap[baseType] = translatedType;
      }

      translatedTypes.push_back(translatedType);
    }

    // set the fields of the new StrucType
    newST->setBody(ArrayRef<Type*>(translatedTypes), T->isPacked());

    return newST;
  }

  ArrayType * mapTypeToGlobal ( ArrayType * T )
  {
    Type* translatedType = mapTypeToGlobal(T->getElementType());
    return ArrayType::get(translatedType, T->getNumElements());
  }

  PointerType * mapTypeToGlobal ( PointerType * PT )
  {
    Type * translatedType = mapTypeToGlobal ( PT->getElementType());
    return PointerType::get ( translatedType, GlobalAddressSpace );
  }

  SequentialType * mapTypeToGlobal ( SequentialType * T ) {
    ArrayType * AT = dyn_cast<ArrayType> (T);
    if ( AT ) return mapTypeToGlobal (AT);

    PointerType * PT = dyn_cast<PointerType> (T);
    if ( PT ) return mapTypeToGlobal (PT);

    return T;
  }

  CompositeType * mapTypeToGlobal (CompositeType * T)
  {
    StructType * ST = dyn_cast<StructType> (T);
    if ( ST ) return mapTypeToGlobal ( ST );

    SequentialType * SQ = dyn_cast<SequentialType> (T);
    if ( SQ ) return mapTypeToGlobal ( SQ );

    DEBUG (llvm::errs () << "Unknown type "; T->dump(); );
    return T;
  }

  Type * mapTypeToGlobal (Type * T)
  {
    CompositeType * C = dyn_cast<CompositeType>(T);
    if ( !C ) return T;
    return mapTypeToGlobal (C);
  }

  /* Create a new function type based on the provided function so that
     each arguments that are pointers, or pointer types within composite
     types, are pointer to global */

  FunctionType * createNewFunctionTypeWithPtrToGlobals (Function * F)
  {
    FunctionType * baseType = F->getFunctionType();

    std::vector<Type *> translatedArgTypes;

    // keep track of mapping of the original type and translated type

    unsigned argIdx = 0;
    for (Function::arg_iterator A = F->arg_begin(), Ae = F->arg_end();
         A != Ae; ++A, ++argIdx) {
      Type * argType = baseType->getParamType(argIdx);
      Type * translatedType;

      // try use previously translated type
      translatedType = translatedTypeMap[argType];

      // create new translated type if there is none
      if (translatedType == nullptr) {

        StringRef argName = A->getName();
        if (argName.equals("scratch")
            || argName.equals("lds")
            || argName.equals("scratch_count")) {
          PointerType * Ptr = dyn_cast<PointerType>(argType);
          assert(Ptr && "Pointer type expected");
          translatedType = PointerType::get(Ptr->getElementType(),
                                            LocalAddressSpace);
        } else {
          if (A->hasByValAttr()) {
            PointerType * ptrType =
              cast<PointerType>(argType);
            Type * elementType =
              ptrType->getElementType();
            Type * translatedElement =
              mapTypeToGlobal(elementType);
            translatedType =
              PointerType::get(translatedElement,
                               0);
          } else {
            translatedType = mapTypeToGlobal (argType);
          }
        }

        // associate the newly created translated type with the old type
        translatedTypeMap[argType] = translatedType;
      }

      translatedArgTypes.push_back ( translatedType );
    }

    FunctionType * newType
      = FunctionType::get(mapTypeToGlobal(baseType->getReturnType()),
                          ArrayRef<Type *>(translatedArgTypes),
                          baseType->isVarArg());
    return newType;
  }

  void nameAndMapArgs (Function * newFunc, Function * oldFunc, ValueToValueMapTy& VMap)
  {
    typedef Function::arg_iterator iterator;
    for (iterator old_arg = oldFunc->arg_begin(),
         new_arg = newFunc->arg_begin(),
         last_arg = oldFunc->arg_end();
         old_arg != last_arg; ++old_arg, ++new_arg) {
      VMap[old_arg] = new_arg;
      new_arg->setName(old_arg->getName());

    }
  }

  BasicBlock * getOrCreateEntryBlock (Function * F)
  {
    if ( ! F->isDeclaration() ) return &F->getEntryBlock();
    return BasicBlock::Create(F->getContext(), "entry", F);
  }

  AllocaInst * createNewAlloca(Type * elementType,
                               AllocaInst* oldAlloca,
                               BasicBlock * dest)
  {
    TerminatorInst * terminator = dest->getTerminator();
    if (terminator) {
      return new AllocaInst(elementType,
                            oldAlloca->getArraySize(),
                            oldAlloca->getName(),
                            terminator);
    }
    return new AllocaInst(elementType,
                          oldAlloca->getArraySize(),
                          oldAlloca->getName(),
                          dest);

  }

  void updateListWithUsers ( User *U, Value * oldOperand, Value * newOperand,
                             InstUpdateWorkList * updates )
  {
    Instruction * I1 = dyn_cast<Instruction>(U);
    if ( I1 ) {
      updates->addUpdate (
        new ForwardUpdate(I1,
                          oldOperand, newOperand ) );
    } else if (ConstantExpr * CE =
               dyn_cast<ConstantExpr>(U)) {
      DEBUG(llvm::errs()<<"CE:";
            CE->dump(););
      for(Value::user_iterator CU = CE->user_begin(),
          CUE = CE->user_end(); CU!=CUE;) {
        if (Instruction *I2 = dyn_cast<Instruction>(*CU)) {
          // patch all the users of the constexpr by
          // first producing an equivalent instruction that
          // computes the constantexpr

          // construct an Instruction from ConstantExpr
          I1 = CE->getAsInstruction();

          // properly order I1 and I2
          I1->insertBefore(I2);

          // update operand, a new Instruction may be created along the way
          Instruction *newI1 = updateInstructionWithNewOperand(I1, oldOperand, newOperand, updates);

          // updated Instruction may be a complete new Instruction
          // if that's the case, ditch the original one
          if (newI1 != I1) I1->eraseFromParent();

          // Let I2 use newI1 as its new operand
          updateInstructionWithNewOperand(I2, CE, newI1, updates);

          // CU is invalidated
          CU = CE->user_begin();
          continue;
        } else if (ConstantExpr *CE2 = dyn_cast<ConstantExpr>(*CU)) {
          // there are cases where a ConstantExpr is used by another ConstantExpr
          // in the current implementation, we assume there would be at most
          // 3 ConstantExpr used in an Instruction
          //
          // assuming the original Instruction is:
          // I3(CE2(CE))
          //
          // we would rewrite it to:
          // I1 (built from CE)
          // I2 (built from CE2, with I1 as operand)
          // I3 (replace CE2 with I2)

          for (Value::user_iterator CU2 = CE2->user_begin(), CUE2 = CE2->user_end(); CU2 != CUE2;) {
            if (Instruction *I3 = dyn_cast<Instruction>(*CU2)) {
              I1 = CE->getAsInstruction();
              Instruction *I2 = CE2->getAsInstruction();

              // properly order I1, I2, I3
              I2->insertBefore(I3);
              I1->insertBefore(I2);

              // for each new Instruction, update its operand
              // notice additional Instruction may be produced along the
              // way. In that case the original Instruction would be
              // removed.
              Instruction *newI1 = updateInstructionWithNewOperand(I1, oldOperand, newOperand, updates);
              if (newI1 != I1) I1->eraseFromParent();
              Instruction *newI2 = updateInstructionWithNewOperand(I2, CE, newI1, updates);
              if (newI2 != I2) I2->eraseFromParent();
              updateInstructionWithNewOperand(I3, CE2, newI2, updates);

              // CU2 is invalidated
              CU2 = CE2->user_begin();
              continue;
            } else if (ConstantExpr *CE3 = dyn_cast<ConstantExpr>(*CU2)) {

              // in another case if the original Instruction is:
              // I4(CE3(CE2(CE)))
              //
              // we would rewrite it to:
              // I1 (built from CE)
              // I2 (built from CE2, with I1 as operand)
              // I3 (built from CE3, with I2 as operand)
              // I4 (replace CE3 with I3)

              for (Value::user_iterator CU3 = CE3->user_begin(), CUE3 = CE3->user_end(); CU3 != CUE3;) {

                if (Instruction *I4 = dyn_cast<Instruction>(*CU3)) {
                  // construct Instruction instances
                  I1 = CE->getAsInstruction();
                  Instruction *I2 = CE2->getAsInstruction();
                  Instruction *I3 = CE3->getAsInstruction();

                  // properly order new Instruction instances
                  I3->insertBefore(I4);
                  I2->insertBefore(I3);
                  I1->insertBefore(I2);

                  // for each new Instruction, update its operand
                  // notice additional Instruction may be produced along the
                  // way. In that case the original Instruction would be
                  // removed.
                  Instruction *newI1 = updateInstructionWithNewOperand(I1, oldOperand, newOperand, updates);
                  if (newI1 != I1) I1->eraseFromParent();
                  Instruction *newI2 = updateInstructionWithNewOperand(I2, CE, newI1, updates);
                  if (newI2 != I2) I2->eraseFromParent();
                  Instruction *newI3 = updateInstructionWithNewOperand(I3, CE2, newI2, updates);
                  if (newI3 != I3) I3->eraseFromParent();
                  updateInstructionWithNewOperand(I4, CE3, newI3, updates);

                  // CU3 is invalidated
                  CU3 = CE3->user_begin();
                  continue;
                }
                ++CU3;
              }
            }
            ++CU2;
          }
        }
        ++CU;
      }
    }
  }
  void updateListWithUsers ( Value::user_iterator U,
                             const Value::user_iterator& Ue,
                             Value * oldOperand, Value * newOperand,
                             InstUpdateWorkList * updates )
  {
    for ( ; U != Ue; ++U ) {
      updateListWithUsers(*U, oldOperand, newOperand, updates);
    }
  }

  Instruction * updateLoadInstWithNewOperand(LoadInst * I, Value * newOperand,
                                             InstUpdateWorkList * updatesNeeded)
  {
    Type * originalLoadedType = I->getType();
    I->setOperand(0, newOperand);
    PointerType * PT = cast<PointerType>(newOperand->getType());
    if ( PT->getElementType() != originalLoadedType ) {
      I->mutateType(PT->getElementType());
      updateListWithUsers(I->user_begin(), I->user_end(), I, I, updatesNeeded);
    }
    return I;
  }

  Instruction * updatePHINodeWithNewOperand(PHINode * I, Value * oldOperand,
                                            Value * newOperand,
                                            InstUpdateWorkList * updatesNeeded)
  {
    DEBUG(llvm::errs() << "=== BEFORE UPDATE PHI ===\n";
          I->dump();
          llvm::errs() << "original type: "; I->getType()->dump(); llvm::errs() << "\n";
          for (unsigned i = 0; i < I->getNumIncomingValues(); ++i) {
          llvm::errs() << "src#" << i << ": "; I->getIncomingValue(i)->getType()->dump(); llvm::errs() << "\n";
          }
          llvm::errs() << "=========================\nnew type: "; newOperand->getType()->dump(); llvm::errs() << "\n";);

    // Update the PHI node itself as well its users
    Type * originalType = I->getType();
    PointerType * PT = cast<PointerType>(newOperand->getType());
    if ( PT != originalType ) {
      I->mutateType(PT);
      updateListWithUsers(I->user_begin(), I->user_end(), I, I, updatesNeeded);
    }

    /* update other incoming nodes as well */
    for (unsigned i = 0; i < I->getNumIncomingValues(); ++i) {
      Value *V = I->getIncomingValue(i);

      if (V == oldOperand) {
        I->setOperand(i, newOperand);
      } else if (V == newOperand) {
        continue;
      } else {
        Type *Ty = V->getType();
        if (isa<PointerType>(Ty)) {
          PointerType *PTy = dyn_cast<PointerType>(Ty);
          if (PT != PTy) {
            if (isa<Instruction>(V)) {
              Instruction *II = dyn_cast<Instruction>(V);

              DEBUG(llvm::errs() << "value#" << i << " update type from: ";  PTy->dump();
                    llvm::errs() << " to: "; PT->dump();
                    llvm::errs() << " for instruction: "; II->dump(); llvm::errs() << "\n";);

              V->mutateType(PT);
              updateListWithUsers(II->user_begin(), II->user_end(), II, II, updatesNeeded);
            } else if (isa<Constant>(V)) {
              Constant *CC = dyn_cast<Constant>(V);

              DEBUG(llvm::errs() << "value#" << i << " update type from: ";  PTy->dump();
                    llvm::errs() << " to: "; PT->dump();
                    llvm::errs() << " for constant: "; CC->dump(); llvm::errs() << "\n";);

              if (CC->isNullValue()) {
                I->setOperand(i, Constant::getNullValue(PT));
              } else {
                DEBUG(llvm::errs() << "unhandled constant\n";);
              }
            }
          }
        }
      }
    }

    DEBUG(llvm::errs() << "=== AFTER UPDATE PHI ====\n";
          I->dump();
          llvm::errs() << "type: "; I->getType()->dump(); llvm::errs() << "\n";
          for (unsigned i = 0; i < I->getNumIncomingValues(); ++i) {
          llvm::errs() << "src#" << i << ": "; I->getIncomingValue(i)->getType()->dump(); llvm::errs() << "\n";
          }
          llvm::errs() << "\n\n";);
    return I;
  }

  Instruction * updateStoreInstWithNewOperand(StoreInst * I,
                                              Value * oldOperand,
                                              Value * newOperand,
                                              InstUpdateWorkList * updatesNeeded)
  {
    unsigned index = I->getOperand(1) == oldOperand?1:0;
    I->setOperand(index, newOperand);
    Value * storeOperand = I->getPointerOperand();
    PointerType * destType =
      dyn_cast<PointerType>(storeOperand->getType());

    if ( destType->getElementType ()
         == I->getValueOperand()->getType() ) return I;


    if ( index == StoreInst::getPointerOperandIndex () ) {
      DEBUG(llvm::errs() << "Source value should be updated\n";);
      DEBUG(llvm::errs() << " as "; I->getValueOperand()->dump();
            llvm::errs() << " is stored in "; I->getPointerOperand()->dump(););
    } else {
      PointerType * newType =
        PointerType::get(I->getValueOperand()->getType(),
                         destType->getAddressSpace());

      DEBUG(llvm::errs() << "newtype: "; newType->dump(); llvm::errs() << "\n";
            llvm::errs() << "pointer operand type: "; I->getPointerOperand()->getType()->dump(); llvm::errs() << "\n";);

      if (isa<Instruction>(I->getPointerOperand())) {
        Instruction * ptrProducer =
          dyn_cast<Instruction> ( I->getPointerOperand () );

        BackwardUpdate::setExpectedType (ptrProducer,
                                         newType, updatesNeeded);

      } else {
        DEBUG(llvm::errs() << "ptrProducer is null\n";);
      }
    }

    return I;
  }

  Instruction * updateCallInstWithNewOperand(CallInst * CI, Value * oldOperand,
                                             Value * newOperand,
                                             InstUpdateWorkList * updatesNeeded)
  {
    for ( unsigned i = 0, numArgs = CI->getNumArgOperands();
          i != numArgs; ++i ) {
      if ( CI->getArgOperand ( i ) == oldOperand ) {
        CI->setArgOperand ( i, newOperand );
      }
    }

    return CI;
  }

  Instruction * updateBitCastInstWithNewOperand(BitCastInst * BI,
                                                Value *oldOperand,
                                                Value * newOperand,
                                                InstUpdateWorkList * updatesNeeded)
  {
    Type * currentType = BI->getType();
    PointerType * currentPtrType = dyn_cast<PointerType>(currentType);
    if (!currentPtrType) return BI;

    // make sure pointers inside the casted type are also promoted
    // this fixes an issue when a class has a vtbl gets captured in the kernel,
    // the size of vtbl would not be correctly calculated
    Type *elementType = currentPtrType->getElementType();
    if (StructType *ST = dyn_cast<StructType>(elementType)) {
      elementType = mapTypeToGlobal(ST);
    }

    Type * sourceType = newOperand->getType();
    PointerType * sourcePtrType = dyn_cast<PointerType>(sourceType);
    if (!sourcePtrType) return BI;

    PointerType * newDestType =
      PointerType::get(elementType,
                       sourcePtrType->getAddressSpace());

    BitCastInst * newBCI = new BitCastInst (newOperand, newDestType,
                                            "", BI);

    updateListWithUsers (BI->user_begin(), BI->user_end(),
                         BI, newBCI, updatesNeeded);

    return newBCI;
  }

  Instruction * updateAddrSpaceCastInstWithNewOperand(AddrSpaceCastInst * AI,
                                                      Value *oldOperand,
                                                      Value * newOperand,
                                                      InstUpdateWorkList * updatesNeeded)
  {
    Type * currentType = AI->getType();
    PointerType * currentPtrType = dyn_cast<PointerType>(currentType);
    if (!currentPtrType) return AI;

    Type * sourceType = newOperand->getType();
    PointerType * sourcePtrType = dyn_cast<PointerType>(sourceType);
    if (!sourcePtrType) return AI;

    if ( sourcePtrType->getAddressSpace()
         == currentPtrType->getAddressSpace() ) {
      Value *nV = AI->getOperand(0);
      AI->replaceAllUsesWith(nV);
      AI->eraseFromParent();
      return AI;
    }

    PointerType * newDestType =
      PointerType::get(currentPtrType->getElementType(),
                       sourcePtrType->getAddressSpace());

    AddrSpaceCastInst * newACI = new AddrSpaceCastInst (newOperand, newDestType,
                                                        "", AI);

    updateListWithUsers (AI->user_begin(), AI->user_end(),
                         AI, newACI, updatesNeeded);

    return newACI;
  }

  Instruction * updateGEPWithNewOperand(GetElementPtrInst * GEP,
                                        Value * oldOperand,
                                        Value * newOperand,
                                        InstUpdateWorkList * updatesNeeded)
  {
    DEBUG(llvm::errs() << "=== BEFORE UPDATE GEP ===\n";
          llvm::errs() << "new operand: "; newOperand->getType()->dump(); llvm::errs() << "\n";);

    if ( GEP->getPointerOperand() != oldOperand ) return GEP;

    std::vector<Value *> Indices(GEP->idx_begin(), GEP->idx_end());

    Type * futureType =
      GEP->getGEPReturnType(newOperand, ArrayRef<Value *>(Indices));

    DEBUG(llvm::errs() << "future type: "; futureType->dump(); llvm::errs() << "\n";
          llvm::errs() << "address space: " << GEP->getAddressSpace() << "\n";
          llvm::errs() << "indexed type: "; GEP->getIndexedType(oldOperand->getType(), Indices)->dump(); llvm::errs() << "\n";);

    PointerType * futurePtrType = dyn_cast<PointerType>(futureType);
    if ( !futurePtrType ) return GEP;

    GEP->setOperand ( GEP->getPointerOperandIndex(), newOperand);

    if ( futurePtrType == GEP->getType()) return GEP;

    GEP->mutateType ( futurePtrType );
    updateListWithUsers(GEP->user_begin(), GEP->user_end(), GEP, GEP, updatesNeeded);
    return GEP;
  }

  Instruction * updateCMPWithNewOperand(CmpInst *CMP, Value *oldOperand,
                                        Value *newOperand,
                                        InstUpdateWorkList *updatesNeeded)
  {
    DEBUG(llvm::errs() << "=== BEFORE UPDATE CMP ===\n";
          llvm::errs() << " new type: "; newOperand->getType()->dump(); llvm::errs() << "\n";
          for (unsigned i = 0; i < CMP->getNumOperands(); ++i) {
          llvm::errs() << " op#" << i << ": "; CMP->getOperand(i)->getType()->dump(); llvm::errs() << "\n";
          });

    bool update = false;
    for (unsigned i = 0; i < CMP->getNumOperands(); ++i) {
      Value *V = CMP->getOperand(i);
      Type *T = V->getType();
      if (T != newOperand->getType()) {
        V->mutateType(newOperand->getType());
        updateListWithUsers(V->user_begin(), V->user_end(), V, V, updatesNeeded);
        update = true;
      }
    }
    if (update)
      updateListWithUsers(CMP->user_begin(), CMP->user_end(), CMP, CMP, updatesNeeded);
    return CMP;
  }

  Instruction * updateSELWithNewOperand(SelectInst * SEL, Value * oldOperand,
                                        Value * newOperand,
                                        InstUpdateWorkList * updatesNeeded)
  {
    DEBUG(llvm::errs() << "=== BEFORE UPDATE SEL ===\n";
          llvm::errs() << " new type: "; newOperand->getType()->dump(); llvm::errs() << "\n";
          llvm::errs() << " op1 type: "; SEL->getOperand(1)->getType()->dump(); llvm::errs() << "\n";
          llvm::errs() << " op2 type: "; SEL->getOperand(2)->getType()->dump(); llvm::errs() << "\n";);

    Type* InstructionType =  SEL->getType();
    bool update = false;
    if(SEL->getOperand(1) == oldOperand) {
      SEL->setOperand (1, newOperand);

      Value *V2 = SEL->getOperand(2);
      Type *T2 = V2->getType();
      if (T2 != newOperand->getType()) {
        V2->mutateType(newOperand->getType());
        updateListWithUsers(V2->user_begin(), V2->user_end(), V2, V2, updatesNeeded);
        update = true;
      }
      if(InstructionType != newOperand->getType()) {
        SEL->mutateType(newOperand->getType());
        update = true;
      }
    }
    if(SEL->getOperand(2) == oldOperand) {
      SEL->setOperand (2, newOperand);

      Value *V1 = SEL->getOperand(1);
      Type *T1 = V1->getType();
      if (T1 != newOperand->getType()) {
        V1->mutateType(newOperand->getType());
        updateListWithUsers(V1->user_begin(), V1->user_end(), V1, V1, updatesNeeded);
        update = true;
      }
      if(InstructionType != newOperand->getType()) {
        SEL->mutateType(newOperand->getType());
        update = true;
      }
    }

    DEBUG(llvm::errs() << "=== AFTER UPDATE SEL ===\n";
          llvm::errs() << " op1 type: "; SEL->getOperand(1)->getType()->dump(); llvm::errs() << "\n";
          llvm::errs() << " op2 type: "; SEL->getOperand(2)->getType()->dump(); llvm::errs() << "\n";);

    if(update)
      updateListWithUsers(SEL->user_begin(), SEL->user_end(), SEL, SEL, updatesNeeded);

    return SEL;
  }

  bool CheckCalledFunction ( CallInst * CI, InstUpdateWorkList * updates,
                             FunctionType *& newFunctionType )
  {
    Function * CalledFunction = CI->getCalledFunction ();
    FunctionType * CalledType = CalledFunction->getFunctionType ();
    unsigned numParams = CalledType->getNumParams ();

    std::vector<Type *> newArgTypes;

    bool changeDetected = false;
    for ( unsigned param = 0; param < numParams; ++param ) {
      Type * paramType = CalledType->getParamType ( param );
      Value * argument = CI->getArgOperand ( param );
      Type * argType = argument->getType ();

      changeDetected |= ( paramType != argType );

      newArgTypes.push_back (argType);
    }

    if ( !changeDetected ) {
      return false;
    }

    Type * returnType = mapTypeToGlobal (CalledType->getReturnType());

    newFunctionType =
      FunctionType::get(returnType,
                        ArrayRef<Type *>(newArgTypes),
                        CalledType->isVarArg());
    return true;
  }

  void promoteCallToNewFunction (CallInst * CI, FunctionType * newFunctionType,
                                 InstUpdateWorkList * updates)
  {
    Function * CalledFunction = CI->getCalledFunction ();
    Function * promoted =
      createPromotedFunctionToType ( CalledFunction, newFunctionType);
    CI->setCalledFunction (promoted);

    Type * returnType = newFunctionType->getReturnType();
    if ( returnType == CI->getType () ) return;

    CI->mutateType (returnType);
    updateListWithUsers (CI->user_begin(), CI->user_end(),
                         CI, CI, updates);
  }

  void CollectChangedCalledFunctions (Function * F,
                                      InstUpdateWorkList * updatesNeeded)
  {
    typedef std::vector<CallInst *> CallInstsTy;
    CallInstsTy foundCalls;
    for (Function::iterator B = F->begin(), Be = F->end();
         B != Be; ++B) {
      for (BasicBlock::iterator I = B->begin(), Ie = B->end();
           I != Ie; ++I) {
        CallInst * CI = dyn_cast<CallInst>(I);
        if ( !CI ) continue;
        foundCalls.push_back(CI);
      }
    }
    typedef CallInstsTy::iterator iterator;
    typedef std::pair<CallInst *, FunctionType *> PromotionTy;
    typedef std::vector<PromotionTy> ToPromoteTy ;
    ToPromoteTy changedFunctions;
    for (iterator C = foundCalls.begin(), Ce = foundCalls.end();
         C != Ce; ++C) {
      FunctionType * newType;
      if ( !CheckCalledFunction ( *C, updatesNeeded, newType ) )
        continue;
      changedFunctions.push_back ( std::make_pair(*C, newType) );
    }

    for (ToPromoteTy::iterator C = changedFunctions.begin(),
         Ce = changedFunctions.end();
         C != Ce; ++C) {
      CallInst * CI = C->first;
      FunctionType *newType = C->second;
      IntrinsicInst * Intrinsic = dyn_cast<IntrinsicInst>(CI);
      if (!Intrinsic) {
        promoteCallToNewFunction(CI, newType, updatesNeeded);
        continue;
      }
      Intrinsic::ID IntrinsicId = Intrinsic->getIntrinsicID ();
      ArrayRef<Type *> Args(newType->param_begin(),
                            newType->param_begin()+3);
      Function * newIntrinsicDecl =
        Intrinsic::getDeclaration (F->getParent(),
                                   IntrinsicId,
                                   Args);
      DEBUG(llvm::errs() << "When updating intrinsic "; CI->dump(););
      CI->setCalledFunction (newIntrinsicDecl);
      DEBUG(llvm::errs() << " expecting: "
            << Intrinsic::getName(IntrinsicId, Args);
            CI->dump();
            llvm::errs() << CI->getCalledFunction()->getName() << "\n";);
    }
  }

  // HSA-specific : memory scope for atomic operations
  //
  // For atomic instructions (load atomic, store atomic, atomicrmw, cmpxchg, fence),
  // add !mem.scope metadata to specify its memory scope, which is required in HSAIL.
  // Since there is no way to specify memory scope in C++ atomic operations <atomic>
  // yet, we set default memory scope as: _sys_ (5)
  void appendMemoryScopeMetadata(Instruction *I) {
    // set default memory scope as: _sys_ (5)
    ConstantInt *C = ConstantInt::get(Type::getInt32Ty(I->getContext()), 5);
#if LLVM_VERSION_MAJOR == 3
#if (LLVM_VERSION_MINOR >= 3) && (LLVM_VERSION_MINOR <= 5)
    // logic which is compatible from LLVM 3.3 till LLVM 3.5
    MDNode *MD = MDNode::get(I->getContext(), C);
#elif LLVM_VERSION_MINOR > 5
    // support new MDTuple type introduced in LLVM 3.6+
    MDTuple *MD = MDTuple::get(I->getContext(), ConstantAsMetadata::get(C));
#else
#error Unsupported LLVM MINOR VERSION
#endif
#else
#error Unsupported LLVM MAJOR VERSION
#endif
    I->setMetadata("mem.scope", MD);
  }

  Instruction * updateInstructionWithNewOperand(Instruction * I,
                                                Value * oldOperand,
                                                Value * newOperand,
                                                InstUpdateWorkList * updatesNeeded)
  {
    if (LoadInst * LI = dyn_cast<LoadInst>(I)) {
      if (LI->isAtomic()) {
        appendMemoryScopeMetadata(I);
      }

      return updateLoadInstWithNewOperand(LI, newOperand, updatesNeeded);
    }

    if (StoreInst * SI = dyn_cast<StoreInst>(I)) {
      if (SI->isAtomic()) {
        appendMemoryScopeMetadata(I);
      }

      return updateStoreInstWithNewOperand(SI, oldOperand, newOperand, updatesNeeded);
    }

    if (CallInst * CI = dyn_cast<CallInst>(I)) {
      return updateCallInstWithNewOperand(CI, oldOperand, newOperand, updatesNeeded);
    }

    if (BitCastInst * BI = dyn_cast<BitCastInst>(I)) {
      return updateBitCastInstWithNewOperand(BI, oldOperand, newOperand, updatesNeeded);
    }

    if (AddrSpaceCastInst * AI = dyn_cast<AddrSpaceCastInst>(I)) {
      return updateAddrSpaceCastInstWithNewOperand(AI, oldOperand, newOperand, updatesNeeded);
    }

    if (GetElementPtrInst * GEP = dyn_cast<GetElementPtrInst>(I)) {
      return updateGEPWithNewOperand(GEP, oldOperand, newOperand, updatesNeeded);
    }

    if (PHINode * PHI = dyn_cast<PHINode>(I)) {
      return updatePHINodeWithNewOperand(PHI, oldOperand, newOperand, updatesNeeded);
    }

    if (SelectInst * SEL = dyn_cast<SelectInst>(I)) {
      return updateSELWithNewOperand(SEL, oldOperand, newOperand,updatesNeeded);
    }

    if (CmpInst *CMP = dyn_cast<CmpInst>(I)) {
      return updateCMPWithNewOperand(CMP, oldOperand, newOperand, updatesNeeded);
    }

    if (isa<PtrToIntInst>(I)) {
      DEBUG(llvm::errs() << "No need to update ptrtoint\n";);
      return I;
    }

    if (isa<InvokeInst>(I)) {
      DEBUG(llvm::errs() << "No need to update invoke\n";);
      return I;
    }

    if (isa<BranchInst>(I)) {
      DEBUG(llvm::errs() << "No need to update branch\n";);
      return I;
    }

    if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) || isa<FenceInst>(I)) {
      appendMemoryScopeMetadata(I);
      return I;
    }

    if (isa<ReturnInst>(I)) {
      DEBUG(llvm::errs() << "No need to update ret\n";);
      return I;
    }

    DEBUG(llvm::errs() << "DO NOT KNOW HOW TO UPDATE INSTRUCTION: ";
          I->print(llvm::errs()); llvm::errs() << "\n";);

    // Don't crash the program
    return I;
  }

  static bool usedInTheFunc(const User *U, const Function* F)
  {
    if (!U)
      return false;

    if (const Instruction *instr = dyn_cast<Instruction>(U)) {
      if (instr->getParent() && instr->getParent()->getParent()) {
        const Function *curFunc = instr->getParent()->getParent();
        if(curFunc->getName().str() == F->getName().str())
          return true;
        else
          return false;
      }
    }

#if LLVM_VERSION_MAJOR == 3
#if LLVM_VERSION_MINOR == 3
    for (User::const_use_iterator ui = U->use_begin(), ue = U->use_end();
         ui != ue; ++ui) {
      if (usedInTheFunc(*ui, F) == true)
        return true;
    }
#elif LLVM_VERSION_MINOR == 5
    for (const User *u : U->users()) {
      if (usedInTheFunc(u, F) == true)
        return true;
    }
#elif LLVM_VERSION_MINOR > 5
    for (const User *u : U->users()) {
      if (usedInTheFunc(u, F) == true)
        return true;
    }
#else
#error Unsupported LLVM MINOR VERSION
#endif
#else
#error Unsupported LLVM MAJOR VERSION
#endif

    return false;
  }

  // Tests if address of G is copied into global pointers, which are propageted
  // through arguments of F.
  bool isAddressCopiedToHost(const GlobalVariable &G, const Function &F) {
    std::vector<const User*> pending;
    std::set<const Value*> all_global_ptrs;

    for (const Value &arg : F.args()) {
      if (arg.getType()->isPointerTy()) {
        pending.insert(pending.end(), arg.user_begin(), arg.user_end());
        all_global_ptrs.insert(&arg);
      }
    }

    while (!pending.empty()) {
      const User *u = pending.back();
      pending.pop_back();

      if (isa<GetElementPtrInst>(u)) {
        all_global_ptrs.insert(u);
        pending.insert(pending.end(), u->user_begin(), u->user_end());
      }
      else if (isa<LoadInst>(u)) {
        all_global_ptrs.insert(u);
        pending.insert(pending.end(), u->user_begin(), u->user_end());
      }
      else if (const ConstantExpr *cexp = dyn_cast<ConstantExpr>(u)) {
        if (cexp->getOpcode() == Instruction::GetElementPtr) {
          all_global_ptrs.insert(u);
          pending.insert(pending.end(), u->user_begin(), u->user_end());
        }
      }
    }

    DEBUG(
      errs() << "Possible pointers to global memory in " << F.getName() << ":\n";
      for (auto &&p : all_global_ptrs) {
      errs() << "  " << *p << "\n";
      }
      errs() << "\n";
      );

    DEBUG(errs() << "Possible pointers to GV(" << G << ") in " << F.getName() << ":\n";);

    pending.insert(pending.end(), G.user_begin(), G.user_end());
    while (!pending.empty()) {
      const User *u = pending.back();
      pending.pop_back();

      if (const Instruction *inst = dyn_cast<Instruction>(u)) {
        if (&F != inst->getParent()->getParent())
          continue;
      }

      DEBUG(errs() << "user (" << u << "): " << *u << "\n";);
      if (const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(u)) {
        pending.insert(pending.end(), GEP->user_begin(), GEP->user_end());
      }
      else if (const ConstantExpr *cexp = dyn_cast<ConstantExpr>(u)) {
        if (cexp->getOpcode() == Instruction::GetElementPtr)
          pending.insert(pending.end(), cexp->user_begin(), cexp->user_end());
      }
      else if (const StoreInst *S = dyn_cast<StoreInst>(u)) {
        if (all_global_ptrs.find(S->getPointerOperand()) != all_global_ptrs.end()) {
          DEBUG(errs() << "target found, returns true\n";);
          return true;
        }
      }
    }

    DEBUG(errs() << "returns false\n";);
    return false;
  }

  // Assign addr_space to global variables.
  //
  // Assign address space to global variables, and make one copy for each user function.
  // Processed global variable could be in global, local or constant address space.
  //
  // tile_static are declared as static variables in section("clamp_opencl_local")
  // for each tile_static, make a modified clone with address space 3 and update users
  void promoteGlobalVars(Function *Func, InstUpdateWorkList * updateNeeded)
  {
    Module *M = Func->getParent();
    Module::GlobalListType &globals = M->getGlobalList();
    for (Module::global_iterator I = globals.begin(), E = globals.end();
         I != E; I++) {
      unsigned the_space = LocalAddressSpace;
      if (!I->hasSection() && I->isConstant() &&
          I->getType()->getPointerAddressSpace() == 0 &&
          I->hasName() && I->getLinkage() == GlobalVariable::InternalLinkage) {
        // Though I'm global, I'm constant indeed.
        if(usedInTheFunc(I, Func))
          the_space = ConstantAddressSpace;
        else
          continue;
      } else if (!I->hasSection() && I->isConstant() &&
                 I->getType()->getPointerAddressSpace() == 0 &&
                 I->hasName() && I->getLinkage() == GlobalVariable::PrivateLinkage) {
        // Though I'm private, I'm constant indeed.
        // FIXME: We should determine constant with address space (2) for OpenCL SPIR
        //              during clang front-end. It is not reliable to determine that in Promte stage
        if(usedInTheFunc(I, Func))
          the_space = ConstantAddressSpace;
        else
          continue;

      } else if (!I->hasSection() ||
                 I->getSection() != std::string(TILE_STATIC_NAME) ||
                 !I->hasName()) {
        // promote to global address space if the variable is used in a kernel
        // and does not come with predefined address space
        if (usedInTheFunc(I, Func) && I->getType()->getPointerAddressSpace() == 0) {
          the_space = GlobalAddressSpace;
        } else {
          continue;
        }
      }

      // If the address of this global variable is available from host, it
      // must stay in global address space.
      if (isAddressCopiedToHost(*I, *Func))
        the_space = GlobalAddressSpace;
      DEBUG(llvm::errs() << "Promoting variable: " << *I << "\n";
            errs() << "  to addrspace(" << the_space << ")\n";);

      std::set<Function *> users;
      typedef std::multimap<Function *, llvm::User *> Uses;
      Uses uses;

      // First visit all users which are of type ConstantExpr and flatten
      // them into Instruction
      for (Value::user_iterator U = I->user_begin(), Ue = I->user_end();
           U != Ue;) {
        if (ConstantExpr *C = dyn_cast<ConstantExpr>(*U)) {
          // Replace ConstantExpt with Instruction so we can track it
          updateListWithUsers(*U, I, I, updateNeeded);
          if (C->getNumUses() == 0) {
            // Only non-zero ref can be destroyed, otherwise deadlock occurs
            C->destroyConstant();
            U = I->user_begin();
            continue;
          }
        }
        ++U;
      }

      // Now all ConstantExpr should be flattened to Instruction
      // Revisit users of I
      for (Value::user_iterator U = I->user_begin(), Ue = I->user_end();
           U!=Ue;) {
        if (Instruction *Ins = dyn_cast<Instruction>(*U)) {
          users.insert(Ins->getParent()->getParent());
          uses.insert(std::make_pair(Ins->getParent()->getParent(), *U));
        }
        DEBUG(llvm::errs() << "U: \n";
              U->dump(););
        ++U;
      }

      int i = users.size()-1;
      // Create a clone of the tile static variable for each unique
      // function that uses it
      for (std::set<Function*>::reverse_iterator
           F = users.rbegin(), Fe = users.rend();
           F != Fe; F++, i--) {

        // tile static variables cannot have an initializer
        llvm::Constant *Init = nullptr;
        if (I->hasSection() && (I->getSection() == std::string(TILE_STATIC_NAME))) {
          Init = llvm::UndefValue::get(I->getType()->getElementType());
        } else {
          Init = I->hasInitializer() ? I->getInitializer() : 0;
        }

        GlobalVariable *new_GV =
          new GlobalVariable(*M,
                             I->getType()->getElementType(),
                             I->isConstant(), I->getLinkage(),
                             Init,
                             "",
                             (GlobalVariable *)0,
                             I->getThreadLocalMode(),
                             the_space);
        new_GV->copyAttributesFrom(I);
        if (i == 0) {
          new_GV->takeName(I);
        } else {
          new_GV->setName(I->getName());
        }
        if (new_GV->getName().find('.') == 0) {
          // HSAIL does not accept dot at the front of identifier
          // (clang generates dot names for string literals)
          std::string tmp = new_GV->getName();
          tmp[0] = 'x';
          new_GV->setName(tmp);
        }
        std::pair<Uses::iterator, Uses::iterator> usesOfSameFunction;
        usesOfSameFunction = uses.equal_range(*F);
        for ( Uses::iterator U = usesOfSameFunction.first, Ue =
              usesOfSameFunction.second; U != Ue; U++)
          updateListWithUsers (U->second, I, new_GV, updateNeeded);
      }
    }
  }

  void eraseOldTileStaticDefs(Module *M)
  {
    std::vector<GlobalValue*> todo;
    Module::GlobalListType &globals = M->getGlobalList();
    for (Module::global_iterator I = globals.begin(), E = globals.end();
         I != E; I++) {
      if (!I->hasSection() ||
          I->getSection() != std::string(TILE_STATIC_NAME) ||
          I->getType()->getPointerAddressSpace() != 0) {
        continue;
      }
      I->removeDeadConstantUsers();
      if (I->getNumUses() == 0)
        todo.push_back(I);
    }
    for (std::vector<GlobalValue*>::iterator I = todo.begin(),
         E = todo.end(); I!=E; I++) {
      (*I)->eraseFromParent();
    }
  }

  void promoteAllocas (Function * Func,
                       InstUpdateWorkList * updatesNeeded)
  {
    typedef BasicBlock::iterator iterator;
    for (iterator I = Func->begin()->begin();
         isa<AllocaInst>(I); ++I) {
      AllocaInst * AI = cast<AllocaInst>(I);
      Type * allocatedType = AI->getType()->getElementType();
      Type * promotedType = mapTypeToGlobal(allocatedType);

      if ( allocatedType == promotedType ) continue;

      AllocaInst * clonedAlloca = new AllocaInst(promotedType,
                                                 AI->getArraySize(),
                                                 "", AI);

      updateListWithUsers ( AI->user_begin(), AI->user_end(),
                            AI, clonedAlloca, updatesNeeded );
    }
  }

  void promoteBitcasts (Function * F, InstUpdateWorkList * updates)
  {
    typedef std::vector<BitCastInst *> BitCastList;
    BitCastList foundBitCasts;
    for (Function::iterator B = F->begin(), Be = F->end();
         B != Be; ++B) {
      for (BasicBlock::iterator I = B->begin(), Ie = B->end();
           I != Ie; ++I) {
        BitCastInst * BI = dyn_cast<BitCastInst>(I);
        if ( ! BI ) continue;
        foundBitCasts.push_back(BI);
      }
    }

    for (BitCastList::const_iterator I = foundBitCasts.begin(),
         Ie = foundBitCasts.end(); I != Ie; ++I) {
      BitCastInst * BI = *I;

      Type *destType = BI->getType();
      PointerType * destPtrType =
        dyn_cast<PointerType>(destType);
      if ( ! destPtrType ) continue;

      Type * srcType = BI->getOperand(0)->getType();
      PointerType * srcPtrType =
        dyn_cast<PointerType>(srcType);
      if ( ! srcPtrType ) continue;
#if 0
      unsigned srcAddressSpace =
        srcPtrType->getAddressSpace();

      unsigned destAddressSpace =
        destPtrType->getAddressSpace();
#endif
      Type * elementType = destPtrType->getElementType();
      Type * mappedType = mapTypeToGlobal(elementType);
      unsigned addrSpace = srcPtrType->getAddressSpace();
      Type * newDestType = PointerType::get(mappedType, addrSpace);
      if (elementType == mappedType) continue;

      BitCastInst * newBI = new BitCastInst(BI->getOperand(0),
                                            newDestType, BI->getName(),
                                            BI);
      updateListWithUsers (BI->user_begin(), BI->user_end(),
                           BI, newBI, updates);
    }

  }

  bool hasPtrToNonZeroAddrSpace (Value * V)
  {
    Type * ValueType = V->getType();
    PointerType * ptrType = dyn_cast<PointerType>(ValueType);
    if ( !ptrType ) return false;
    return true;
    return ptrType->getAddressSpace() != 0;

  }

  void updateArgUsers (Function * F, InstUpdateWorkList * updateNeeded)
  {
    typedef Function::arg_iterator arg_iterator;
    for (arg_iterator A = F->arg_begin(), Ae = F->arg_end();
         A != Ae; ++A) {
      if ( !hasPtrToNonZeroAddrSpace (A) ) continue;
      updateListWithUsers (A->user_begin(), A->user_end(),
                           A, A, updateNeeded);
    }
  }


  // updateOperandType - Replace types of operand and return values with the promoted types if necessary
  // This function goes through the function's body and handles GEPs and select instruction specially.
  // After a function is cloned by calling CloneFunctionInto, some of the operands types
  // might not be updated correctly. Neither are some of  the instructions' return types.
  // For example,
  // (1) getelementptr instruction will leave type of its pointer operand un-promoted
  // (2) select instruction will not update its return type as what has been changed to its #1 or #2 operand
  // Note that It is always safe to call this function right after CloneFunctionInto
  //
  void updateOperandType(Function * oldF, Function * newF, FunctionType* ty,
                         InstUpdateWorkList* workList)
  {
    // Walk all the BBs
    for (Function::iterator B = newF->begin(), Be = newF->end();B != Be; ++B) {
      // Walk all instructions
      for (BasicBlock::iterator I = B->begin(), Ie = B->end(); I != Ie; ++I) {
        if (SelectInst *Sel = dyn_cast<SelectInst>(I)) {
          assert(Sel->getOperand(1) && "#1  operand  of Select Instruction is invalid!");
          if (Sel->getType() != I->getOperand(1)->getType()) {
            // mutate type only when absolutely necessary
            Sel->mutateType(I->getOperand(1)->getType());
            updateListWithUsers(I->user_begin(), I->user_end(), I, I, workList);
          }
        } else if( GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
          // Only handle GEPs with parameters promoted (ex: after a select instruction)
          if (GEP->getPointerAddressSpace() != 0) {
            Type*T = GEP->getPointerOperandType();
            // Traverse the old args to find source type
            unsigned argIdx = 0;
            for (Function::arg_iterator A = oldF->arg_begin(), Ae = oldF->arg_end();
                 A != Ae; ++A, ++argIdx) {
              // Since Type* is immutable, pointer comparison to see if they are the same
              if(T == oldF->getFunctionType()->getParamType(argIdx)) {
                Argument* V = new Argument(ty->getParamType(argIdx), GEP->getPointerOperand()->getName());
                // Note that only forward udpate is allowed.
                updateGEPWithNewOperand(GEP, GEP->getPointerOperand(), V, workList);
              }
            }
          }
        }
      }
    }
  }

  Function * createPromotedFunctionToType ( Function * F,
                                            FunctionType * promoteType)
  {
    DEBUG(llvm::errs() << "========================================\n";);
    Function * newFunction = Function::Create (promoteType,
                                               F->getLinkage(),
                                               F->getName(),
                                               F->getParent());
    DEBUG(llvm::errs() << "New function name: " << newFunction->getName() << "\n" << "\n";);


    // let new function get all attributes from the old function
    newFunction->setAttributes(F->getAttributes());
    DEBUG(llvm::errs() << "Old function attributes: "; F->getAttributes().dump();
          llvm::errs() << "New function attributes: "; newFunction->getAttributes().dump(););

    // rewrite function with pointer type parameters
    if (F->getName().find("opencl_") != StringRef::npos ||
        F->getName().find("atomic_") != StringRef::npos) {

      DEBUG(llvm::errs() << "Old function name: " << F->getName() << "\n";
            llvm::errs() << "Old function type: "; F->getFunctionType()->dump(); llvm::errs() << "\n";
            llvm::errs() << "Old function has definition: " << !F->isDeclaration() << "\n";
            llvm::errs() << "New function type: "; newFunction->getFunctionType()->dump(); llvm::errs() << "\n";);

      unsigned Addrspace = PrivateAddressSpace;
      for (Function::const_arg_iterator it = newFunction->arg_begin(), ie = newFunction->arg_end(); it != ie; ++it) {
        Type *Ty = (*it).getType();
        if (isa<PointerType>(Ty)) {
          Addrspace = Ty->getPointerAddressSpace();
          DEBUG(llvm::errs() << "Pointer address space: " << Addrspace << "\n";);
          break;
        }
      }

      std::string newFuncName;
      Function *promotedFunction;
      switch (Addrspace) {
      case PrivateAddressSpace:
      case ConstantAddressSpace:
        break;
      case LocalAddressSpace:
        newFuncName = F->getName().str() + "_local";
        DEBUG(llvm::errs() << newFuncName << "\n";);
        promotedFunction = F->getParent()->getFunction(newFuncName);
        if (!promotedFunction) {
          newFunction->setName(F->getName() + "_local");
        } else {
          newFunction = promotedFunction;
        }
        break;
      case GlobalAddressSpace:
        newFuncName = F->getName().str() + "_global";
        DEBUG(llvm::errs() << newFuncName << "\n";);
        promotedFunction = F->getParent()->getFunction(newFuncName);
        if (!promotedFunction) {
          newFunction->setName(F->getName() + "_global");
        } else {
          newFunction = promotedFunction;
        }
        break;
      default:
        break;
      }

      DEBUG(llvm::errs() << "New function name: " << newFunction->getName() << "\n";);
    }


    ValueToValueMapTy CloneMapping;
    nameAndMapArgs(newFunction, F, CloneMapping);


    SmallVector<ReturnInst *, 1> Returns;
    if (!F->isDeclaration()) {
      // only clone the function if it's defined
      CloneFunctionInto(newFunction, F, CloneMapping, false, Returns);
    }

    ValueToValueMapTy CorrectedMapping;
    InstUpdateWorkList workList;
    //        promoteAllocas(newFunction, workList);
    //        promoteBitcasts(newFunction, workList);
    promoteGlobalVars(newFunction, &workList);
    updateArgUsers (newFunction, &workList);
    updateOperandType(F, newFunction, promoteType, &workList);

    do {
      /*while( !workList.empty() ) {
        update_token update = workList.back();
        workList.pop_back();
        updateInstructionWithNewOperand (update.subject,
        update.oldOperand,
        update.newOperand,
        workList);

        }*/
      workList.run();
      CollectChangedCalledFunctions ( newFunction, &workList );
    } while ( !workList.empty() );

    eraseOldTileStaticDefs(F->getParent());

    // don't verify the new function if it is only a declaration
    if (!newFunction->isDeclaration() && verifyFunction (*newFunction/*, PrintMessageAction*/)) {
      llvm::errs() << "When checking the updated function of: ";
      F->dump();
      llvm::errs() << " into: ";
      newFunction->dump();
    }
    DEBUG(llvm::errs() << "-------------------------------------------";);
    return newFunction;
  }

  Function * createPromotedFunction ( Function * F )
  {
    FunctionType * promotedType =
      createNewFunctionTypeWithPtrToGlobals (F);
    return createPromotedFunctionToType (F, promotedType);
  }

  /* A visitor function which is called for each MDNode
     located into another MDNode */
  class KernelNodeVisitor {
  public:
    KernelNodeVisitor(FunctionVect& FV);
    void operator()(MDNode * N);

  private:
    FunctionVect& found_kernels;
  };

  KernelNodeVisitor::KernelNodeVisitor(FunctionVect& FV)
    : found_kernels(FV)
  {}

  void KernelNodeVisitor::operator()(MDNode *N)
  {
    if ( N->getNumOperands() < 1) return;
#if LLVM_VERSION_MAJOR == 3
#if (LLVM_VERSION_MINOR >= 3) && (LLVM_VERSION_MINOR <= 5)
    // logic which is compatible from LLVM 3.3 till LLVM 3.5
    Value * Op = N->getOperand(0);
    if (!Op)
      return;
    if ( Function * F = dyn_cast<Function>(Op)) {
      found_kernels.push_back(F);
    }
#elif LLVM_VERSION_MINOR > 5
    // support new metadata data structure introduced in LLVM 3.6+
    const MDOperand& Op = N->getOperand(0);
    if ( Function * F = mdconst::dyn_extract<Function>(Op)) {
      found_kernels.push_back(F);
    }
#else
#error Unsupported LLVM MINOR VERSION
#endif
#else
#error Unsupported LLVM MAJOR VERSION
#endif

  }

  /* Call functor for each MDNode located within the Named MDNode */
  void visitMDNodeOperands(NamedMDNode * N, KernelNodeVisitor& visitor)
  {
    for (unsigned operand = 0, end = N->getNumOperands();
         operand < end; ++operand) {
      visitor(N->getOperand(operand));
    }
  }

  /* Accumulate LLVM functions that are kernels within the
     found_kernels vector. Return true if kernels are found.
     False otherwise. */
  bool findKernels(Module& M, FunctionVect& found_kernels)
  {
    NamedMDNode * root = getKernelListMDNode(M);
    if (!root || (root->getNumOperands() == 0)) return false;

    KernelNodeVisitor visitor(found_kernels);
    visitMDNodeOperands(root, visitor);

    return found_kernels.size() != 0;
  }

  void updateKernels(Module& M, const FunctionMap& new_kernels)
  {
    NamedMDNode * root = getKernelListMDNode(M);
    typedef FunctionMap::const_iterator iterator;
    // for each kernel..
    for (unsigned i = 0; i < root->getNumOperands(); i++) {
      // for each metadata of the kernel..
      MDNode * kernel = root->getOperand(i);
#if LLVM_VERSION_MAJOR == 3
#if (LLVM_VERSION_MINOR >= 3) && (LLVM_VERSION_MINOR <= 5)
      // logic which is compatible from LLVM 3.3 till LLVM 3.5
      Function * f = dyn_cast<Function>(kernel->getOperand(0));
#elif LLVM_VERSION_MINOR > 5
      // support new metadata data structure introduced in LLVM 3.6+
      Function * f = mdconst::dyn_extract<Function>(kernel->getOperand(0));
#else
#error Unsupported LLVM MINOR VERSION
#endif
#else
#error Unsupported LLVM MAJOR VERSION
#endif
      assert(f != NULL);
      iterator I = new_kernels.find(f);
      if (I != new_kernels.end()) {
#if LLVM_VERSION_MAJOR == 3
#if (LLVM_VERSION_MINOR >= 3) && (LLVM_VERSION_MINOR <= 5)
        // logic which is compatible from LLVM 3.3 till LLVM 3.5
        kernel->replaceOperandWith(0, I->second);
#elif LLVM_VERSION_MINOR > 5
        // support new metadata data structure introduced in LLVM 3.6+
        kernel->replaceOperandWith(0, ValueAsMetadata::get(I->second));
#else
#error Unsupported LLVM MINOR VERSION
#endif
#else
#error Unsupported LLVM MAJOR VERSION
#endif

      }
    }
    for (iterator kern = new_kernels.begin(), end = new_kernels.end();
         kern != end; ++kern) {
      // Remove the original function
      kern->first->deleteBody();
      kern->first->setCallingConv(llvm::CallingConv::C);
    }
  }

  StructType * wrapStructType (StructType * src, Type * Subst) {
    std::vector<Type *> ElementTypes;
    bool changed = false;
    typedef StructType::element_iterator iterator;

    for (iterator E = src->element_begin(), Ee = src->element_end();
         E != Ee; ++E) {
      Type * newType = *E;
      PointerType * ptrType = dyn_cast<PointerType>(newType);
      if (ptrType) {
        newType = Subst;
        changed = true;
      }
      ElementTypes.push_back(newType);
    }

    if (!changed) return src;

    return StructType::get (src->getContext(),
                            ArrayRef<Type *>(ElementTypes),
                            src->isPacked());
  }

  Function * createWrappedFunction (Function * F)
  {
    typedef Function::arg_iterator arg_iterator;
    typedef std::vector<Type *> WrappedTypes;
    typedef std::pair<unsigned, StructType *> WrapPair;
    typedef std::vector<WrapPair> WrappingTodo;
    WrappedTypes wrappedTypes;
    WrappingTodo Todo;
    bool changed = false;
    unsigned argNum  = 0;
    Module * M = F->getParent ();
    DataLayout TD(M);
    unsigned ptrSize = TD.getPointerSizeInBits ();
    Type * PtrDiff = Type::getIntNTy (F->getContext(), ptrSize);
    for (arg_iterator A = F->arg_begin(), Ae = F->arg_end();
         A != Ae; ++A, ++argNum) {
      Type * argType = A->getType ();
      if (!A->hasByValAttr()) {
        wrappedTypes.push_back (argType);
        continue;
      }
      // ByVal args are pointers
      PointerType * ptrArgType = cast<PointerType>(argType);
      StructType * argStructType =
        dyn_cast<StructType>(ptrArgType->getElementType());
      if (!argStructType) {
        wrappedTypes.push_back (argType);
        continue;
      }
      StructType * wrapped =
        wrapStructType (argStructType, PtrDiff);
      if (wrapped == argStructType) {
        wrappedTypes.push_back (argType);
        continue;
      }
      PointerType * final =
        PointerType::get(wrapped,
                         ptrArgType->getAddressSpace());
      wrappedTypes.push_back(final);
      Todo.push_back (std::make_pair(argNum, argStructType));
      changed = true;
    }
    if ( !changed ) return F;

    FunctionType * newFuncType =
      FunctionType::get(F->getReturnType(),
                        ArrayRef<Type*>(wrappedTypes),
                        F->isVarArg());
    Function * wrapped = Function::Create (newFuncType,
                                           F->getLinkage(),
                                           F->getName(),
                                           M);
    std::vector<Value *> callArgs;
    for (arg_iterator sA = F->arg_begin(), dA = wrapped->arg_begin(),
         Ae = F->arg_end(); sA != Ae; ++sA, ++dA) {
      dA->setName (sA->getName());
      callArgs.push_back(dA);
    }
    wrapped->setAttributes (F->getAttributes());
    BasicBlock * entry = BasicBlock::Create(F->getContext(), "entry",
                                            wrapped, NULL);

    Type * BoolTy = Type::getInt1Ty(F->getContext());
    Type * Int8Ty = Type::getInt8Ty(F->getContext());
    Type * Int32Ty = Type::getInt32Ty(F->getContext());
    Type * Int64Ty = Type::getInt64Ty(F->getContext());
    Type * castSrcType = PointerType::get(Int8Ty, 0);
    Type * castTargetType = PointerType::get(Int8Ty, 0);

    std::vector<Type *> MemCpyTypes;
    MemCpyTypes.push_back (castTargetType);
    MemCpyTypes.push_back (castSrcType);
    MemCpyTypes.push_back (Int64Ty);
    Function * memCpy = Intrinsic::getDeclaration (M, Intrinsic::memcpy,
                                                   ArrayRef<Type *>(MemCpyTypes));
    Constant * align = ConstantInt::get (Int32Ty, 4, false);
    Constant * isVolatile = ConstantInt::getFalse (BoolTy);
    for (WrappingTodo::iterator W = Todo.begin(), We = Todo.end();
         W != We; ++W) {
      Function::arg_iterator A = wrapped->arg_begin();
      for (unsigned i = 0; i < W->first; ++i, ++A) {}
      AllocaInst * AI = new AllocaInst(W->second, NULL, "", entry);
      std::vector<Value *> memCpyArgs;
      memCpyArgs.push_back (new BitCastInst (AI, castTargetType,
                                             "", entry));
      memCpyArgs.push_back (new BitCastInst (A, castSrcType,
                                             "", entry));

      memCpyArgs.push_back (ConstantInt::get(Int64Ty,
                                             TD.getTypeStoreSize(W->second),
                                             false));
      memCpyArgs.push_back (align);
      memCpyArgs.push_back (isVolatile);
      CallInst::Create (memCpy, ArrayRef<Value *>(memCpyArgs),
                        "", entry);
      callArgs [W->first] = AI;
    }

    CallInst::Create (F, ArrayRef <Value *> (callArgs), "", entry);
    ReturnInst::Create (F->getContext(), NULL, entry);
    return wrapped;
  }


  class PromoteGlobals : public ModulePass {
  public:
    static char ID;
    PromoteGlobals();
    virtual ~PromoteGlobals();
    virtual void getAnalysisUsage(AnalysisUsage& AU) const;
    bool runOnModule(Module& M);
  };
} // ::<unnamed> namespace

PromoteGlobals::PromoteGlobals() : ModulePass(ID)
{}

PromoteGlobals::~PromoteGlobals()
{}

void PromoteGlobals::getAnalysisUsage(AnalysisUsage& AU) const
{
  AU.addRequired<CallGraphWrapperPass>();
}
static std::string escapeName(const std::string &orig_name)
{
  std::string oldName(orig_name);
  // AMD OpenCL doesn't like kernel names starting with _
  if (oldName[0] == '_')
    oldName = oldName.substr(1);
  size_t loc;
  // escape name: $ -> _EC_
  while ((loc = oldName.find('$')) != std::string::npos) {
    oldName.replace(loc, 1, "_EC_");
  }
  return oldName;
}

bool PromoteGlobals::runOnModule(Module& M)
{
  FunctionVect foundKernels;
  FunctionMap promotedKernels;
  if (!findKernels(M, foundKernels)) return false;

  typedef FunctionVect::const_iterator kernel_iterator;
  for (kernel_iterator F = foundKernels.begin(), Fe = foundKernels.end();
       F != Fe; ++F) {
    if ((*F)->empty())
      continue;
    Function * promoted = createPromotedFunction (*F);
    promoted->takeName (*F);
    promoted->setName(escapeName(promoted->getName().str()));

    promoted->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
    // lambdas can be set as internal. This causes problem
    // in optimizer and we shall mark it as non-internal
    if (promoted->getLinkage() ==
        GlobalValue::InternalLinkage) {
      promoted->setLinkage(GlobalValue::ExternalLinkage);
    }
    (*F)->setLinkage(GlobalValue::InternalLinkage);
    promotedKernels[*F] = promoted;
  }
  updateKernels (M, promotedKernels);

  /// FIXME: The following code can be removed. It is too late to add
  ///        NoDuplicate attribute on barrier in SPIRify pass. We already
  //         add NoDuplicate attribute in clang
#if 0
  // If the barrier present is used, we need to ensure it cannot be duplicated.
  for (Module::iterator F = M.begin(), Fe = M.end(); F != Fe; ++F) {
    StringRef name = F->getName();
    if (name.equals ("barrier")) {
      F->addFnAttr (Attribute::NoDuplicate);
    }
  }
#endif

  // Rename local variables per SPIR naming rule
  Module::GlobalListType &globals = M.getGlobalList();
  for (Module::global_iterator I = globals.begin(), E = globals.end();
       I != E; I++) {
    if (I->hasSection() &&
        I->getSection() == std::string(TILE_STATIC_NAME) &&
        I->getType()->getPointerAddressSpace() != 0) {

      std::string oldName = escapeName(I->getName().str());
      // Prepend the name of the function which contains the user
      std::set<std::string> userNames;
      for (Value::user_iterator U = I->user_begin(), Ue = I->user_end();
           U != Ue; U ++) {
        Instruction *Ins = dyn_cast<Instruction>(*U);
        if (!Ins)
          continue;
        userNames.insert(Ins->getParent()->getParent()->getName().str());
      }
      // A local memory variable belongs to only one kernel, per SPIR spec
      assert(userNames.size() < 2 &&
             "__local variable belongs to more than one kernel");
      if (userNames.empty())
        continue;
      oldName = *(userNames.begin()) + "."+oldName;
      I->setName(oldName);
      // AMD SPIR stack takes only internal linkage
      if (I->hasInitializer())
        I->setLinkage(GlobalValue::InternalLinkage);
    }
  }
  return false;
}


char PromoteGlobals::ID = 0;
#if 1
static RegisterPass<PromoteGlobals>
Y("promote-globals", "Promote Pointer To Global Pass");
#else
INITIALIZE_PASS(PromoteGlobals, "promote-globals", "Promote Pointer to Global", false, false);
#endif // BoltTranslator_EXPORTS

llvm::ModulePass * createPromoteGlobalsPass ()
{
  return new PromoteGlobals;
}
