/*
 * llvmBuiltins.cpp
 *
 *  Created on: 31 Jan 2012
 *      Author: v1smoll
 */


#include <axtor/util/llvmBuiltins.h>

#include <llvm/Constants.h>
#include <llvm/Instructions.h>
#include <llvm/DerivedTypes.h>


namespace axtor {

	llvm::Function * create_memset(llvm::Module & M, std::string funcName, uint space)
	{
	llvm::Module * mod = &M;
	// Type Definitions
	std::vector<llvm::Type*>FuncTy_0_args;
	llvm::PointerType* PointerTy_1 = llvm::PointerType::get(llvm::IntegerType::get(mod->getContext(), 8), space);
	FuncTy_0_args.push_back(PointerTy_1);

	FuncTy_0_args.push_back(llvm::IntegerType::get(mod->getContext(), 8));
	FuncTy_0_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
	FuncTy_0_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
	FuncTy_0_args.push_back(llvm::IntegerType::get(mod->getContext(), 1));
	llvm::FunctionType* FuncTy_0 = llvm::FunctionType::get(
	/*Result=*/llvm::Type::getVoidTy(mod->getContext()),
	/*Params=*/FuncTy_0_args,
	/*isVarArg=*/false);

	// Constant Definitions
	llvm::ConstantInt* const_int32_5 = llvm::ConstantInt::get(mod->getContext(), llvm::APInt(32, llvm::StringRef("0"), 10));
	llvm::ConstantInt* const_int32_6 = llvm::ConstantInt::get(mod->getContext(), llvm::APInt(32, llvm::StringRef("1"), 10));

	//function declaration
	llvm::Function* func = mod->getFunction(funcName);
	if (!func) {
		func = llvm::Function::Create(
		/*Type=*/FuncTy_0,
		/*Linkage=*/ llvm::GlobalValue::ExternalLinkage,
		/*Name=*/funcName, mod);
		func->setCallingConv(llvm::CallingConv::PTX_Kernel);
	}
	llvm::AttrListPtr func_PAL;
	{
		llvm::SmallVector<llvm::AttributeWithIndex, 4> Attrs;
		llvm::AttributeWithIndex PAWI;
		PAWI.Index = 1U; PAWI.Attrs = llvm::Attribute::NoCapture;
		Attrs.push_back(PAWI);
		PAWI.Index = 4294967295U; PAWI.Attrs = llvm::Attribute::NoUnwind;
		Attrs.push_back(PAWI);
		func_PAL = llvm::AttrListPtr::get(Attrs);

	}
	func->setAttributes(func_PAL);



	// function body
	llvm::Function::arg_iterator args = func->arg_begin();
	llvm::Value* ptr_ptr = args++;
	ptr_ptr->setName("ptr");
	llvm::Value* int8_p = args++;
	int8_p->setName("p");
	llvm::Value* int32_len_39 = args++;
	int32_len_39->setName("len");
	llvm::Value* int32_align_40 = args++;
	int32_align_40->setName("align");
	llvm:: Value* int1_isVolatile_41 = args++;
	int1_isVolatile_41->setName("isVolatile");

	llvm::BasicBlock* label_entry_42 = llvm::BasicBlock::Create(mod->getContext(), "entry",func,0);
	llvm::BasicBlock* label_for_body_43 = llvm::BasicBlock::Create(mod->getContext(), "for.body",func,0);
	llvm::BasicBlock* label_for_end_44 = llvm::BasicBlock::Create(mod->getContext(), "for.end",func,0);

	// Block entry (label_entry_42)
	llvm::ICmpInst* int1_cmp1 = new llvm::ICmpInst(*label_entry_42, llvm::ICmpInst::ICMP_SGT, int32_len_39, const_int32_5, "cmp1");
	llvm::BranchInst::Create(label_for_body_43, label_for_end_44, int1_cmp1, label_entry_42);

	// Block for.body (label_for_body_43)
	llvm::Argument* fwdref_46 = new llvm::Argument(llvm::IntegerType::get(mod->getContext(), 32));
	llvm::PHINode* int32_i_02 = llvm::PHINode::Create(llvm::IntegerType::get(mod->getContext(), 32), 2, "i.02", label_for_body_43);
	int32_i_02->addIncoming(fwdref_46, label_for_body_43);
	int32_i_02->addIncoming(const_int32_5, label_entry_42);

	llvm::GetElementPtrInst* ptr_arrayidx = llvm::GetElementPtrInst::Create(ptr_ptr, int32_i_02, "arrayidx", label_for_body_43);
	llvm::StoreInst* void_47 = new llvm::StoreInst(int8_p, ptr_arrayidx, false, label_for_body_43);
	void_47->setAlignment(1);
	llvm::BinaryOperator* int32_inc = llvm::BinaryOperator::Create(llvm::Instruction::Add, int32_i_02, const_int32_6, "inc", label_for_body_43);
	llvm::ICmpInst* int1_exitcond_48 = new llvm::ICmpInst(*label_for_body_43, llvm::ICmpInst::ICMP_EQ, int32_inc, int32_len_39, "exitcond");
	llvm::BranchInst::Create(label_for_end_44, label_for_body_43, int1_exitcond_48, label_for_body_43);

	// Block for.end (label_for_end_44)
	llvm:: ReturnInst::Create(mod->getContext(), label_for_end_44);

	// Resolve Forward References
	fwdref_46->replaceAllUsesWith(int32_inc); delete fwdref_46;

	return func;
	}

	llvm::Function * create_memcpy(llvm::Module & M, std::string funcName, uint destSpace, uint srcSpace)
	{
		llvm::Module * mod = &M;
		// Type Definitions
		std::vector<llvm::Type*>FuncTy_0_args;
		llvm::PointerType* PointerTy_1 = llvm::PointerType::get(llvm::IntegerType::get(mod->getContext(), 8), destSpace);
		FuncTy_0_args.push_back(PointerTy_1);

		llvm::PointerType* PointerTy_2 = llvm::PointerType::get(llvm::IntegerType::get(mod->getContext(), 8), srcSpace);
		FuncTy_0_args.push_back(PointerTy_2);

		FuncTy_0_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
		FuncTy_0_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
		FuncTy_0_args.push_back(llvm::IntegerType::get(mod->getContext(), 1));
		llvm::FunctionType* FuncTy_0 = llvm::FunctionType::get(
		/*Result=*/llvm::Type::getVoidTy(mod->getContext()),
		/*Params=*/FuncTy_0_args,
		/*isVarArg=*/false);

		// Constant Definitions
		llvm::ConstantInt* const_int32_4 = llvm::ConstantInt::get(mod->getContext(), llvm::APInt(32, llvm::StringRef("0"), 10));
		llvm::ConstantInt* const_int32_5 = llvm::ConstantInt::get(mod->getContext(), llvm::APInt(32, llvm::StringRef("1"), 10));


		//function declaration
		llvm::Function* func = mod->getFunction(funcName);
		if (!func) {
			func = llvm::Function::Create(
			/*Type=*/FuncTy_0,
			/*Linkage=*/ llvm::GlobalValue::ExternalLinkage,
			/*Name=*/funcName, mod);
			func->setCallingConv(llvm::CallingConv::PTX_Kernel);
		}
		llvm::AttrListPtr func_PAL;
		{
			llvm::SmallVector<llvm::AttributeWithIndex, 4> Attrs;
			llvm::AttributeWithIndex PAWI;
			PAWI.Index = 1U; PAWI.Attrs = llvm::Attribute::NoCapture;
			Attrs.push_back(PAWI);
			PAWI.Index = 2U; PAWI.Attrs =llvm::Attribute::NoCapture;
			Attrs.push_back(PAWI);
			PAWI.Index = 4294967295U; PAWI.Attrs = llvm::Attribute::NoUnwind;
			Attrs.push_back(PAWI);
			func_PAL = llvm::AttrListPtr::get(Attrs);

		}
		func->setAttributes(func_PAL);

		// function body
		llvm::Function::arg_iterator args = func->arg_begin();
		llvm::Value* ptr_dest = args++;
		ptr_dest->setName("dest");
		llvm::Value* ptr_src = args++;
		ptr_src->setName("src");
		llvm::Value* int32_len = args++;
		int32_len->setName("len");
		llvm::Value* int32_align = args++;
		int32_align->setName("align");
		llvm::Value* int1_isVolatile = args++;
		int1_isVolatile->setName("isVolatile");

		llvm::BasicBlock* label_entry = llvm::BasicBlock::Create(mod->getContext(), "entry",func,0);
		llvm::BasicBlock* label_for_body_lr_ph = llvm::BasicBlock::Create(mod->getContext(), "for.body.lr.ph",func,0);
		llvm::BasicBlock* label_for_body = llvm::BasicBlock::Create(mod->getContext(), "for.body",func,0);
		llvm::BasicBlock* label_for_end = llvm::BasicBlock::Create(mod->getContext(), "for.end",func,0);

		// Block entry (label_entry)
		llvm::ICmpInst* int1_cmp2 = new llvm::ICmpInst(*label_entry, llvm::ICmpInst::ICMP_SGT, int32_len, const_int32_4, "cmp2");
		llvm::BranchInst::Create(label_for_body_lr_ph, label_for_end, int1_cmp2, label_entry);

		// Block for.body.lr.ph (label_for_body_lr_ph)
		llvm::GetElementPtrInst* ptr_lftr_limit = llvm::GetElementPtrInst::Create(ptr_dest, int32_len, "lftr.limit", label_for_body_lr_ph);
		llvm::BranchInst::Create(label_for_body, label_for_body_lr_ph);

		// Block for.body (label_for_body)
		llvm::Argument* fwdref_8 = new llvm::Argument(PointerTy_1);
		llvm::PHINode* ptr_dest_addr_05 = llvm::PHINode::Create(PointerTy_1, 2, "dest.addr.05", label_for_body);
		ptr_dest_addr_05->addIncoming(ptr_dest, label_for_body_lr_ph);
		ptr_dest_addr_05->addIncoming(fwdref_8, label_for_body);

		llvm::Argument* fwdref_9 = new llvm::Argument(PointerTy_2);
		llvm::PHINode* ptr_src_addr_04 = llvm::PHINode::Create(PointerTy_2, 2, "src.addr.04", label_for_body);
		ptr_src_addr_04->addIncoming(ptr_src, label_for_body_lr_ph);
		ptr_src_addr_04->addIncoming(fwdref_9, label_for_body);

		llvm::GetElementPtrInst* ptr_incdec_ptr = llvm::GetElementPtrInst::Create(ptr_src_addr_04, const_int32_5, "incdec.ptr", label_for_body);
		llvm::LoadInst* int8_10 = new llvm::LoadInst(ptr_src_addr_04, "", false, label_for_body);
		int8_10->setAlignment(1);
		llvm::GetElementPtrInst* ptr_incdec_ptr1 = llvm::GetElementPtrInst::Create(ptr_dest_addr_05, const_int32_5, "incdec.ptr1", label_for_body);
		llvm::StoreInst* void_11 = new llvm::StoreInst(int8_10, ptr_dest_addr_05, false, label_for_body);
		void_11->setAlignment(1);
		llvm::ICmpInst* int1_exitcond = new llvm::ICmpInst(*label_for_body, llvm::ICmpInst::ICMP_EQ, ptr_incdec_ptr1, ptr_lftr_limit, "exitcond");
		llvm::BranchInst::Create(label_for_end, label_for_body, int1_exitcond, label_for_body);

		// Block for.end (label_for_end)
		llvm::ReturnInst::Create(mod->getContext(), label_for_end);

		// Resolve Forward References
		fwdref_9->replaceAllUsesWith(ptr_incdec_ptr); delete fwdref_9;
		fwdref_8->replaceAllUsesWith(ptr_incdec_ptr1); delete fwdref_8;
		return func;
	}

}
