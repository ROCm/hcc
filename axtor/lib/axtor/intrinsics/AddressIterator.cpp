/*
 * AddressIterator.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/intrinsics/AddressIterator.h>
#include <axtor/util/llvmShortCuts.h>
#include <axtor/util/ResourceGuard.h>

namespace axtor {

	AddressIterator::AddressIterator() :
		val(0),
		next(0),
		cumulativeValue(false)
	{}

	AddressIterator::AddressIterator(AddressIterator * _next, llvm::Value * _val, bool _cumulativeValue) :
		val(_val),
		next(_next),
		cumulativeValue(_cumulativeValue)
	{}

	bool AddressIterator::isCumulative() const
	{
		return cumulativeValue;
	}

	void AddressIterator::dump(int idx)
	{
		std::cerr << "ADDRESS(" << idx << ")\n";
		std::cerr << "IndexVal="; val->dump();
		std::cerr << "cumulative=" << cumulativeValue << "\n";

		if (next) {
			next->dump();
		}
	}

	void AddressIterator::dump()
	{
		dump(1);
	}

	llvm::Value * AddressIterator::getValue()
	{
		return val;
	}

	AddressIterator * AddressIterator::getNext()
	{
		return next;
	}

	bool AddressIterator::isEnd() const
	{
		return next == 0;
	}

	bool AddressIterator::isEmpty() const
	{
		return val == 0;
	}

	/*
	 * returns a pair consisting of the dereferenced root object and a corresponding index value list
	 */

	AddressIterator::AddressResult::AddressResult(llvm::Value * _rootValue, AddressIterator * _iterator) :
		rootValue(_rootValue),
		iterator(_iterator)
	{}


	/*
	 * creates an address of either an extract-value instruction or a streak of GEPs (pointer mode)
	 *
	 * (obvious issue with sequences of dereferncing instructions)
	 */
	AddressIterator::AddressResult AddressIterator::createAddress(llvm::Value * derefValue, StringSet & derefFuncs)
	{
#ifdef DEBUG
		std::cerr << "!! AdressIterator::createAddress()\n";
#endif
		AddressIterator * next = 0;
		llvm::Value * rootValue = derefValue;

		// TODO: add support for constant extract/insert-Value
		// value extraction
		if (llvm::isa<llvm::ExtractValueInst>(rootValue)) {
			llvm::ExtractValueInst * inst = llvm::cast<llvm::ExtractValueInst>(rootValue);

#ifdef DEBUG
			std::cerr << "add from extract value\n. no.op==" << inst->getNumOperands() << "\n";
#endif
			for (llvm::ExtractValueInst::idx_iterator it = inst->idx_end() - 1; it != inst->idx_begin() - 1; it--)
			{
				next = new AddressIterator(next, get_uint(*it), false);
			}
			rootValue = inst->getOperand(0);
			return AddressIterator::AddressResult(rootValue, next);

		//value insertion
		} else if (llvm::isa<llvm::InsertValueInst>(rootValue)) {
			llvm::InsertValueInst * inst = llvm::cast<llvm::InsertValueInst>(rootValue);

#ifdef DEBUG
			std::cerr << "add from insert value\n. no.op==" << inst->getNumOperands() << "\n";
#endif
			for (llvm::InsertValueInst::idx_iterator it = inst->idx_end() - 1; it != inst->idx_begin() - 1; it--)
			{
				next = new AddressIterator(next, get_uint(*it), false);
			}
			rootValue = inst->getOperand(0);
			return AddressIterator::AddressResult(rootValue, next);
		}

		//dereferencing a pointer value
		for (;;) {
			/*
			 * if this is a GEP of a GEP then we need to consider that the GEP before returns a pointer again so if we chain them up we need to eliminate
			 * the indexing into that pointer
			 */
			//used to eliminate superfluous zeros in chained GEPs

			/* constant GEP */
			if (llvm::isa<llvm::ConstantExpr>(rootValue)) {
				llvm::ConstantExpr * constExpr = llvm::cast<llvm::ConstantExpr>(rootValue);
				if (constExpr->getOpcode() == llvm::Instruction::GetElementPtr) {
					llvm::Value * dereffedValue = constExpr->getOperand(0);

					// if this GEP performs a non-trivial Offset-operation on the input Ptr keep it
					bool cumulativeOffset = false;
					bool tossFirstOffset = false;

					if (isGEP(dereffedValue)) {
						uint64_t offset;
						tossFirstOffset = (evaluateInt(constExpr->getOperand(1), offset) && offset == 0);
						cumulativeOffset = !tossFirstOffset;
					}

					uint stopIndex = tossFirstOffset ? 1 : 0;

					for (uint idx = constExpr->getNumOperands() - 1; idx > stopIndex; --idx)
					{
						llvm::Constant * constOperand = constExpr->getOperand(idx);
						next = new AddressIterator(next, constOperand, cumulativeOffset);
						cumulativeOffset = false;
					}
					rootValue = dereffedValue;
					continue;
				}


			/* GEP instruction */
			} else if (llvm::isa<llvm::GetElementPtrInst>(rootValue)) {
				llvm::GetElementPtrInst * gep = llvm::cast<llvm::GetElementPtrInst>(rootValue);
				llvm::Value * dereffedValue = gep->getOperand(0);
				llvm::Value * firstOffsetValue = gep->getOperand(1);

				// if this GEP performs a non-trivial Offset-operation on the input Ptr keep it
				bool cumulativeOffset = false;
				bool tossFirstOffset = false;

				if (isGEP(dereffedValue)) {
					uint64_t offset;
					tossFirstOffset = (evaluateInt(firstOffsetValue, offset) && offset == 0);
					cumulativeOffset = !tossFirstOffset;
				}

#ifdef DEBUG
				std::cerr << "&&&&& tossFirst=" << tossFirstOffset << " ; cumulative=" << cumulativeOffset << "\n";
				firstOffsetValue->dump();
#endif

				uint stopIndex = tossFirstOffset ? 1 : 0;
				for(uint idx = gep->getNumOperands() - 1; idx > stopIndex; --idx)
				{
#ifdef DEBUG
					std::cerr << "!!! pushing " << idx << " of ";
					gep->dump();
#endif
					llvm::Value * operandVal = gep->getOperand(idx);
					next = new AddressIterator(next, operandVal, cumulativeOffset);
					cumulativeOffset = false;
				}
				rootValue = dereffedValue;
				continue;

			/* Deref Call */
			//op[0] = "function Name"
			//op[1] = "deref value"
			//op[N...]=elements
			} else if (llvm::isa<llvm::CallInst>(rootValue)) {
				llvm::CallInst * call = llvm::cast<llvm::CallInst>(rootValue);
				std::string name = call->getCalledFunction()->getName().str();

				if (derefFuncs.find(name) != derefFuncs.end())
				{
					for(uint idx = call->getNumOperands() - 1; idx > 1; idx--)
					{
						next = new AddressIterator(next, call->getOperand(idx), false);
					}
					rootValue = call->getOperand(1);
				}
				continue;
			}

			break;
		}


#ifdef DEBUG
		std::cerr << "root value:\n";
		rootValue->dump();
#endif

		//add a dummy
		/* if (! next) {
			next = new AddressIterator(0, get_int(0));
		} */

		return AddressIterator::AddressResult(rootValue, next);
	}

	template class ResourceGuard<AddressIterator>;

}
