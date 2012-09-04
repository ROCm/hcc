/*
 * llvmTypeFactory.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/util/llvmTypeFactory.h>
#include "llvm/ADT/ArrayRef.h"

llvm::Type * axtor::generateType(char * data, char ** oData)
{
	llvm::LLVMContext & context = SharedContext::get();

	switch(*data)
	{
		case 'i': //INTEGER :i<SIZE>
		{
			int bits = parseInt(data + 1, oData);
			return llvm::IntegerType::get(context, bits);
		}

		case 'f': //FLOAT :f
		{
			*oData = data + 1;
			return llvm::Type::getFloatTy(context);
		}

		case 'b': //BOOL :b
		{
			*oData = data + 1;
			return llvm::Type::getInt1Ty(context);
		}

		case '<': //packed STRUCT :s
		case '{': //unpacked STRUCT
		{
			bool packed = *data == '<';

			if (data[1] == '}' || data[1] == '>')
			{
				return NULL; //no empty structs!

			} else {
				std::vector<llvm::Type*> params;
				char termChar = packed ? '>' : '}';

				//parse subtypes
				for(data++; *data != '\0'; ++data)
				{
					llvm::Type * subType = generateType(data, &data);
					params.push_back(subType);

					if (*data == termChar) {
						*oData = data + 1;
            llvm::ArrayRef<llvm::Type*> paramArray = 
              llvm::ArrayRef<llvm::Type*>(params);
						return llvm::StructType::get(context, paramArray, packed);

					} else if ( *data != ',') {
						return NULL;
					}
				}
			}
		}

		default:
			return NULL;
	};
}

llvm::Type * axtor::generateType(const char * in)
{
	char data[1024];
	char * out;
	memcpy(data, in, 1024);

	llvm::Type * type = generateType(data, &out);

	if (*out == '\0') {
		return type;
	} else {
		std::cerr << "error parsing typestring at " << *out << std::endl;
		return NULL;
	}
}
