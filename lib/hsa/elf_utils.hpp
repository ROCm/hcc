#ifndef ELF_UTILS_H
#define ELF_UTILS_H

#include <cstdlib>
#include "hsa.h"
#include "hsa_ext_finalize.h"

#ifdef __cplusplus
extern "C" {
#endif

bool CreateBrigModuleFromBrigFile(const char* file_name, hsa_ext_brig_module_t** brig_module);
bool CreateBrigModuleFromBrigMemory(char* buffer, const size_t buffer_size, hsa_ext_brig_module_t** brig_module);
bool DestroyBrigModule(hsa_ext_brig_module_t* brig_module);


#ifdef __cplusplus
}
#endif


#endif
