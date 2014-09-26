


#include <stdlib.h>
#include <libelf.h>
#include <iostream>
#include <string.h>
#include <fstream>

#include "elf_utils.hpp"


enum {
  SECTION_HSA_DATA = 0,
  SECTION_HSA_CODE,
  SECTION_HSA_OPERAND,
};

struct SectionDesc {
  int sectionId;
  const char *brigName;
  const char *bifName;
} sectionDescs[] = {
  { SECTION_HSA_DATA, "hsa_data",".brig_hsa_data" },
  { SECTION_HSA_CODE, "hsa_code",".brig_hsa_code" },
  { SECTION_HSA_OPERAND,"hsa_operand",".brig_hsa_operand"},
};
enum status_t {
    STATUS_SUCCESS=1,
    STATUS_KERNEL_INVALID_SECTION_HEADER,
    STATUS_KERNEL_ELF_INITIALIZATION_FAILED,
    STATUS_KERNEL_INVALID_ELF_CONTAINER,
    STATUS_KERNEL_MISSING_DATA_SECTION,
    STATUS_KERNEL_MISSING_CODE_SECTION,
    STATUS_KERNEL_MISSING_OPERAND_SECTION,
    STATUS_UNKNOWN,
 };

 // status messages in-sync with status code
 const char* status_msg[] = {
    "unknown status",
    "success",
    "invalid section header",
    "ELF initialization failed",
    "invalid ELF container",
    "missing data section",
    "missing code section",
    "missing operand section",
    "unknown status"
 };

static void printError(status_t status) {

#if 0 
  if (status==STATUS_SUCCESS) {
    return;
  }
     
  std::cout<<"Error - Could not create BRIG module : "<<status_msg[status]<<"("<< -status<<")"<<std::endl;
  int elf_err = elf_errno();
  if (elf_err)
    std::cout<<"ELF error message: "<<elf_errmsg(elf_errno())<<std::endl;

  switch (status) {
    case STATUS_KERNEL_INVALID_SECTION_HEADER:
    case STATUS_KERNEL_ELF_INITIALIZATION_FAILED:
    case STATUS_KERNEL_INVALID_ELF_CONTAINER:
      std::cout<<"Possible corruption in ELF file "<<std::endl;
      break;
    case STATUS_KERNEL_MISSING_DATA_SECTION:
    case STATUS_KERNEL_MISSING_CODE_SECTION:
    case STATUS_KERNEL_MISSING_OPERAND_SECTION:
      std::cout << "One or more ELF sections are missing. Use readelf command to \
                   to check if hsa_data, hsa_code and hsa_operands exist" <<std::endl;
      break;
    default:
      break;
  }
  return;

#endif
}


static const SectionDesc* getSectionDesc(int sectionId) {
  const int NUM_PREDEFINED_SECTIONS = sizeof(sectionDescs)/sizeof(sectionDescs[0]);
  for(int i=0; i<NUM_PREDEFINED_SECTIONS; ++i) {
    if (sectionDescs[i].sectionId == sectionId) {
      return &sectionDescs[i];
    }
  }
  return NULL;
}

static Elf_Scn* extractElfSection ( Elf *elfP,
  				    Elf_Data *secHdr,
				    const SectionDesc* desc ) {
  int cnt = 0;
  Elf_Scn* scn = NULL;
  Elf32_Shdr* shdr = NULL;
  char* sectionName = NULL;
  /* Walk thru elf sections */
  for (cnt = 1, scn = NULL; scn = elf_nextscn(elfP, scn); cnt++) {
      if (((shdr = elf32_getshdr(scn)) == NULL))
            return NULL;
      sectionName = (char *)secHdr->d_buf + shdr->sh_name;
      if ( sectionName &&
           ((strcmp(sectionName, desc->brigName) == 0) ||
               (strcmp(sectionName, desc->bifName) == 0))) {
		  return scn;
      }
   }
   return NULL;
}


/* Extract section and copy into HsaBrig */
static status_t ExtractSectionAndCopy ( Elf *elfP,
					 Elf_Data *secHdr, 
					 const SectionDesc* desc,
                     hsa_ext_brig_module_t* brig_module,
                     hsa_ext_brig_section_id_t section_id,
					 bool verify
				 	) {
  Elf_Scn* scn = NULL;
  Elf_Data* data = NULL;
  void* address_to_copy;
  size_t section_size=0;
  scn = extractElfSection ( elfP, secHdr, desc );
  if (scn) {
    if ( (data = elf_getdata(scn, NULL)) == NULL) {
       return STATUS_UNKNOWN;
    }
    section_size = data->d_size;
    if (section_size > 0 ) {
      address_to_copy = malloc(section_size);
      memcpy(address_to_copy, data->d_buf, section_size);
    }
  }
  if (verify && ( !scn ||  section_size == 0 ) )  {
     return STATUS_UNKNOWN;
  }

  //Create a section header
  brig_module->section[section_id] = 
        (hsa_ext_brig_section_header_t*) address_to_copy; 
  return STATUS_SUCCESS;
} 


/* Reads binary of BRIG and BIF format */
static status_t readBinary(hsa_ext_brig_module_t **brig_module_t, Elf* elfP) {

  //Create the brig_module
  uint32_t number_of_sections = 3;
  hsa_ext_brig_module_t* brig_module;
  brig_module = (hsa_ext_brig_module_t*)
                (malloc (sizeof(hsa_ext_brig_module_t) + sizeof(void*)*number_of_sections));
  brig_module->section_count = number_of_sections;

  status_t status;
  Elf32_Ehdr* ehdr = NULL;
  Elf_Data *secHdr = NULL;
  Elf_Scn* scn = NULL;

  if ( elf_kind (elfP) != ELF_K_ELF ) {
    return STATUS_KERNEL_INVALID_ELF_CONTAINER;
  }
  
  /* Obtain the .shstrtab data buffer */
  if (((ehdr = elf32_getehdr(elfP)) == NULL) ||
       ((scn = elf_getscn(elfP, ehdr->e_shstrndx)) == NULL) ||
         ((secHdr = elf_getdata(scn, NULL)) == NULL)) {
            return STATUS_KERNEL_INVALID_SECTION_HEADER;
  }

  status = ExtractSectionAndCopy ( elfP, 
				    secHdr,
                    getSectionDesc(SECTION_HSA_DATA), 
                    brig_module,
                    HSA_EXT_BRIG_SECTION_DATA,
		     	    true);
  if (status != STATUS_SUCCESS) {
    return STATUS_KERNEL_MISSING_DATA_SECTION;
  }
  status = ExtractSectionAndCopy ( elfP, 
				   secHdr,
                   getSectionDesc(SECTION_HSA_CODE), 
                   brig_module,
                   HSA_EXT_BRIG_SECTION_CODE,
				   true
			  	 );
  if (status != STATUS_SUCCESS) {
    return STATUS_KERNEL_MISSING_CODE_SECTION;
  }
  status = ExtractSectionAndCopy ( elfP, 
				   secHdr,
                   getSectionDesc(SECTION_HSA_OPERAND), 
                   brig_module,
                   HSA_EXT_BRIG_SECTION_OPERAND,
				   true
			  	 );
  if (status != STATUS_SUCCESS) {
    return STATUS_KERNEL_MISSING_OPERAND_SECTION;
  }
  elf_end(elfP);
  *brig_module_t = brig_module;
  return STATUS_SUCCESS;
}


bool CreateBrigModuleFromBrigFile(const char* file_name, hsa_ext_brig_module_t** brig_module) {

 FILE *fp = ::fopen(file_name, "rb");
 int fd = fileno(fp);
 Elf* elfP = NULL;

 status_t status;
 if ( elf_version ( EV_CURRENT ) == EV_NONE ) {
     status = STATUS_KERNEL_ELF_INITIALIZATION_FAILED;
 }
 else {
     if (( elfP = elf_begin ( fd, ELF_C_READ, (Elf *)0 )) == NULL ) {
         status = STATUS_KERNEL_INVALID_ELF_CONTAINER;
     }
     else {
         status = readBinary(brig_module, elfP);
     }
 }

 if (status != STATUS_SUCCESS) {
     printError(status);
 }
 return (status==STATUS_SUCCESS);
}


bool CreateBrigModuleFromBrigMemory(char* buffer, const size_t buffer_size, hsa_ext_brig_module_t** brig_module) {

 Elf* elfP = NULL;

 status_t status;

 if ( elf_version ( EV_CURRENT ) == EV_NONE ) {
     status = STATUS_KERNEL_ELF_INITIALIZATION_FAILED;
 }
 else {
     if (( elfP = elf_memory (buffer, buffer_size)) == NULL ) {
         status = STATUS_KERNEL_INVALID_ELF_CONTAINER;
     }
     else {
         status = readBinary(brig_module, elfP);
     }
 }

 if (status != STATUS_SUCCESS) {
   printError(status);
 }
 return (status==STATUS_SUCCESS);
}


bool DestroyBrigModule(hsa_ext_brig_module_t* brig_module) {
    for (int i=0; i<brig_module->section_count; i++) {
        free (brig_module->section[i]);
    }
    free (brig_module);
}

