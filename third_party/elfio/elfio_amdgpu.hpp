//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ELFIO_AMDGPU_HPP
#define ELFIO_AMDGPU_HPP

namespace hc {

// AMDGPU e_flags mirroring from llvm/include/llvm/BinaryFormat/ELF.h

// AMDGPU specific e_flags.
enum : unsigned {
  // Processor selection mask for EF_AMDGPU_MACH_* values.
  EF_AMDGPU_MACH = 0x0ff,

  // Not specified processor.
  EF_AMDGPU_MACH_NONE = 0x000,

  // R600-based processors.

  // Radeon HD 2000/3000 Series (R600).
  EF_AMDGPU_MACH_R600_R600 = 0x001,
  EF_AMDGPU_MACH_R600_R630 = 0x002,
  EF_AMDGPU_MACH_R600_RS880 = 0x003,
  EF_AMDGPU_MACH_R600_RV670 = 0x004,
  // Radeon HD 4000 Series (R700).
  EF_AMDGPU_MACH_R600_RV710 = 0x005,
  EF_AMDGPU_MACH_R600_RV730 = 0x006,
  EF_AMDGPU_MACH_R600_RV770 = 0x007,
  // Radeon HD 5000 Series (Evergreen).
  EF_AMDGPU_MACH_R600_CEDAR = 0x008,
  EF_AMDGPU_MACH_R600_CYPRESS = 0x009,
  EF_AMDGPU_MACH_R600_JUNIPER = 0x00a,
  EF_AMDGPU_MACH_R600_REDWOOD = 0x00b,
  EF_AMDGPU_MACH_R600_SUMO = 0x00c,
  // Radeon HD 6000 Series (Northern Islands).
  EF_AMDGPU_MACH_R600_BARTS = 0x00d,
  EF_AMDGPU_MACH_R600_CAICOS = 0x00e,
  EF_AMDGPU_MACH_R600_CAYMAN = 0x00f,
  EF_AMDGPU_MACH_R600_TURKS = 0x010,

  // Reserved for R600-based processors.
  EF_AMDGPU_MACH_R600_RESERVED_FIRST = 0x011,
  EF_AMDGPU_MACH_R600_RESERVED_LAST = 0x01f,

  // First/last R600-based processors.
  EF_AMDGPU_MACH_R600_FIRST = EF_AMDGPU_MACH_R600_R600,
  EF_AMDGPU_MACH_R600_LAST = EF_AMDGPU_MACH_R600_TURKS,

  // AMDGCN-based processors.

  // AMDGCN GFX6.
  EF_AMDGPU_MACH_AMDGCN_GFX600 = 0x020,
  EF_AMDGPU_MACH_AMDGCN_GFX601 = 0x021,
  // AMDGCN GFX7.
  EF_AMDGPU_MACH_AMDGCN_GFX700 = 0x022,
  EF_AMDGPU_MACH_AMDGCN_GFX701 = 0x023,
  EF_AMDGPU_MACH_AMDGCN_GFX702 = 0x024,
  EF_AMDGPU_MACH_AMDGCN_GFX703 = 0x025,
  EF_AMDGPU_MACH_AMDGCN_GFX704 = 0x026,
  // AMDGCN GFX8.
  EF_AMDGPU_MACH_AMDGCN_GFX801 = 0x028,
  EF_AMDGPU_MACH_AMDGCN_GFX802 = 0x029,
  EF_AMDGPU_MACH_AMDGCN_GFX803 = 0x02a,
  EF_AMDGPU_MACH_AMDGCN_GFX810 = 0x02b,
  // AMDGCN GFX9.
  EF_AMDGPU_MACH_AMDGCN_GFX900 = 0x02c,
  EF_AMDGPU_MACH_AMDGCN_GFX902 = 0x02d,
  EF_AMDGPU_MACH_AMDGCN_GFX904 = 0x02e,
  EF_AMDGPU_MACH_AMDGCN_GFX906 = 0x02f,

  // Reserved for AMDGCN-based processors.
  EF_AMDGPU_MACH_AMDGCN_RESERVED0 = 0x027,
  EF_AMDGPU_MACH_AMDGCN_RESERVED1 = 0x030,

  // First/last AMDGCN-based processors.
  EF_AMDGPU_MACH_AMDGCN_FIRST = EF_AMDGPU_MACH_AMDGCN_GFX600,
  EF_AMDGPU_MACH_AMDGCN_LAST = EF_AMDGPU_MACH_AMDGCN_GFX906,

  // Indicates if the xnack target feature is enabled for all code contained in
  // the object.
  EF_AMDGPU_XNACK = 0x100,
};

} // namespace ELFIO

#endif // ELFIO_AMDGPU_HPP
