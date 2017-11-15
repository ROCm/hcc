// Feature flags for HCC capabilities
//
// Intended for HCC users to be able to determine if the HCC version supports a desired feature
//
#pragma once

//
// If set, am_memtracker_update API accepts appPtr parm
#define __HCC_HAS_EXTENDED_AM_MEMTRACKER_UPDATE (1)

// Indicate the version of hc::printf being supported
#define __HCC_FEATURE_PRINTF (1)
