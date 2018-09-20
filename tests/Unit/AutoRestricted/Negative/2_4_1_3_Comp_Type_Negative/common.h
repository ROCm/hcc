#pragma once

#ifndef __GPU_ONLY
#define __GPU_ONLY [[hc]]
#endif

#ifndef __CPU_ONLY
#define __CPU_ONLY [[cpu]]
#endif

#ifndef __GPU
#define __GPU [[cpu, hc]]
#endif

#ifndef __AUTO
#define __AUTO restrict(auto)
#endif
