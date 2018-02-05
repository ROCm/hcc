#pragma once

#ifndef __GPU_ONLY
#define __GPU_ONLY restrict(amp)
#endif

#ifndef __CPU_ONLY
#define __CPU_ONLY restrict(cpu)
#endif

#ifndef __GPU
#define __GPU restrict(amp,cpu)
#endif

#ifndef __AUTO
#define __AUTO restrict(auto)
#endif
