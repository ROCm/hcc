struct grid_launch_parm;

struct Kernel {
  uint64_t _kernelCodeHandle;
  int      _groupSegmentSize;
  int      _privateSegmentSize;
};

extern Kernel load_hsaco(hc::accelerator_view *av, const char * fileName, const char *kernelName);
void dispatch_glp_kernel(const grid_launch_parm *lp, const Kernel &k, void *args, int argSize, bool systemScope);
