extern uint64_t load_hsaco(hc::accelerator_view *av, const char * fileName, const char *kernelName);
void dispatch_glp_kernel(const grid_launch_parm *lp, uint64_t kernelCodeHandle, void *args, int argSize);
