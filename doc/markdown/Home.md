# HC API : Moving Beyond C++AMP for Accelerated GPU Computing

HC is a C++ API for accelerated computing provided by the HCC compiler.  It has some similarities to C++ AMP and therefore, reference materials (blogs, articles, books) that describe C++ AMP also proivide an excellent way to become familiar with HC.  For example, both APIs use a parallel_for_each construct to specify a parallel execution region that runs on accelerator.  However, HC has several important differences from C++ AMP, including the removal of the "restrict" keyword to annotate device code, an explicit asynchronous launch behavior for parallel_for_each, the support for non-constant tile size, the support for memory pointer, etc..

---

# HC API

HC comes with two header files as of now:

- <hc.hpp> : Main header file for HC
- <hc_math.hpp> : Math functions for HC

Most HC APIs are stored under "hc" namespace, and the class name is the same as their counterpart in C++AMP "Concurrency" namespace.  Users of C++AMP should find it easy to switch from C++AMP to HC.

| C++AMP | HC |
|----|--------|
| Concurrency::accelerator | hc::accelerator |
| Concurrency::accelerator_view | hc::accelerator_view |
| Concurrency::extent | hc::extent |
| Concurrency::index | hc::index |
| Concurrency::completion_future | hc::completion_future |
| Concurrency::array | hc::array |
| Concurrency::array_view | hc::array_view |

---

# How to build programs with HC API

Use "hcc-config", instead of "clamp-config", or you could manually add "-hc" when you invoke clang++. Also, "hcc" is added as an alias for "clang++".

For example:

```
hcc `hcc-config --cxxflags --ldflags` foo.cpp -o foo
```

---

# HCC built-in macros

Built-in macros:

| Macro | Meaning |
|----|--------|
| ```__HCC__``` | always be 1 |
| ```__hcc_major__``` | major version number of HCC |
| ```__hcc_minor__``` | minor version number of HCC |
| ```__hcc_patchlevel__``` | patchlevel of HCC |
| ```__hcc_version__``` | combined string of ```__hcc_major__```, ```__hcc_minor__```, ```__hcc_patchlevel__``` |

The rule for ```__hcc_patchlevel__``` is: yyWW-(HCC driver git commit #)-(HCC clang git commit #)
- yy stands for the last 2 digits of the year
- WW stands for the week number of the year

Macros for language modes in use:

| Macro | Meaning |
|----|--------|
| ```__HCC_AMP__``` | 1 in case in C++ AMP mode (-std=c++amp) |
| ```__HCC_HC__``` | 1 in case in HC mode (-hc) |

Compilation mode:
HCC is a single-source compiler where kernel codes and host codes can reside in the same file. Internally HCC would trigger 2 compilation iterations, and the following macros can be user by user programs to determine which mode the compiler is in.

| Macro | Meaning |
|----|--------|
| ```__HCC_ACCELERATOR__``` | not 0 in case the compiler runs in kernel code compilation mode |
| ```__HCC_CPU__``` | not 0 in case the compiler runs in host code compilation mode |

---

# HC-specific features

- relaxed rules in operations allowed in kernels
- new syntax of tiled_extent and tiled_index
- dynamic group segment memory allocation
- true asynchronous kernel launching behavior
- additional HSA-specific APIs


# Differences between HC API and C++ AMP

Despite HC and C++ AMP share a lot of similarities in terms of programming constructs (e.g. parallel_for_each, array, array_view, etc.), there are several significant differences between the two APIs.

## Support for explicit asynchronous ```parallel_for_each```

In C++ AMP, the  ```parallel_for_each``` appears as a synchronous function call in a program (i.e. the host waits for the kernel to complete); howevever, the compiler may optimize it to execute the kernel asynchronously and the host would synchronize with the device on the first access of the data modified by the kernel.  For example, if a ```parallel_for_each``` writes the an array_view, then the first access to this array_view on the host after the ```parallel_for_each``` would block until the ```parallel_for_each``` completes. 

HC supports the automatic synchronization behavior as in C++ AMP.  In addition, HC's ```parallel_for_each``` supports explicit asynchronous execution.  It returns a ```completion_future``` (similar to C++ std::future) object that other asynchronous operations could synchronize with, which provides better flexibility on task graph construction and enables more precise control on optimization.        

## Annotation of device functions

C++ AMP uses the ```restrict(amp)``` keyword to annotatate functions that runs on the device.

```
void foo() restrict(amp) {
..
}
...
parallel_for_each(...,[=] () restrict(amp) {
 foo();
});

```

HC uses a function attribute (```[[hc]]``` or ``` __attribute__((hc))``` ) to annotate a device function. 

```
void foo()  [[hc]] {
..
}
...
parallel_for_each(...,[=] () [[hc]] {
 foo();
});
```

The \[\[hc\]\] annotation for the kernel function called by ```parallel_for_each``` is optional as it is automatically annotated as a device function by the hcc compiler.  The compiler also supports partial automatic \[\[hc\]\] annotation for functions that are called by other device functions within the same source file:

```
// Since bar is called by foo, which is a device function, the hcc compiler
// will automatically annotate bar as a device function
void bar() {
...
}

void foo() [[hc]] {
  bar();
}
```

## Dynamic tile size

C++ AMP doesn't support dynamic tile size.  The size of each tile dimensions has to be a compile-time constant specified as template arguments to the tile_extent object:

```
extent<2> ex(x, y);

// create a tile extent of 8x8 from the extent object
// note that the tile dimensions have to be constant values
tiled_extent<8,8> t_ex(ex);

parallel_for_each(t_ex, [=](tiled_index<8,8> t_id) restrict(amp) {
...
});
```
HC supports both static and dynamic tile size:
```
extent<2> ex(x,y)

// create a tile extent from dynamically calculated values
// note that the the tiled_extent template takes the rank instead of dimensions
tx = test_x ? tx_a : tx_b;
ty = test_y ? ty_a : ty_b;
tiled_extent<2> t_ex(ex, tx, ty);

parallel_for_each(t_ex, [=](tiled_index<2> t_id) [[hc]] {
...
});

```

## Support for memory pointer

C++ AMP doens't support lambda capture of memory pointer into a GPU kernel.

HC supports capturing memory pointer by a GPU kernel.

```
// allocate GPU memory through the HSA API
int* gpu_pointer;
hsa_memory_allocate(..., &gpu_pointer);
...
parallel_for_each(ext, [=](index i) [[hc]] {
  gpu_pointer[i[0]]++;
}

```
For HSA APUs that supports system wide shared virtual memory, a GPU kernel can directly access system memory allocated by the host:
```
int* cpu_memory = (int*) malloc(...);
...
parallel_for_each(ext, [=](index i) [[hc]] {
  cpu_memory[i[0]]++;
});
```



