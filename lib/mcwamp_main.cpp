//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <iostream>
#include <cassert>
#include "clamp-config.hxx"
/* Flag set by ‘--verbose’. */
static int verbose_flag;
static bool build_mode = false, install_mode = true; // use install mode by default
static bool gpu_path = false, cpu_path = false;

void replace(std::string& str,
        const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    while(start_pos != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos = str.find(from);
    }
}

void cxxflags(void) {
    if (!build_mode && !install_mode) {
        std::cerr << "Please specify --install or --build mode before flags\n";
        abort();
    }
    // Common options
    std::cout << "-std=c++amp";

#if defined(CXXAMP_ENABLE_HSA)
    std::cout << " -DCXXAMP_ENABLE_HSA=1";
    std::cout << " -I" CMAKE_HSA_ROOT;
#endif

#if !defined(CXXAMP_ENABLE_HSA)
    // OpenCL headers
    std::cout << " -I" CMAKE_OPENCL_INC;
#endif

    // clamp
    if (build_mode) {
        std::cout << " -I" CMAKE_CLAMP_INC_DIR;
        // libcxx
        std::cout << " -I" CMAKE_LIBCXX_INC;
#if !defined(CXXAMP_ENABLE_HSA)
        // GMAC options, build tree
        std::cout << " -I" CMAKE_GMAC_INC_BIN_DIR;
        std::cout << " -I" CMAKE_GMAC_INC_DIR;
#endif
    } else if (install_mode) {
        std::cout << " -I" CMAKE_INSTALL_INC;
        std::cout << " -I" CMAKE_INSTALL_LIBCXX_INC;
    } else {
        assert(0 && "Unreacheable!");
    }

    if (gpu_path) {
#if !defined(CXXAMP_ENABLE_HSA)
        std::cout << " -D__GPU__=1 -Xclang -famp-is-device -fno-builtin -fno-common -m32 -O2";
#else
        std::cout << " -D__GPU__=1 -Xclang -famp-is-device -fno-builtin -fno-common -m64 -O2";
#endif
    } else if (cpu_path) {
#if !defined(CXXAMP_ENABLE_HSA)
        std::cout << " -D__CPU__=1";
#else
        std::cout << " -D__CPU__=1";
#endif
    }

    std::cout << std::endl;
}

void ldflags(void) {
    // Common options
    std::cout << "-std=c++amp";
    if (build_mode) {
        std::cout << " -L" CMAKE_GMAC_LIB_DIR;
#ifdef __APPLE__
        std::cout << " -Wl,-rpath," CMAKE_GMAC_LIB_DIR;
#else
        std::cout << " -L" CMAKE_LIBCXX_LIB_DIR;
        std::cout << " -L" CMAKE_LIBCXXRT_LIB_DIR;
        std::cout << " -Wl,--rpath="
            CMAKE_GMAC_LIB_DIR ":"
            CMAKE_LIBCXX_LIB_DIR ":"
            CMAKE_LIBCXXRT_LIB_DIR ;
#endif
    } else if (install_mode) {
        std::cout << " -L" CMAKE_INSTALL_LIB;
#ifdef __APPLE__
        std::cout << " -Wl,-rpath," CMAKE_INSTALL_LIB;
#else
        std::cout << " -Wl,--rpath=" CMAKE_INSTALL_LIB;
#endif
    }
#ifndef __APPLE__
#if defined(CXXAMP_ENABLE_HSA)
    std::cout << " -Wl,--rpath=" CMAKE_HSA_LIB;
    std::cout << " -L" CMAKE_HSA_LIB;
    std::cout << " -Wl,--whole-archive -lhsacontext -Wl,--no-whole-archive ";
    std::cout << " -lelf -lhsa-runtime64 ";
    std::cout << " " CMAKE_HSA_LIB "/libhsail.a ";
    std::cout << " -Wl,--unresolved-symbols=ignore-in-shared-libs ";
#else
    std::cout << " -lgmac-hpe ";
#endif

    std::cout << " -lc++ -lcxxrt -ldl -lpthread ";
    std::cout << "-Wl,--whole-archive -lmcwamp -Wl,--no-whole-archive ";

#else // __APPLE__
    std::cout << " -lgmac-hpe -lc++ -lmcwamp ";
#endif
}

void prefix(void) {
    std::cout << CMAKE_INSTALL_PREFIX;
}

int main (int argc, char **argv) {
    int c;
    while (1)
    {
        static struct option long_options[] =
        {
            /* These options set a flag. */
            {"verbose", no_argument,       &verbose_flag, 1},
            {"brief",   no_argument,       &verbose_flag, 0},
            /* These options don't set a flag.
               We distinguish them by their indices. */
            {"gpu",      no_argument,       0, 'g'},
            {"cpu",      no_argument,       0, 'c'},
            {"cxxflags", no_argument,       0, 'a'},
            {"build",    no_argument,       0, 'b'},
            {"install",  no_argument,       0, 'i'},
            {"ldflags",  no_argument,       0, 'l'},
            {"prefix",  no_argument,       0, 'p'},
            {0, 0, 0, 0}
        };
        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long (argc, argv, "",
                long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
            break;

        switch (c)
        {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0)
                    break;
                printf ("option %s", long_options[option_index].name);
                if (optarg)
                    printf (" with arg %s", optarg);
                printf ("\n");
                break;

            case 'a':   // --cxxflags
                cxxflags();
                break;
            case 'l':   // --ldflags
                ldflags();
                break;
            case 'p':   // --prefix
                prefix();
                break;
            case 'b':   // --build
                build_mode = true;
                install_mode = false;
                break;
            case 'i':   // --install
                build_mode = false;
                install_mode = true;
                break;
            case 'g':   // --gpu
                gpu_path = true;
                cpu_path = false;
                break;
            case 'c':   // --cpu
                gpu_path = false;
                cpu_path = true;
                break;
            case '?':
                /* getopt_long already printed an error message. */
                break;

            default:
                abort ();
        }
    }

    /* Instead of reporting ‘--verbose’
       and ‘--brief’ as they are encountered,
       we report the final status resulting from them. */
    if (verbose_flag)
        puts ("verbose flag is set");

    /* Print any remaining command line arguments (not options). */
    if (optind < argc)
    {
        printf ("non-option ARGV-elements: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        putchar ('\n');
    }

    exit (0);
}
