#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>


#define DB_API        0  /* 0x0001  HCC runtime API calls */
#define DB_CMD        1  /* 0x0002  Kernel and Copy Commands and Barriers */
#define DB_WAIT       2  /* 0x0004  Synchronization and waiting for commands to finish. */
#define DB_AQL        3  /* 0x0008  Decode and display AQL packets  */
#define DB_QUEUE      4  /* 0x0010  Queue creation and desruction commands, and queue contents after each command push. */
#define DB_SIG        5  /* 0x0020  Signal creation, allocation, pool */
#define DB_LOCK       6  /* 0x0040  Locks and HCC thread-safety code */
#define DB_KERNARG    7  /* 0x0080  Show first 128 bytes of kernarg blocks passed to kernels */
#define DB_COPY       8  /* 0x0100  Copy debug */
#define DB_COPY2      9  /* 0x0200  Detailed copy debug */
#define DB_RESOURCE  10  /* 0x0400  Resource (signal/kernarg/queue) allocation and growth, and other unusual potentially performance-impacting events. */
#define DB_INIT      11  /* 0x0800  HCC initialization and shutdown. */
#define DB_MISC      12  /* 0x1000  misc debug, not yet classified. */
#define DB_AQL2      13  /* 0x2000  Show raw bytes of AQL packet */
#define DB_CODE      14  /* 0x4000  Show CreateKernel and code creation debug */
#define DB_CMD2      15  /* 0x8000  More detailed command info, including barrier commands created by hcc rt. */
// If adding new define here update the table below:

extern unsigned HCC_DB;

#define DBPARM(x) #x << "=" << x


// Keep close to debug defs above since these have to be kept in-sync
static std::vector<std::string> g_DbStr = {"api", "cmd", "wait", "aql", "queue", "sig", "lock", "kernarg", "copy", "copy2", "resource", "init", "misc", "aql2", "code", "cmd2"};


// Macro for prettier debug messages, use like:
// DBOUT(" Something happened" << myId() << " i= " << i << "\n");
#define COMPILE_HCC_DB 1

#define DBFLAG(db_flag) (COMPILE_HCC_DB && (HCC_DB & (1<<(db_flag))))

#define DBSTREAM  std::cerr
#define DBWSTREAM std::wcerr

// Use str::stream so output is atomic wrt other threads:
#define DBOUT(db_flag, msg) \
if (DBFLAG(db_flag)) { \
    std::stringstream sstream;\
    sstream << "   hcc-" << g_DbStr[db_flag] << " tid:" << hcc_tlsShortTid._shortTid << " " << msg ; \
    DBSTREAM << sstream.str();\
};

// Like DBOUT, but add newline:
#define DBOUTL(db_flag, msg) DBOUT(db_flag, msg << "\n")

// get a the current filename without the path
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/')+1 : __FILE__)

// Class with a constructor that gets called when new thread is created:
struct ShortTid {
    ShortTid() ;
    int _shortTid;
};

extern thread_local ShortTid hcc_tlsShortTid;

namespace hc {

  static void print_backtrace() {
  }

} // namespace hc
