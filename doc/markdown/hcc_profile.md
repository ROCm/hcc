# HCC Profile Mode

HCC supports low-overhead profiler to trace or summarize command timestamp information to stderr for any HCC  or HIP program.
Tho profiler messages are interleaved with the trace output from the application -  which is handy to identify the region-of-interest and 
can complement deeper analysis with the CodeXL GUI
Additionally, the hcc profiler requires only console mode access and can be used on machine where graphics are not available or are hard to access.

Some other useful features:
- Calculates the actual bandwidth for memory transfers
- Identifies PeerToPeer memory copies
- Shows start / stop timestamps for each command (if requested)
- Shows barrier commands and the time they spent waiting to resolve (if requested)

### Enable and configure

HCC_PROFILE=1 shows a summary of kernel and data commands when hcc exits. (under development)
HCC_PROFILE=2 enables a profile message after each command (kernel or data movement) completes. 

Additionally, the HCC_PROFILE_VERBOSE variable controls the information shown in the profile log.  This is a bit-vector:
0x2 : Show start and stop timestamps for each command.
0x4 : Show the device.queue.cmdseqnum for each command.
0x8 : Show the short CPU TID for each command. (not supported)
0x10 : Show logs for barrier commands. Time shown for barrier commands includes the time to wait for all input dependencies, plus the previous command to complete, plus any fence operations performed by the barrier.

## Sample Output

### Kernel Commands

This shows the simplest trace output for kernel commands with no additional verbosity flags:
```
$ HCC_PROFILE=2 ./my-hcc-app ...
profile:  kernel;                 ihipMemsetKernel<unsigned int>;        107.7 us;
profile:  kernel;                                         Im2Col;         19.0 us;
profile:  kernel;                               tg_betac_alphaab;         93.9 us;
profile:  kernel;                                  MIOpenConvUni;        230.1 us;
        ^command type;           ^kernel name                 ;        ^cmd duration
```

This example shows profiled kernel commands with full verbose output:
```
$ HCC_PROFILE=2 HCC_PROFILE_VERBOSE=0xf ./my-hcc-app ...
profile:  kernel;                 ihipMemsetKernel<unsigned int>;        836.5 us;      94858752897230; 94858753733702; #0.0.4;
profile: barrier;                        deps:0_acq:none_rel:sys;          3.4 us;      94859075640573; 94859075643933; #0.0.5;
profile:  kernel;                                         Im2Col;         17.8 us;      94859076277181; 94859076294941; #0.3.1;
profile:  kernel;                               tg_betac_alphaab;         32.6 us;      94859537593679; 94859537626319; #0.3.2;
profile:  kernel;                                  MIOpenConvUni;        125.4 us;      94860077852212; 94860077977651; #0.3.3;
^profile  ^type;                   ^kernel_name                 ;        ^duration;     ;start          ;end;           ^id
```

- ^profile:  always "profile:" to distinguish it from other output.
- ^type is the command type : kernel, copy, copyslo,or barrier.  The examples and descriptions in this section are all kernel commands.
- ^kernel_name shows the (short) kernel name.  
- ^duration shows the command duration measured in us.  This is measured using the GPU timestamps and represents the command execution on the acclerator device.
- ^start shows command start time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- ^stop shows command stop time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- ^id shows the command id in device.queue.cmd format.  (if HCC_PROFILE_VERBOSE & 0x4).  The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue. uniquely identifies the command during the process execution.

### Memory Copy Commands

```
profile: copyslo;                         HostToDevice_sync_slow;        909.2 us;      94858703102939; 94858704012090; #0.0.0; 2359296 bytes;  2.2 MB;   2.5 GB/s;
profile:    copy;                         DeviceToHost_sync_fast;        117.0 us;      94858726408586; 94858726525545; #0.0.0; 1228800 bytes;  1.2 MB;   10.0 GB/s;
profile:    copy;                         DeviceToHost_sync_fast;          9.0 us;      94858726668652; 94858726677612; #0.0.0; 400 bytes;      0.0 MB;   0.0 GB/s;
profile:    copy;                         HostToDevice_sync_fast;         15.2 us;      94858727639572; 94858727654772; #0.0.0; 9600 bytes;     0.0 MB;   0.6 GB/s;
profile:    copy;                         HostToDevice_async_fast;        131.5 us;     94858729198931; 94858729330450; #0.6.1; 1228800 bytes;  1.2 MB;   8.9 GB/s;
^profile  ^type;                          ^copy_name;                    ^duration;     ;start          ;end;           ^id     ^size_bytes     ^size_mb; ^xfer_bandwidth
```

- ^profile:  always "profile:" to distinguish it from other output.
- ^type is the command type : kernel, copy, copyslo,or barrier.  The examples and descriptions in this section are all copy or copyslo commands.
- ^copy_name has 3 parts: 
    - Copy kind: HostToDevice, HostToHost, DeviceToHost, DeviceToDevice, or PeerToPeer.  DeviceToDevice indicates the copy occurs on a single device while PeerToPeer indicates a copy between devices.
    - Sync or Async.  Synchronous copies indicate the host waits for the completion for the copy. Asynchronous copies are launched by the host without waiting for the copy to complete.
    - Fast or Slow.  Fast copies use the GPUs optimized copy routines from the hsa_amd_memory_copy routine.  Slow copies typically involve unpinned host memory and can't take the fast path.
    - For example `HostToDevice_async_fast`
- ^duration shows the command duration measured in us.  This is measured using the GPU timestamps and represents the command execution on the acclerator device.
- ^start shows command start time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- ^stop shows command stop time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- ^id shows the command id in device.queue.cmd format.  (if HCC_PROFILE_VERBOSE & 0x4).  The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue. uniquely identifies the command during the process execution.

### Barrier Commands
Barrier commands are only enabled if HCC_PROFILE_VERBOSE 0x

An example barrier command with full vebosity:
```
profile: barrier;                        deps:0_acq:none_rel:sys;          5.3 us;      94858731419410; 94858731424690; #0.0.2;
^profile  ^type;                          ^barrier_name;                    ^duration;     ;start          ;end;           ^id     
```
- ^profile:  always "profile:" to distinguish it from other output.
- ^type is the command type : kernel, copy, copyslo,or barrier.  The examples and descriptions in this section are all copy commands.
- ^barrier_name has 3 parts: 
    - deps:#  - the number of input dependencies the barrier packet is waiting for.
    - acq:    - the acquire fence for the barrier.  May be none, acc(accelerator or agent), sys(system).  See HSA AQL spec for additional information.
    - rel:    - the release fence for the barrier.  May be none, acc(accelerator or agent), sys(system).  See HSA AQL spec for additional information.
- ^duration shows the command duration measured in us.  This is measured using the GPU timestamps and represents the command execution on the acclerator device.
- ^start shows command start time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- ^stop shows command stop time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- ^id shows the command id in device.queue.cmd format.  (if HCC_PROFILE_VERBOSE & 0x4).  The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue. uniquely identifies the command during the process execution.

### Overhead
The hcc profiler does not add any additional synchronization between commands or queues.
Profile information is recorded when a command is deleteed.
The profile mode will allocate a signal for each command to record the timestamp information. This can add 1-2 us to the overall program execution for command which do not already use a completion signal.  However, the command duration (start-stop) is still accurate.
Trace mode will generate strings to stderr which will likely impact the overall application exection time.  However, the GPU duration and timestamps are still valid.
Summary mode accumulates statistics into an array and should have little impact on application execution time.


##Additional Details and tips
- Commands are logged in the order they are removed from the internal HCC command tracker. Typically this is the same order that commands are dispatched, though sometimes these may diverge.  For example, commands from different  devices,queues, or cpu threads may be interleaved on the stderr display.  If a single view in timeline order is required, enable and sort by the profiler timestamps (HCC_PROFILE_VERBOSE=0x2)
- If the application keeps a reference to a completion_future, then the command timestamp may be reported significantly after it occurs. 
- HCC_PROFILE has an (untested) feature to write to a log file.


