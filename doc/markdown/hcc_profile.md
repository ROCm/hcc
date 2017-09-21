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
0x10 : Show logs for barrier commands. 

## Sample Output

### Kernel Commands

This shows the simplest trace output for kernel commands with no additional verbosity flags:
```bash
$ HCC_PROFILE=2 ./my-hcc-app ...
profile:  kernel;            Im2Col;   17.8 us;
profile:  kernel;  tg_betac_alphaab;   32.6 us;
profile:  kernel;     MIOpenConvUni;  125.4 us;
```
```
PROFILE:  TYPE;    KERNEL_NAME     ;  DURATION;
```

This example shows profiled kernel commands with full verbose output:
```bash
$ HCC_PROFILE=2 HCC_PROFILE_VERBOSE=0xf ./my-hcc-app ...
profile:  kernel;            Im2Col;   17.8 us;  94859076277181; 94859076294941; #0.3.1;
profile:  kernel;  tg_betac_alphaab;   32.6 us;  94859537593679; 94859537626319; #0.3.2;
profile:  kernel;     MIOpenConvUni;  125.4 us;  94860077852212; 94860077977651; #0.3.3;
```
```bash
PROFILE:  TYPE;    KERNEL_NAME     ;  DURATION;  START         ; STOP          ; ID
```

- PROFILE:  always "profile:" to distinguish it from other output.
- TYPE: the command type : kernel, copy, copyslo, or barrier.  The examples and descriptions in this section are all kernel commands.
- KERNEL_NAME: the (short) kernel name.
- DURATION: command duration measured in us.  This is measured using the GPU timestamps and represents the command execution on the acclerator device.
- START: command start time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- STOP: command stop time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- ID: command id in device.queue.cmd format.  (if HCC_PROFILE_VERBOSE & 0x4).  The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue.cmdseqnum uniquely identifies the command during the process execution.

### Memory Copy Commands
This example shows memory copy commands with full verbose output:
```
profile: copyslo; HostToDevice_sync_slow;   909.2 us; 94858703102; 94858704012; #0.0.0; 2359296 bytes;  2.2 MB;   2.5 GB/s;
profile:    copy; DeviceToHost_sync_fast;   117.0 us; 94858726408; 94858726525; #0.0.0; 1228800 bytes;  1.2 MB;   10.0 GB/s;
profile:    copy; DeviceToHost_sync_fast;     9.0 us; 94858726668; 94858726677; #0.0.0; 400 bytes;      0.0 MB;   0.0 GB/s;
profile:    copy; HostToDevice_sync_fast;    15.2 us; 94858727639; 94858727654; #0.0.0; 9600 bytes;     0.0 MB;   0.6 GB/s;
profile:    copy; HostToDevice_async_fast;  131.5 us; 94858729198; 94858729330; #0.6.1; 1228800 bytes;  1.2 MB;   8.9 GB/s;
PROFILE:  TYPE;    COPY_NAME             ;  DURATION;       START;       STOP;  ID    ; SIZE_BYTES;     SIZE_MB;  BANDWIDTH;
```

- PROFILE:  always "profile:" to distinguish it from other output.
- TYPE: the command type : kernel, copy, copyslo,or barrier.  The examples and descriptions in this section are all copy or copyslo commands.
- COPY_NAME has 3 parts: 
    - Copy kind: HostToDevice, HostToHost, DeviceToHost, DeviceToDevice, or PeerToPeer.  DeviceToDevice indicates the copy occurs on a single device while PeerToPeer indicates a copy between devices.
    - Sync or Async.  Synchronous copies indicate the host waits for the completion for the copy. Asynchronous copies are launched by the host without waiting for the copy to complete.
    - Fast or Slow.  Fast copies use the GPUs optimized copy routines from the hsa_amd_memory_copy routine.  Slow copies typically involve unpinned host memory and can't take the fast path.
    - For example `HostToDevice_async_fast.
- DURATION: command duration measured in us.  This is measured using the GPU timestamps and represents the command execution on the acclerator device.
- START: command start time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- STOP: command stop time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- ID: command id in device.queue.cmd format.  (if HCC_PROFILE_VERBOSE & 0x4).  The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue.cmdseqnum uniquely identifies the command during the process execution.
- SIZE_BYTES: the size of the transfer, measured in bytes.
- SIZE_MB: the size of the transfer, measured in megabytes.
- BANDWIDTH: the bandwidth of the transfer, measured in GB/s.

### Barrier Commands
Barrier commands are only enabled if HCC_PROFILE_VERBOSE 0x

An example barrier command with full vebosity:
```
profile: barrier; deps:0_acq:none_rel:sys;  5.3 us;   94858731419410; 94858731424690; #0.0.2;
PROFILE:  TYPE;   BARRIER_NAME           ;  DURATION; START         ; STOP          ; ID    ; 
```
- PROFILE:  always "profile:" to distinguish it from other output.
- TYPE: the command type: either kernel, copy, copyslo, or barrier.  The examples and descriptions in this section are all copy commands.  Copy indicates that the runtime used a call to the fast hsa memory copy routine while copyslo indicates that the copy was implemented with staging buffers or another less optimal path.  copy computes the commands using device-side timestamps while copyslo computes the bandwidth based on host timestamps. 
- BARRIER_NAME has 3 parts: 
    - deps:#  - the number of input dependencies into the barrier packet.
    - acq:    - the acquire fence for the barrier.  May be none, acc(accelerator or agent), sys(system).  See HSA AQL spec for additional information.
    - rel:    - the release fence for the barrier.  May be none, acc(accelerator or agent), sys(system).  See HSA AQL spec for additional information.
- DURATION: command duration measured in us.  This is measured using the GPU timestamps from the time the barrier reaches the head of the queue to when it executes.  Thus this includes the time to wait for all input dependencies, plus the previous command to complete, plus any fence operations performed by the barrier.
- START: command start time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- STOP: command stop time in ns.  (if HCC_PROFILE_VERBOSE & 0x2)
- ID: the command id in device.queue.cmd format.  (if HCC_PROFILE_VERBOSE & 0x4).  The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue.cmdseqnum uniquely identifies the command during the process execution.

### Overhead
The hcc profiler does not add any additional synchronization between commands or queues.
Profile information is recorded when a command is deleted.
The profile mode will allocate a signal for each command to record the timestamp information. This can add 1-2 us to the overall program execution for command which do not already use a completion signal.  However, the command duration (start-stop) is still accurate.
Trace mode will generate strings to stderr which will likely impact the overall application exection time.  However, the GPU duration and timestamps are still valid.
Summary mode accumulates statistics into an array and should have little impact on application execution time.


### Additional Details and tips
- Commands are logged in the order they are removed from the internal HCC command tracker. Typically this is the same order that commands are dispatched, though sometimes these may diverge.  For example, commands from different  devices,queues, or cpu threads may be interleaved on the hcc trace display to stderr.  If a single view in timeline order is required, enable and sort by the profiler START timestamps (HCC_PROFILE_VERBOSE=0x2)
- If the application keeps a reference to a completion_future, then the command timestamp may be reported significantly after it occurs. 
- HCC_PROFILE has an (untested) feature to write to a log file.


