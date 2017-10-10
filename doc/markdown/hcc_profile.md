# HCC Profile Mode

HCC supports low-overhead profiler to trace or summarize command timestamp information to stderr for any HCC  or HIP program.
Tho profiler messages are interleaved with the trace output from the application -  which is handy to identify the region-of-interest and 
can complement deeper analysis with the CodeXL GUI
Additionally, the hcc profiler requires only console mode access and can be used on machine where graphics are not available or are hard to access.

Some other useful features:
- Calculates the actual bandwidth for memory transfers
- Identifies PeerToPeer memory copies
- Shows queue, start, and stop timestamps for each command
- Shows barrier commands and the time they spent waiting to resolve (if requested)

HCC_PROFILE=2 enables a profile message after each command (kernel or data movement) completes. 
HCC_PROFILE=1 is a future capability that will shows a summary of kernel and data commands when hcc exits. (under development)

## One-liners
```
# Trace collection:
$ HCC_PROFILE=2  RunMyApp &> prof.out

# Post-processing examples:
# Show summary
$ rpt prof.out

# Show summary for a specified ROI for lines 100-200 of the file
$ rpt prof.out -r @100 --R @200

# Generate text trace to stdout
$ rpt  prof.out -t

# Generate gui trace for reading with chrome://tracing
$ rpt prof.out -g prof.json
```

## HCC text profile format

### Kernel Commands

This example shows profiled kernel commands:
```bash
$ HCC_PROFILE=2 ./my-hcc-app ...
profile:  kernel;            Im2Col;   17.8 us; 94859076277181; 94859076277181; 94859076294941; #0.3.1;
profile:  kernel;  tg_betac_alphaab;   32.6 us; 94859076277181; 94859537593679; 94859537626319; #0.3.2;
profile:  kernel;     MIOpenConvUni;  125.4 us; 94859076277181; 94860077852212; 94860077977651; #0.3.3;
```
```bash
PROFILE:  TYPE;    KERNEL_NAME     ;  DURATION;        ENQUEUE; START         ; STOP          ; ID
```

- PROFILE:  always "profile:" to distinguish it from other output.
- TYPE: the command type : kernel, copy, copyslo, or barrier.  The examples and descriptions in this section are all kernel commands.
- KERNEL_NAME: the (short) kernel name.
- DURATION: command duration measured in us.  This is measured using the GPU timestamps and represents the command execution on the acclerator device.
- ENQUEUE: When the command is enqueued to the GPU from the host, measured in ns and using host-side timers.  This timestamp is collected inside host API that enqueues the command.
- START: When the command starts executing, measured in ns.  For "fast" command executed by the GPU copy engines, this is measured using GPU-side timers.
- STOP: When the command stops executing, measured in ns.  For "fast" command executed by the GPU copy engines, this is measured using GPU-side timers.
- ID: command id in device.queue.cmd format.  The cmd number is a mononotically increasing per-queue, so the triple of device.queue.cmd uniquely identifies the command during the process execution.

### Memory Copy Commands
This example shows memory copy commands with full verbose output:
```
profile: copyslo; HostToDevice_sync_slow;   909.2 us; 94858703002; 94858703102; 94858704012; #0.0.0; 2359296 bytes;  2.2 MB;   2.5 GB/s;
profile:    copy; DeviceToHost_sync_fast;   117.0 us; 94858726008; 94858726408; 94858726525; #0.0.0; 1228800 bytes;  1.2 MB;   10.0 GB/s;
profile:    copy; DeviceToHost_sync_fast;     9.0 us; 94858726068; 94858726668; 94858726677; #0.0.0; 400 bytes;      0.0 MB;   0.0 GB/s;
profile:    copy; HostToDevice_sync_fast;    15.2 us; 94858727039; 94858727639; 94858727654; #0.0.0; 9600 bytes;     0.0 MB;   0.6 GB/s;
profile:    copy; HostToDevice_async_fast;  131.5 us; 94858729098; 94858729198; 94858729330; #0.6.1; 1228800 bytes;  1.2 MB;   8.9 GB/s;
PROFILE:  TYPE;    COPY_NAME             ;  DURATION;     ENQUEUE;       START;        STOP;  ID   ; SIZE_BYTES;     SIZE_MB;  BANDWIDTH;
```

- PROFILE:  always "profile:" to distinguish it from other output.
- TYPE: the command type : kernel, copy, copyslo,or barrier.  The examples and descriptions in this section are all copy or copyslo commands.
- COPY_NAME has 3 parts: 
    - Copy kind: HostToDevice, HostToHost, DeviceToHost, DeviceToDevice, or PeerToPeer.  DeviceToDevice indicates the copy occurs on a single device while PeerToPeer indicates a copy between devices.
    - Sync or Async.  Synchronous copies indicate the host waits for the completion for the copy. Asynchronous copies are launched by the host without waiting for the copy to complete.
    - Fast or Slow.  Fast copies use the GPUs optimized copy routines from the hsa_amd_memory_copy routine.  Slow copies typically involve unpinned host memory and can't take the fast path.
    - For example `HostToDevice_async_fast.
- DURATION: command duration measured in us.  This is measured using the GPU timestamps and represents the command execution on the acclerator device.
- ENQUEUE: When the command is enqueued to the GPU from the host, measured in ns and using host-side timers.  This timestamp is collected inside host API that enqueues the command.
- START: When the command starts executing, measured in ns.  For "fast" command executed by the GPU copy engines, this is measured using GPU-side timers.
- STOP: When the command stops executing, measured in ns.  For "fast" command executed by the GPU copy engines, this is measured using GPU-side timers.
- ID: command id in device.queue.cmd format.  The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue.cmdseqnum uniquely identifies the command during the process execution.
- SIZE_BYTES: the size of the transfer, measured in bytes.
- SIZE_MB: the size of the transfer, measured in megabytes.
- BANDWIDTH: the bandwidth of the transfer, measured in GB/s.

### Barrier Commands

An example barrier command with full vebosity:
```
profile: barrier; deps:0_acq:none_rel:sys;  5.3 us; 94858731419410;  94858731419410; 94858731424690; #0.0.2;
PROFILE:  TYPE;   BARRIER_NAME           ;  DURATION; ENQUEUE; START         ; STOP          ; ID    ; 
```
- PROFILE:  always "profile:" to distinguish it from other output.
- TYPE: the command type: either kernel, copy, copyslo, or barrier.  The examples and descriptions in this section are all copy commands.  Copy indicates that the runtime used a call to the fast hsa memory copy routine while copyslo indicates that the copy was implemented with staging buffers or another less optimal path.  copy computes the commands using device-side timestamps while copyslo computes the bandwidth based on host timestamps. 
- BARRIER_NAME has 3 parts: 
    - deps:#  - the number of input dependencies into the barrier packet.
    - acq:    - the acquire fence for the barrier.  May be none, acc(accelerator or agent), sys(system).  See HSA AQL spec for additional information.
    - rel:    - the release fence for the barrier.  May be none, acc(accelerator or agent), sys(system).  See HSA AQL spec for additional information.
- DURATION: command duration measured in us.  This is measured using the GPU timestamps from the time the barrier reaches the head of the queue to when it executes.  Thus this includes the time to wait for all input dependencies, plus the previous command to complete, plus any fence operations performed by the barrier.
- ENQUEUE: When the command is enqueued to the GPU from the host, measured in ns and using host-side timers.  This timestamp is collected inside host API that enqueues the command.
- START: When the command starts executing, measured in ns.  For "fast" command executed by the GPU copy engines, this is measured using GPU-side timers.
- STOP: When the command stops executing, measured in ns.  For "fast" command executed by the GPU copy engines, this is measured using GPU-side timers.
- ID: the command id in device.queue.cmd format.  The cmdsequm is a unique mononotically increasing number per-queue, so the triple of device.queue.cmdseqnum uniquely identifies the command during the process execution.

### Overhead
The hcc profiler does not add any additional synchronization between commands or queues.
Profile information is recorded when a command is "disposed" by the HCC runtime.  This may differ from the order that commands are dispatched ; the rpt tool should be used to recover the original command order.
The profile mode will allocate a signal for each command to record the timestamp information. This can add 1-2 us to the overall program execution for command which do not already use a completion signal.  However, the command duration (start-stop) is still accurate.
Trace mode will generate profile records to stderr which will likely impact the overall application exection time.  However, the GPU duration and timestamps are still valid.


### Additional Details and tips
- Commands are logged in the order they are removed from the internal HCC command tracker. Typically this is the same order that commands are dispatched, though sometimes these may diverge.  For example, commands from different  devices,queues, or cpu threads may be interleaved on the hcc trace display to stderr.  If a single view in timeline order is required, enable and sort by the profiler START timestamps.
- If the application keeps a reference to a completion_future, then the profile record may be reported significantly after it occurs. rpt can be used to print profile records in time-order.
- Some timers are collected on the GPU and some on the CPU.  These are normalized to the same clock domain and can compared.
- Barriers include the time spent waiting for their dependents to complete.  Barriers may begin executing before their dependents do, and can appear to have a very long execution time 
- HCC_PROFILE has an (untested) feature to write to a log file.
- For profiling python commands with stderr and stdout, it can be useful to merge stdout into stderr so the RPT messages and markers are better interleaved with the text from the application.

## rpt
ROCm Profiling Tool ("rpt") is a command-line tool that post-processes the output from the HCC_PROFILE mode.   Key features include:
- Sort profile records by time. 
- Compute gaps where GPUs or data busses are unused.
- Display profile summaries for each resource (see below)
- Display a textual trace of profile records.
- Display a graphical trace of profile records. (can be viewed in chrome://tracing mode)
- Powerful Region-of-Interest capabilities to select which regions contribute to the summary statistics.  Useful for example to skip over initialization and warmup code.

#### Resources and Gaps
rpt organizes profile records into the parent resources where they execute.  Resources can currently be GPUs (numbered 0..N-1) or the DATA bus.
GPU resources execute kernel and barrier commands.  DATA resources execute copy commands.

Gaps indicate periods of times when the resource is not utilized.  rpt computes gap time by examining the delta between the stop and start time of consecutive records on the resource.
Commands may be overlapped or completely hidden if a command starts eariler - in this case the duration will be 0.

The summary view uses a histogram to aggregate gaps into buckets and will show the range covered by the bucket.  For example, the "gap >=10000us" buckets sumarizes all gaps which are equal or more than 10000us.  Gaps are inclusive of the lower bound and exclusive of the upper bound.  The --gaps option can be used to customize the histogram boundaries and this can be useful to isolate causes of gaps in some cases.

Large gaps can indicate the ROI is not set correctly - see the next section for suggestions on how to set the ROI.  Inside a valid ROI:

- Gaps of <10us typically indicate GPU hardware overhead - this is maximum rate that the GPU can execute back-to-back commands in the same queue.
If this is significant overhead, it may be beneficial to combine kernels, or use multiple queues if the work is independent.

- Larger gaps can indicate excessive host serialization or cases where the GPU was not fed commands frequently enough, perhaps due to some expensive CPU activity.  To analyze the cause of these gaps, you can use a profiler such as operf or can use the --text_trace or --gui_trace options combined with progtram instrumentation (stderr output) to identify what is occurring in the gap regions. 


#### Specifying ROI
A common problem when performing profiling analysis is to determine which region to focus the analysis on, and in particular to exclude initialization, warmup, and tear-down activity from the results.  RPT provides options to specify the start and stop of the "Region of Interest" (ROI) using line numbers from the input file, timestamps, or regular expressions.  The latter is particularly useful - modify the program to emit a marker message and then search for the specified marker.

ROI_POINTs can be specified with line numbers, start times, or search strings.
- @LINENUM : specify a line number from the input file.  Example: @1342.\n"\
- ^TIME    : specify start time from the beginning of the file.  Example: ^55.12345\n"\
- MATCH_NUM%SEARCH_RE : Specify string in python regular expression format, rpt will use the profile record after the found text. MATCH_NUM specifies the nth match in the file of the string.  This allows applications to embed markers using simple stderr print statements and instruct rpt to include records only between the markers. Examples: %MyStartMarker, 2%MyStartMarker (finds 2nd instance of MyMarker), %"^iteration *10"

#### Text trace
The `-t` option generates a text trace, with one row per profile record.  Gaps between the records are also computed and displayed.

```
Resource        Start(ms) +Time(us)    Type #Dev.Queue.Cmd    LineNum Name
GPU0       980.530247:      +0.00 barrier #0.2.2586   195139: depcnt=0,acq=none,rel=sys
DATA       166.264953: +814268.55 gap                ==============
DATA       980.533506:      +2.67 copy    #0.2.2587   195140: HostToDevice_async_fast_128_bytes
GPU0       980.530247:     +11.70 gap                ====
GPU0       980.541950:      +0.00 barrier #0.2.2588   195138: depcnt=1,acq=none,rel=acc
GPU0       980.541950:    +223.59 gap                ==========
GPU0       980.765535:      +5.63 kernel  #0.1.192922   201310: TensorKernel
GPU0       980.771165:     +14.37 gap                ====
GPU0       980.785536:      +0.00 barrier #0.1.192923   201309: depcnt=0,acq=none,rel=acc
GPU0       980.785536:      +5.94 gap                ==
GPU0       980.791477:      +0.00 barrier #0.3.304   195141: depcnt=1,acq=none,rel=acc
GPU0       980.791477:      +7.70 gap                ==
GPU0       980.799181:      +0.00 barrier #0.3.305   195143: depcnt=0,acq=none,rel=sys
DATA       980.536173:    +272.19 gap                ==========
DATA       980.808366:      +2.37 copy    #0.3.306   195144: DeviceToHost_async_fast_4_bytes
```
- Resource : The resource this command executed on.  May be GPU# for a specific GPU or DATA to indicate data transfer command
- Time(us) : Time which the command contributed to the 'critical' path
- Start(ms) : Start time of the command, displayed in ms.  Command start from time 0
- Type  : Type of the command (kernel,barrier, copy)
- Dev.Queue.Cmd : The cmd sequm is a unique mononotically increasing number per-queue, so the triple of device.queue.cmdseqnum uniquely identifies the command during the process execution.
- LineNum : Line number from the input HCC_PROF_FILE
- Name  : Name of the record.  For kernels, this is kernel name.  For copy commands, the name contains the size and direction.  For barrier commands this is contains dependency counts, acquire and release fences.

Text output from the application(is shown interleaved with the profile records, preceded by leading ">".
The --hide_app_text option can be used to supress this if desired.

The gaps are shown with a series of '==" indicating their length - longer bars indicating a longer gap..  The --gaps parmater (or defaults if not specified) also controls the split points for the bar display.
The intent is to provide a visually distinct way to visualize the length of the gaps.

The combined view interleaves the records for all resources, sorted by their start order.
However, the time delta and gap computation are computed on a per-resource basis.
For example, in the case above the large "DATA" gaps show the time since the last DATA command.

Copy commands show:
- Copy kind: HostToDevice, HostToHost, DeviceToHost, DeviceToDevice, or PeerToPeer.  DeviceToDevice indicates the copy occurs on a single device while PeerToPeer indicates a copy between devices.
- Sync or Async.  Synchronous copies indicate the host waits for the completion for the copy. Asynchronous copies are launched by the host without waiting for the copy to complete.
- Fast or Slow.  Fast copies use the GPUs optimized copy routines from the hsa_amd_memory_copy routine.  Slow copies typically involve unpinned host memory and can't take the fast path.
- Size of the copy in bytes.

Barrier commands show:
- depcnt=N : Number of other barriers this one depends on.
- acq      : The "acquire" fence executed before the barrier.  May be "none", acc(meaning "accelerator" or "agent"), or "sys" ("system")
- rel      : The "release" fence executed before the barrier.  May be "none", acc(meaning "accelerator" or "agent"), or "sys" ("system")
- Dependent list (Under development) : List of dependents

More information is available in the HSA Platofrm System Architecture Specification section 2.9 ("Architected Queueing Language").

The ROI controls apply to the text trace as well, and are useful to limit the generated text.

#### GUI trace

The `-g` or `--gui_trace` option generates a JSON-format file that can be input into chrome://tracing.
Chrome tracing was originally designed for viewing browser performance information but is also a fully capable timeline viewing tool which is readily available on many platforms.
To access the gui, start the Chrome brower and open the site "chrome://tracing".  In the browser:
    - Select "Load" from the upper left hand corner and navigate to the saved JSON file.
    - Select "?" from the upper right hand corner to see the navigation options.
        - w/s Zoom in/out
        - a/d Pan left/right
        - Many more options available.

In the timeline view, each row shows a GPU queue or data resource.  Rectangles indicate commands that execute on that row.  More information can be obtained by clicking on the rectangle.

The ROI controls apply to the giu trace as well.

```
# Use rpt to create a json gui trace file
$ rpt prof.out -g prof.json
```
Start the Chrome brower and open the site "chrome://tracing".  In the browswer, select "Load" from the upper left hand corner and navigate to the saved JSON file.




### Example profiling analysis

```
# Collect the profiling information for the application
$ HCC_PROFILE=2 myapp &> myapp.prof
```

In this example, we had previously modified the program to print the message "iteration start" before every 10 iterations in the critical loop.
We use the markers to filter the information parsed by roi, so we are examining exactly one iteration (specifically the profile records printed between the 3rd and 4th ocurrences of the phrase "iteration start" at the beginning of a line).  By default, rpt emits a summary table showing the top users of each resource:

```
$ rpt -r 3%"^iteration start" -R 4%"^iteration start"

ROI_START: GPU0         0.000000:      +0.00  kernel   195169: miog_alphaab                   #0.1.199064
ROI_STOP : GPU0      6819.188139:      +0.00  kernel   288991: miog_betac_alphaab             #0.1.282703
ROI_TIME=   6.819 secs
Resource=GPU0 Showing 20/84 records   78.43% busy
      Total(%)    Time(ms)    Calls  Avg(us)  Min(us)  Max(us)  Name
        17.69%  1206185.8    23064     52.3     20.1    676.2  miog_betac_alphaab
        11.34%   773334.6       10  77333.5  75585.2  84103.3  gap>=10000us
         9.17%   625546.4      769    813.5    204.0   1810.3  MIOpenConv1x1
         7.98%   544091.0      420   1295.5    417.9   3776.7  sp3AsmConvRxSF
         6.77%   461441.2    81499      5.7      0.1      9.6  gap<10us
         5.60%   382048.1     7718     49.5     28.7    145.2  miog_alphaab
         4.55%   310281.5    11200     27.7     12.0     95.6  Col2Im
         3.39%   231113.1       50   4622.3   2562.4  12843.2  MIOpenConvUni
         3.37%   229728.0    12478     18.4      7.0     65.9  Im2Col
         2.80%   190663.2      273    698.4    121.0   3323.0  mloPooling
         2.25%   153615.5       99   1551.7    700.8   7101.0  gcnAsmConv3x3WrW
         1.99%   135681.8      199    681.8    416.3   2355.9  sp3AsmConv3x3F
         1.80%   122478.6       85   1440.9    191.4   2055.6  mloPoolingAveBwd
         1.55%   105427.5       90   1171.4    693.3   1438.2  MLOpenCvBwdWrWLmap
         1.48%   106004.4      940    112.8     17.5   1590.7  TensorAdder::ScalarSumSquares<float>
         1.38%    94248.5      331    284.7    101.0    997.7  gap<1000us
         1.37%    93703.6       40   2342.6   1347.6   5417.6  MIOpenCvBwdWrW
         1.29%    87982.8       26   3384.0   1035.7   7178.7  gap<10000us
         1.20%    82023.0      280    292.9     53.2   1909.4  BatchNormBwdSpatialDX
         1.05%    71775.9     2340     30.7      3.9    131.7  TensorOp

Resource=DATA Showing 8/8 records    0.00% busy
      Total(%)    Time(ms)    Calls  Avg(us)  Min(us)  Max(us)  Name
        93.20%  6355648.1        9 706183.1 693182.3 716972.0  gap>=10000us
         0.12%     8100.3       21    385.7    161.7    636.0  gap<1000us
         0.01%      666.9        8     83.4     76.4     94.5  gap<100us
         0.00%       70.1        2     35.0     24.7     45.3  gap<50us
         0.00%       67.7       30      2.3      1.9      2.7  DeviceToHost_async_fast_4_bytes
         0.00%       41.4        9      4.6      2.7      6.3  gap<10us
         0.00%       28.6       10      2.9      2.7      3.1  HostToDevice_async_fast_4_bytes
         0.00%       27.7       10      2.8      2.7      3.1  HostToDevice_async_fast_128_bytes
```
 

#### Observations and further analysis
- This GPU is used only ~78% of the time.  The remaining 22% of time is displayed in the "gap" categories ; about half this comes from the second row showing 10 gaps averaging 77ms.
These are relatively large gaps and likely indicate the CPU is not feeding the GPU quickly enough.   One approach would be to use the `-t` option and examine the kernels and other profile activity around the gap.
Another approach is to use oprofile (see the next section).

- The miog_betac_alphaab and MIOpenConv1x kernels take 17.69% and 9.17% of the overall GPU execution time respectively.  Optimizing thise or choosing alternate algorithms could improve performance significantly.

- The DATA resource is used very infrequently, nearly 0% active with only 8 records and plenty of gaps.


## oprofile
### Introduction
- http://oprofile.sourceforge.net/docs/
- https://www.ibm.com/support/knowledgecenter/linuxonibm/liacf/oprofile_pdf.pdf

### Installation
$ apt-get install oprofile

### Data Collection

Start the program of interest.
```
$ ./myapp
```

In another terminal, determine the process id (perhaps with `pgrep`) and pass to 
```
$ operf --pid `pgrep myapp`
```

By default, operf will backup any previously recorded perf data and create a new file.

### Data Reporting

```
# Show callgraph
$ opreport -cl 

# Show hotspots in code:
$ opreport -dg
```
