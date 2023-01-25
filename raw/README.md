# Summary of files

A brief summary of each file in the `raw` folder.

`allocator_benchmark.py` - ??

`collectives.py` - example illustrating all reduce and broadcast. Investigate why broadcast and
broadcast async have different perf.

`cpu_math.py` - performs matrix operations on the cpu

`cuda_allocator.py` - attempt to understand what allocations look like with cuda caching allocator.
It isn't clear how allocation can be studied with the current code snippet.

`cuda_memory_allocation_understand.py` - digs a bit deeper into memory allocation

`cuda_vs_cpu_sort.py` - compares cpu vs gpu sort

`dcgm_mwe.py` - experiments with dcgm library for gpu stats, not relevent for gpu puzzlers (%%)

`ddp_with_profiling.py` - DDP hello world with profiling enabled

`embedding_mwe.py` - ??

`flops_bw.py` - Run matrix multiplication and addition in different streams and see if there is
concurrency since the former is flops bound and the latter is bandwidth bound. This can be a great
example to show the value of utilizing the GPU well using different streams.

`foreach_mwe.py` - horizontal operator fusion example using foreach

`kineto_resnet50.py` - profiling the resnet model

`memory_bounce_scalene.py` - exploring the scalene profiler (%%)

`memory_leak.py` - profile memory in the presence and absence of leaks

`nondeterministic_compute.py` - find non determinism in numerics; no example found yet (%%%)

`nondeterministic_scatter.py` - ??

`pinned_memory.py` - measure perf when copying from pinned vs. non-pinned memory

`python_function_trace.py` - the last code block shows how to enable python tracing (%%)

`run_send_recv_hello_world.sh` - bash utility script for send_recv_hello_world.py

`send_recv_hello_world.py` - mwe showing how send and receive show up on a trace (%%)

`static_runtime_mwe.py` - ?? (%%)

`stream_sync_check.py` - illustrates race condition across streams

`tensor_core.py`- shows the relative difference in performance across fp32, tf32 and fp16 gemms

`torchscript_mwe.py` - torchscript mwe

`vector_ops.py` - achieved flops and bandwidth for various operations
