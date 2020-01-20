# Hardware Implementation

a multiprocessor is designed to execute hundreds of threads concurrently.

the instructiions are pipelined, leveraging instruction-level 

## 4.1 SIMT Architecture

the multiprocessor creates, manages, schedules and executes threads in groups of 32 parallel threads called warps. Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. The term warp origine
