A hands-on repository for learning inline PTX in CUDA, with a focus on matrix multiplication and addition (mma) using Tensor Core. Comprehensive documentation references are provided to support learning.

## Basics
**Tensor Cores in CUDA:** 
Tensor Cores are specialized hardware units in NVIDIA GPUs designed to dramatically accelerate matrix multiply-accumulate (mma) operations. They support a range of precisions, including FP16, BF16, TF32, INT8, INT4, and FP64. <br>
**Parallel Thread Execution (PTX):** 
PTX (Parallel Thread Execution) is NVIDIA’s low-level instruction set architecture (ISA) within the CUDA programming environment. It serves as a virtual architecture code and acts as the intermediate representation (IR) for GPU code.
<br>
**Role of PTX in CUDA Compilation Flow**
PTX bridges high-level CUDA code and GPU hardware by serving as an intermediate representation, which is later compiled into SASS (real architecture code) for execution.

*References* <br>
[Cuda Compilation from Lei Mao Blog Post](https://leimao.github.io/blog/CUDA-Compilation/) <br>
[Nvidia Technical Blog for Fat Binaries](https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/)


## Tensor Core Programming Guide
**Programming for Tensor Cores**
The Warp Matrix Multiply-Accumulate (WMMA) API enables CUDA programs to use Tensor Cores for faster matrix computations. However, since it provides higher-level memory abstractions, fine-grained memory operations are often handled using inline PTX. The repository focuses on the use of inline PTX instructions for these fine-grained operations.

*References* <br>
[Nvidia Technical Blog for Programming Tensor Cores](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
<br>

**Inline-PTX** <br>
Inline PTX allows embedding low-level PTX instructions directly within CUDA code for fine-grained control, making it easy to exploit PTX-level matrix multiply-accumulate (MMA) operations for Tensor Core optimization.

*References* <br>
[Inline PTX Assembly in CUDA](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)

## Instructions
**Matrix multiply-accumulate (mma)**
MMA operations are executed at the warp level, with each thread contributing its register-stored data to the collective matrix computation. Depending on the data type and matrix size, each thread can hold multiple registers. A collection of these registers is referred to as a vector expression in PTX.

*References* <br>
[mma instruction in PTX Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma)

**ldmatrix** <br>
Collective load operation from shared memory to register.

*References* <br>
[ldmatrix instruction in PTX Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)

**stmatrix** <br>
Collective store operation from shared memory to register.

*References* <br>
[stmatrix instruction in PTX Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix)


## NOTES
Important notes for the Target ISA regarding double-precision MMA operations are detailed below.

* .f64 floating point type mma operation with .m8n8k4 shape requires sm_80 or higher.

* .f64 floating point type mma operation with .m16n8k4, .m16n8k8, and .m16n8k16 shapes require sm_90 or higher.

## Code Sample
A simple example (mma.cu) is provided to demonstrate matrix multiply-accumulate operations in double precision.



