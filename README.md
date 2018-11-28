# Matrix Multiply-Accumulate(MMA) on GPU 
### Sample code for undergrads on the Capstone Project Course of Hallym university in autumn semester 2018.
**Purpose:** To implement and measure performance of Matrix Multiply-Accumulate(like D = A * B + C) on CPU, GPU (with/without Tensor Cores), respectively.

**Note** that this repository only contains the **less performant** version of implementations. It is designed for demonstration purposes only to show how your project should be done.

#### matrix_cpu
includes sample code of MMA with a single thread on CPU

#### matrix_gpu
includes sample code of MMA on GPU without Tensor Cores by CUDA API

#### matrix_wmma
includes sample code of MMA on GPU with Tensor Cores by WMMA API

#### project
To show how your project organized the algorithm implementation, performance metrics and result verification

---

### Tips for compiling *.cu
$ nvcc -o main main.cu -arch sm_75

**Tensor Core is only supported by CUDA compute capability 7.0 and above**

7.0 <=> Volta (Titian V / Quadro GV100)

7.5 <=> Turing (RTX 2080/ RTX 2080 Ti / Quadro RTX 6000)

---

### References
- Programming Tensor Cores in CUDA 9
   - https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/
- How to Implement Performance Metrics in CUDA C/C++
   - https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
- NVIDIA Turing Architecture Whitepaper
   - https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
- NVIDIA Volta Architecture Whitepaper
   - http://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
- Tensorコアを使ってみた
  - http://proc-cpuinfo.fixstars.com/2018/10/tensorcore/
- CUTLASS: Fast Linear Algebra in CUDA C++
   - https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

