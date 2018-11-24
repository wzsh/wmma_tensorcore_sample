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
