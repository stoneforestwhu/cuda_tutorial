## chapter2 Heterogeneous Programming

### 2.4 Heterogeneous Programming(异构编程)

## chapter3 Programming Interface

### 3.2 CUDA Runtime

静态链接库 cudart.lib libcudart.a

动态链接库 cudart.dll cudart.so



page-locked host memory 页锁定的主机内存

asynchronous concurrent execution (异步并发执行)

multi-device system 单机多卡系统

error checking

call stack

texture and surface memory

graphics interoperability

#### 3.2.1 Initialization

​        运行时没有显式的初始化函数,当地一个运行时函数被调用时直接就初始化了(除了错误处理和版本管理的函数).当计算运行时函数的运行时间和解释第一次调用运行时函数的错误码时必须时刻记住这一点.

​        在初始化过程中,运行时在系统中为每个设备(gpu)创建了一个CUDA上下文.这个上下文只是这个设备的一个初始上下文,并且在应用的所有主机线程中共享.作为上下文创建的一部分,如果需要的话,设备的代码是实时编译的,并且加载到设备的内存中. 所有这一切全是透明的.如果需要,对于驱动API的互操作性(interoperability),设备的基础上下文能够从驱动的API中获取

​       当一个主机线程调用cudaDeviceReset(), 这个操作销毁了主机线程当前操作的设备的初始上下文.下一次任何将这个设备作为current的主机线程调用运行时函数都会为这个设备创建一个新的初始上下文.

```
cuda接口使用的全局状态(global state)是在主机程序初始化时初始化的,在主程序终结时被销毁的.CUDA运行时和驱动不能检测到全局状态(global state)是否有效,所以在主程序初始化或者main函数结束后使用这些接口都会导致未定义行为
```

#### 3.2.2 Device Memory

​         在异构编程中,CUDA编程模型假定系统由一个主机(host)和一个设备(device)组成, 而且每个都有自己独立的内存.核函数运行在设备内存中,所以运行时需要提供函数来分配,回收和复制设备的内存,同时在主机内存和设备内存之间传输数据.

​         设备内存被分配成线性内存或者CUDA数组.

​         CUDA数组的内存布局时不透明的,专门为了获取纹理做了优化. 这一点在章节3.2.11 Texture and Surface Memory中描述.

​         线性内存是在单个统一的地址空间中分配的,这就意味着独立分配的实体能够通过指针来引用,例如二叉树和链表.地址空间的大小取决于主机(host)系统和GPU的compute capability

```
在compute capability 为5.3或者更早的显卡中,CUDA驱动创建一个未提交的40bit虚拟保留地址来保证分配的内存在支持的范围中.这个保留呈现为一个保留的虚拟地址,但是并不占用任何物理地址直到程序真正分配了内存
```

​        线性存储器一般使用cudaMalloc()来分配内存, 使用cudaFree()来释放内存, 使用cudaMemcpy()来在主机内存和设备内存之间传输数据.在向量加的代码例子中,向量需要从主机内存拷贝到设备内存.

```c++
__global__ void VecAdd(float* A, float* B, float* C, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N)
		C[i] = A[i] + B[i];
}
// Host code
int main()
{
	int N = ...;
	size_t size = N * sizeof(float);
	// Allocate input vectors h_A and h_B in host memory
	float* h_A = (float*)malloc(size);
	float* h_B = (float*)malloc(size);
	// Initialize input vectors
	 ...
	// Allocate vectors in device memory
	float* d_A;
	cudaMalloc(&d_A, size);
	float* d_B;
	cudaMalloc(&d_B, size);
	float* d_C;
	cudaMalloc(&d_C, size);
	// Copy vectors from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	// Invoke kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	// Copy result from device memory to host memory
	// h_C contains the result in host memory
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	// Free host memory
	...
}
```

​        线性存储器也可以通过cudaMallocPitch()和cudaMalloc3D()来分配.推荐使用这两个函数用来分配2D和3D的数组,因为这能确保分配的空间会通过padding来满足章节 5.3.2 Device memory Accesses中描述的字节对齐的要求, 也因此能保证在获取行地址或者使用别的API(cudaMemcpy2D和cudaMemcpy3D)来在2D数组和设备内存的其他区域之间复制数据时获得最好的性能.返回的pitch或者stride必须用来获取数组元素.

​        以下代码是在设备中分配一个二维数组并且循环遍历数组元素的设备代码

```c++
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, pitch, width * sizeof(float), height);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);
// Device code
__global__ void MyKernel(float* devPtr, size_t pitch, int width, int height)
{
	for (int r = 0; r < height; ++r) {
		float* row = (float*)((char*)devPtr + r * pitch);
		for (int c = 0; c < width; ++c) {
			float element = row[c];
		}
	}
}
```

​        以下代码是在设备中分配一个三维数组并且循环遍历数组元素的设备代码

```c++
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);
MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);
// Device code
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
int width, int height, int depth)
{
	char* devPtr = devPitchedPtr.ptr;
	size_t pitch = devPitchedPtr.pitch;
	size_t slicePitch = pitch * height;
	for (int z = 0; z < depth; ++z) {
		char* slice = devPtr + z * slicePitch;
		for (int y = 0; y < height; ++y) {
			float* row = (float*)(slice + y * pitch);
			for (int x = 0; x < width; ++x) {
			    float element = row[x];
            }             
        }
    }
}
```



## chapter4 Hardware Implementation

​        a multiprocessor is designed to execute hundreds of threads concurrently.the instructiions are pipelined, leveraging instruction-level

### 4.1 SIMT Architecture

​        the multiprocessor creates, manages, schedules and executes threads in groups of 32 parallel threads called warps. Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. The term warp origine
