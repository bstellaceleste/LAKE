/*
 * Trying to use the lake kapi with a vector_add hello.
 */


#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include "hello.h"
#include <linux/delay.h>


static int devID = 0;
module_param(devID, int, 0444);
MODULE_PARM_DESC(devID, "GPU device ID in use, default 0");

static char *cubin_path = "vectoradd.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to the cubin file.");

// we cannot define the "__global__ void vectorAdd" here, like in a C program, because we are inside the kernel; the kernel module code is not familiar to cuda expressions. We will instead create a vector_add.cu file and call it usign cuModuleGetFunction and cuModuleLoad

static int vector_add(void)
{
    int i, N = 4, blocksPerGrid, threadsPerBlock = 16;
    CUdevice dev;
    CUcontext ctx;	
    CUmodule mod;
    CUfunction vectorAdd, hello_kernel;
	u64 t_start, t_stop, t_stop2;
    unsigned long *h_A, *h_B, *h_C;
    CUdeviceptr d_A, d_B, d_C;
    size_t size = N * sizeof(float);

	PRINT(V_INFO, "Running vector_add hello.\n");

	// Get the GPU ready
    cuInit(0);
	check_error(cuDeviceGet(&dev, devID), "cuDeviceGet", __LINE__);
	check_error(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate", __LINE__);
    check_error(cuModuleLoad(&mod, cubin_path), "cuModuleLoad", __LINE__);
    check_error(cuModuleGetFunction(&vectorAdd, mod, "_Z9vectorAddPmS_S_i"),
            "cuModuleGetFunction", __LINE__);
    // check_error(cuModuleGetFunction(&hello_kernel, mod, "_Z12hello_kernelPii"),
    //         "cuModuleGetFunction", __LINE__);

    // 1. Allocate Host Memory
    h_A = kava_alloc(size);
    h_B = kava_alloc(size);
    h_C = kava_alloc(size);

    // Initialize host arrays -- should look into how the kava allocator is built to understand why we can only allocate a certain amount of mem here (I tried with N>=10 and it didn't worked, I was having h_C[0] returning the values of h_B)
    for (i = 0; i < N; ++i) {
        h_A[i] = 1;
        h_B[i] = 24;
        h_C[i] = 3;
    }
	PRINT(V_INFO, "Printing resulting array after allocation: %lu\n", h_C[0]);

    // 2. Allocate Device Memory (GPU)
    check_error(cuMemAlloc((CUdeviceptr *)&d_A, size), "cuMemAlloc d_A", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr *)&d_B, size), "cuMemAlloc d_B", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr *)&d_C, size), "cuMemAlloc d_C", __LINE__);

    // 3. Copy data from Host to Device
    check_error(cuMemcpyHtoD(d_A, h_A, size), "cuMemcpyHtoD", __LINE__);
    check_error(cuMemcpyHtoD(d_B, h_B, size), "cuMemcpyHtoD", __LINE__);

     // 4. Launch Kernel
    t_start = ktime_get_ns();

    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    PRINT(V_INFO, "Bitchebe: Function %s, Line %d - blocksPerGrid = %d\n", __func__, __LINE__, blocksPerGrid);
	void *args[] = {
		&d_A, &d_B, &d_C, &N
	};

    /* 
    * cuLaunchKernel is the equivalent of the C/C++ angle bracket expression: vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N). To avoid the latter, we can use cuLaunchKernel or cudaLaunchKernel(func, numBlocks, threadsPerBlock, shared_mem, kernel_args, stream_id)
    * numBlocks and threadsPerBlock are dim3 structures of the form (numBlocks, 1, 1) and (threadsPerBlock, 1, 1)
    * given this, we can write the equivalent cuLaunKernel expression
    */
    check_error(cuLaunchKernel(vectorAdd, 
					blocksPerGrid, 1, 1,
					threadsPerBlock, 1, 1, 
					0, NULL, args, NULL),
				"cuLaunchKernel", __LINE__);

    PRINT(V_INFO, "Bitchebe: Function %s, Line %d\n", __func__, __LINE__);
	t_stop = ktime_get_ns();
	cuCtxSynchronize();
	t_stop2 = ktime_get_ns();
    PRINT(V_INFO, "Times (us): %llu, %llu\n", (t_stop - t_start)/1000, (t_stop2 - t_start)/1000);

	// if ( !check_error(cuMemcpyDtoH(h_C, d_C, size), "cuMemcpyDtoH", __LINE__) )
    // {
    //     PRINT(V_INFO, "cuMemcpyDtoH error, exiting!\n");
    //     return 0;
    // };
	PRINT(V_INFO, "Printing resulting array before copy: %lu\n", h_C[0]);
    check_error(cuMemcpyDtoH(h_C, d_C, size), "cuMemcpyDtoH", __LINE__);
	PRINT(V_INFO, "Printing resulting array after copy: %lu\n", h_C[0]);
	
	cuCtxSynchronize();
	cuMemFree(d_A);
	cuMemFree(d_B);
	cuMemFree(d_C);
    PRINT(V_INFO, "Bitchebe: Function %s, Line %d\n", __func__, __LINE__);
	kava_free(h_A); // because of the the kava allocator is built, you should only use free once because it merges all other shunks in the shared mem
	// kava_free(h_B);
	// kava_free(h_C);

	return 0;
}


static int __init vector_add_init(void)
{
	return vector_add();
}

static void __exit vector_add_end(void)
{
}

module_init(vector_add_init);
module_exit(vector_add_end);

MODULE_AUTHOR("StellaTest");
MODULE_DESCRIPTION("Example kernel module of using CUDA in lake");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
