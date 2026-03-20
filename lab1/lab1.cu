#include <stdio.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__global__ void square_kernel(double *arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (idx < n) {
        arr[idx] = arr[idx] * arr[idx];
        idx += offset;
    }
}
int main() {
    int n;

    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "ERROR: Failed to read vector size\n");
        return 0;
    }

    if (n <= 0 || n >= (1 << 25)) {
        fprintf(stderr, "ERROR: Invalid vector size. Must be 0 < n < 2^25\n");
        return 0;
    }

    double *arr = (double*)malloc(sizeof(double) * n);
    if (arr == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate host memory\n");
        return 0;
    }

    for (int i = 0; i < n; i++) {
        if (scanf("%lf", &arr[i]) != 1) {
            fprintf(stderr, "ERROR: Failed to read vector element %d\n", i);
            free(arr);
            return 0;
        }
    }

    double *dev_arr;

    CSC(cudaMalloc(&dev_arr, sizeof(double) * n));
    CSC(cudaMemcpy(dev_arr, arr, sizeof(double) * n, cudaMemcpyHostToDevice));

    square_kernel<<<1024, 1024>>>(dev_arr, n);

    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(arr, dev_arr, sizeof(double) * n, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < n; i++) {
        printf("%.10f", arr[i]);
        if (i < n - 1) {
            printf(" ");
        }
    }
    printf("\n");

    CSC(cudaFree(dev_arr));
    free(arr);

    return 0;
}