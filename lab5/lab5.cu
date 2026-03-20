#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MAX_VALUE (1 << 24)
#define MAX_N 135000000
#define BLOCK_SIZE 512
#define GRID_SIZE 512

#define CSC(call)                                                   \
do {                                                                \
    cudaError_t res = call;                                         \
    if (res != cudaSuccess) {                                       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                                                    \
    }                                                               \
} while(0)

__device__ __host__ int conflict_free_index(int idx) {
    return idx + (idx >> 5);
}

void read_input(int *n, int **arr) {
    if (fread(n, sizeof(int), 1, stdin) != 1) {
        fprintf(stderr, "Error reading input size\n");
        exit(0);
    }
    
    *arr = (int*)malloc(*n * sizeof(int));
    if (*arr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(0);
    }
    
    if (fread(*arr, sizeof(int), *n, stdin) != *n) {
        fprintf(stderr, "Error reading input arr\n");
        exit(0);
    }
}

void write_output(int *arr, int n) {
    if (fwrite(arr, sizeof(int), n, stdout) != n) {
        fprintf(stderr, "Error writing output\n");
    }
}

__global__ void histogram_kernel(int* arr, int *hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += offset) {
        int value = arr[i];
        atomicAdd(&hist[value], 1);
    }
}

__global__ void scan_kernel(int *data, int *block_sums, int n_blocks) {
    extern __shared__ int temp[];
    
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    for (int block_offset = block_id; block_offset < n_blocks; block_offset += gridDim.x) {
        int global_idx = block_offset * blockDim.x + thread_id;
        int shared_idx = conflict_free_index(thread_id);
 
        if (global_idx < MAX_VALUE) {
            temp[shared_idx] = data[global_idx];
        } else {
            temp[shared_idx] = 0;
        }
        
        __syncthreads();

        int offset = 1;
        for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
            if (thread_id < d) {
                int ai = offset * (2 * thread_id + 1) - 1;
                int bi = offset * (2 * thread_id + 2) - 1;
                ai = conflict_free_index(ai);
                bi = conflict_free_index(bi);
                temp[bi] += temp[ai];
            }
            offset <<= 1;
            __syncthreads();
        }

        if (thread_id == 0) {
            int last_idx = conflict_free_index(blockDim.x - 1);
            if (block_offset < n_blocks) {
                block_sums[block_offset] = temp[last_idx];
                temp[last_idx] = 0;
            }
        }
        
        __syncthreads();

        for (int d = 1; d < blockDim.x; d <<= 1) {
            offset >>= 1;
            __syncthreads();
            
            if (thread_id < d) {
                int ai = offset * (2 * thread_id + 1) - 1;
                int bi = offset * (2 * thread_id + 2) - 1;
                ai = conflict_free_index(ai);
                bi = conflict_free_index(bi);
                
                int t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        
        __syncthreads();

        if (global_idx < MAX_VALUE) {
            data[global_idx] = temp[shared_idx];
        }
        
        __syncthreads();
    }
}

__global__ void add_block_sums_kernel(int *data, int *block_sums, int n_blocks) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    for (int block_offset = block_id; block_offset < n_blocks; block_offset += gridDim.x) {
        if (block_offset == 0) continue;

        int global_idx = block_offset * blockDim.x + thread_id;
        
        if (global_idx < MAX_VALUE) {
            data[global_idx] += block_sums[block_offset];
        }
    }
}

__global__ void counting_sort_kernel(int *arr, int *hist, int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += offset) {
        int value = arr[i];
        int position = atomicAdd(&hist[value], 1);
        out[position] = value;
    }
}

void recursive_scan(int *dev_hist, int n) {
    if (n <= 0) return;
    
    int padded_n = n;
    if (padded_n % BLOCK_SIZE != 0) {
        padded_n = ((n / BLOCK_SIZE) + 1) * BLOCK_SIZE;
    }
    
    int n_blocks = padded_n / BLOCK_SIZE;
    
    int *dev_block_sums;
    CSC(cudaMalloc(&dev_block_sums, n_blocks * sizeof(int)));
    
    int shared_mem_size = sizeof(int) * conflict_free_index(BLOCK_SIZE);
    
    scan_kernel<<<GRID_SIZE, BLOCK_SIZE, shared_mem_size>>>(dev_hist, dev_block_sums, n_blocks);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());
    
    if (n_blocks > 1) {
        recursive_scan(dev_block_sums, n_blocks);
        
        add_block_sums_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_hist, dev_block_sums, n_blocks);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());
    }
    
    CSC(cudaFree(dev_block_sums));
}

int main() {
    int n;
    int *h_arr;
    read_input(&n, &h_arr);
    
    int *dev_arr, *dev_hist, *dev_out;
    CSC(cudaMalloc(&dev_arr, n * sizeof(int)));
    CSC(cudaMalloc(&dev_hist, MAX_VALUE * sizeof(int)));
    CSC(cudaMalloc(&dev_out, n * sizeof(int)));

    CSC(cudaMemcpy(dev_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice));
    
    CSC(cudaMemset(dev_hist, 0, MAX_VALUE * sizeof(int)));

    histogram_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_arr, dev_hist, n);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    recursive_scan(dev_hist, MAX_VALUE);

    counting_sort_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(dev_arr, dev_hist, dev_out, n);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());
    
    int *h_out = (int*)malloc(n * sizeof(int));
    if (h_out == NULL) {
        fprintf(stderr, "Memory allocation failed for output\n");
        free(h_arr);
        CSC(cudaFree(dev_arr));
        CSC(cudaFree(dev_hist));
        CSC(cudaFree(dev_out));
        exit(0);
    }
    
    CSC(cudaMemcpy(h_out, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    write_output(h_out, n);

    free(h_arr);
    free(h_out);
    CSC(cudaFree(dev_arr));
    CSC(cudaFree(dev_hist));
    CSC(cudaFree(dev_out));
    
    return 0;
}