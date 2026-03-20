#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

#define BLOCK_SIZE 32
#define GRID_SIZE 64
#define EPS 1e-10

struct SCompare {
    __host__ __device__ bool operator()(const double a, const double b) const {
        return fabs(a) < fabs(b);
    }
};

__global__ void swap_row_kernel(double *matrix, double *identity, int n, int row1, int row2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int col = idx; col < n; col += offset) {
        double temp1 = matrix[col * n + row1];
        double temp2 = identity[col * n + row1];
        
        matrix[col * n + row1] = matrix[col * n + row2];
        matrix[col * n + row2] = temp1;
        
        identity[col * n + row1] = identity[col * n + row2];
        identity[col * n + row2] = temp2;
    }
}

__global__ void normalize_row_kernel(double *matrix, double *identity, int n, int pivot_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    double pivot = matrix[pivot_row * n + pivot_row];
    
    for (int col = idx; col < n; col += offset) {
        matrix[col * n + pivot_row] /= pivot;
        identity[col * n + pivot_row] /= pivot;
    }
}

__global__ void forward_kernel(double *matrix, double *identity, int n, int pivot_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int row = pivot_row + 1 + idx; row < n; row += offset_x) {
        double factor = matrix[pivot_row * n + row] / matrix[pivot_row * n + pivot_row];
        
        for (int col = pivot_row + 1 + idy; col < n; col += offset_y) {
            matrix[col * n + row] -= factor * matrix[col * n + pivot_row];
        }
        
        for (int col = idy; col < n; col += offset_y) {
            identity[col * n + row] -= factor * identity[col * n + pivot_row];
        }
    }
}

__global__ void backward_kernel(double *matrix, double *identity, int n, int pivot_row) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for (int row = pivot_row - 1 - idx; row >= 0; row -= offset_x) {
        double factor = matrix[pivot_row * n + row] / matrix[pivot_row * n + pivot_row];

        for (int col = idy; col < n; col += offset_y) {
                identity[col * n + row] -= factor * identity[col * n + pivot_row];
        }
    }
}

int main() {
    int n;
    scanf("%d", &n);

    double *matrix = (double*)malloc(n * n * sizeof(double));
    double *identity_matrix = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &matrix[j * n + i]);
            identity_matrix[j * n + i] = (i == j) ? 1.0 : 0.0;  
        }
    }

    double *dev_mat;
    double *dev_iden;
    CSC(cudaMalloc(&dev_mat, sizeof(double) * n * n));
    CSC(cudaMemcpy(dev_mat, matrix, sizeof(double) * n * n, cudaMemcpyHostToDevice));
    CSC(cudaMalloc(&dev_iden, sizeof(double) * n * n));
    CSC(cudaMemcpy(dev_iden, identity_matrix, sizeof(double) * n * n, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(GRID_SIZE, GRID_SIZE);

    SCompare comp;

    for (int i = 0; i < n; i++) {
        thrust::device_ptr<double> col_ptr = thrust::device_pointer_cast(dev_mat + i * n + i);
        thrust::device_ptr<double> max_ptr = thrust::max_element(col_ptr, col_ptr + (n - i), comp);
        int pivot_row = (max_ptr - col_ptr) + i;

        if (fabs(*max_ptr) < EPS) {
            printf("Matrix is singular or nearly singular\n");
            break;
        }

        if (pivot_row != i) {
            swap_row_kernel<<<1024, 256>>>(dev_mat, dev_iden, n, i, pivot_row);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());
        }

        normalize_row_kernel<<<1024, 256>>>(dev_mat, dev_iden, n, i);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());

        if (i < n - 1) {
            forward_kernel<<<gridSize, blockSize>>>(dev_mat, dev_iden, n, i);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());
        }

    }

    for (int i = n - 1; i > 0; i--) {
        backward_kernel<<<gridSize, blockSize>>>(dev_mat, dev_iden, n, i);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());
    }

    CSC(cudaMemcpy(identity_matrix, dev_iden, sizeof(double) * n * n, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.10lf ", identity_matrix[j * n + i]);
        }
        printf("\n");
    }

    free(matrix);
    free(identity_matrix);
    CSC(cudaFree(dev_mat));
    CSC(cudaFree(dev_iden));

    return 0; 
}