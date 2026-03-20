#ifndef __SSAA_CUH__
#define __SSAA_CUH__

#include "structures.cuh"
#include "camera.cuh"
#include <cuda_runtime.h>

#define BLOCKS_2D dim3(32, 32)
#define THREADS_2D dim3(16, 16)


__host__ void cpu_ssaa(const uchar4 *data_in, uchar4 *data_out, int width, int height, int sample_square) {

    const double inv_sample_square = 1.0 / (sample_square * sample_square);
    const int wc = width * sample_square;

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            double r = 0, g = 0, b = 0;
            int baseIdx = (sample_square * j) * wc + sample_square * i;

            for (int ki = 0; ki < sample_square; ++ki) {
                for (int kj = 0; kj < sample_square; ++kj) {
                    uchar4 sample_pixel = data_in[baseIdx + kj * wc + ki];
                    r += sample_pixel.x;
                    g += sample_pixel.y;
                    b += sample_pixel.z;
                }
            }

            data_out[j * width + i] = make_uchar4(
                (unsigned char)(r * inv_sample_square), 
                (unsigned char)(g * inv_sample_square), 
                (unsigned char)(b * inv_sample_square), 
                255);
        }
    }
}

__global__ void gpu_ssaa(const uchar4 *data_in, uchar4 *data_out, int width, int height, int sample_square) {

    const double inv_sample_square = 1.0 / (sample_square * sample_square);
    const int wc = width * sample_square;

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int idy = blockDim.y * blockIdx.y + threadIdx.y;
    const int offsetx = blockDim.x * gridDim.x;
    const int offsety = blockDim.y * gridDim.y;

    for (int i = idx; i < width; i += offsetx) {
        for (int j = idy; j < height; j += offsety) {
            double r = 0, g = 0, b = 0;
            int baseIdx = (sample_square * j) * wc + sample_square * i;

            for (int ki = 0; ki < sample_square; ++ki) {
                for (int kj = 0; kj < sample_square; ++kj) {
                    uchar4 sample_pixel = data_in[baseIdx + kj * wc + ki];
                    r += sample_pixel.x;
                    g += sample_pixel.y;
                    b += sample_pixel.z;
                }
            }

            data_out[j * width + i] = make_uchar4(
                (unsigned char)(r * inv_sample_square), 
                (unsigned char)(g * inv_sample_square), 
                (unsigned char)(b * inv_sample_square), 
                255);
        }
    }
}

__host__ void applySSAA_CPU(const uchar4* high_res, uchar4* low_res, int width, int height, int ssaa_sample_square) {
    cpu_ssaa(high_res, low_res, width, height, ssaa_sample_square);
}

__host__ void applySSAA_GPU(const uchar4* d_high_res, uchar4* d_low_res, int width, int height, int ssaa_sample_square) {    
    gpu_ssaa<<<BLOCKS_2D, THREADS_2D>>>(d_high_res, d_low_res, width, height, ssaa_sample_square);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());
}

#endif