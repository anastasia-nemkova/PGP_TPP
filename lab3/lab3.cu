#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

#define MAX_CLASSES 32

__constant__ float const_avg[MAX_CLASSES * 3];
__constant__ float const_cov_inv[MAX_CLASSES * 9];

__global__ void kernel(uchar4* data, int w, int h, int nc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = gridDim.x * blockDim.x;

    for (int pos = idx; pos < w * h; pos += offsetx) {
        uchar4 pixel = data[pos];
        
        float best_value = -FLT_MAX;
        unsigned char best_class = 0;
        float p[3] = {(float)pixel.x, (float)pixel.y, (float)pixel.z};

        for(unsigned char j = 0; j < nc; j++) {
            const float* avgj = &const_avg[j * 3];
            const float* cov_invj = &const_cov_inv[j * 9];

            float diff[3] = {p[0] - avgj[0], p[1] - avgj[1], p[2] - avgj[2]};

            float tmp[3] = {0, 0, 0};
            for (int i = 0; i < 3; i++) {
                for (int k = 0; k < 3; k++) {
                    tmp[i] += diff[k] * cov_invj[k * 3 + i];
                }
            }

            float res = 0;
            for (int i = 0; i < 3; i++) {
                res += tmp[i] * diff[i];
            }

            if (-res > best_value || (-res == best_value && j < best_class)) {
                best_value = -res;
                best_class = j;
            }
        }
        data[pos].w = best_class;
    }
}

float det_matrix(const float* mat) {
    return mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7] - mat[2] * mat[4] * mat[6] - mat[0] * mat[5] * mat[7] - mat[1] * mat[3] * mat[8];
}

void inverse_matrix(const float* mat, float* inv) {
    float det = det_matrix(mat);

    inv[0] = (mat[4] * mat[8] - mat[5] * mat[7]) / det;
    inv[1] = (mat[7] * mat[2] - mat[1] * mat[8]) / det;
    inv[2] = (mat[1] * mat[5] - mat[4] * mat[2]) / det;
    inv[3] = (mat[6] * mat[5] - mat[3] * mat[8]) / det;
    inv[4] = (mat[0] * mat[8] - mat[6] * mat[2]) / det;
    inv[5] = (mat[3] * mat[2] - mat[0] * mat[5]) / det;
    inv[6] = (mat[3] * mat[7] - mat[6] * mat[4]) / det;
    inv[7] = (mat[6] * mat[1] - mat[0] * mat[7]) / det;
    inv[8] = (mat[0] * mat[4] - mat[3] * mat[1]) / det;
}

void avg_vector(float **coords, int npj, float *avg) {
    float sum[3] = {0, 0, 0};

    for (int i = 0; i < npj; i++) {
        sum[0] += coords[i][0];
        sum[1] += coords[i][1];
        sum[2] += coords[i][2];
    }

    avg[0] = sum[0] / npj;
    avg[1] = sum[1] / npj;
    avg[2] = sum[2] / npj;
}

void cov_matrix(float **coords, int npj, const float *avg, float *cov) {
    for (int i = 0; i < 9; i++) {
        cov[i] = 0.0f;
    }

    for (int i = 0; i < npj; i++) {
        float diff[3] = {coords[i][0] - avg[0], coords[i][1] - avg[1], coords[i][2] - avg[2]};

        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                cov[row * 3 + col] += diff[row] * diff[col];
            }
        }
    }

    if (npj > 1) {
        for (int i = 0; i < 9; i++) {
            cov[i] /= (npj - 1);
        }
    }
}

int main() {
    char input_path[PATH_MAX];
    char output_path[PATH_MAX];

    if (scanf("%s", input_path) != 1) {
        fprintf(stderr, "Error read input path\n");
        return 0;
    }

    if (scanf("%s", output_path) != 1) {
        fprintf(stderr, "Error read input path\n");
        return 0;
    }

    FILE *fp = fopen(input_path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open input file\n");
        return 0;
    }

    int w, h;
 	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);

    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(fp);
        return 0;
    }

    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    int nc;
    if (scanf("%d", &nc) != 1) {
        fprintf(stderr, "Error read number classes\n");
        free(data);
        return 0;
    }

    float *avg = (float*)malloc(nc * 3 * sizeof(float));
    float *cov = (float*)malloc(nc * 9 * sizeof(float));
    float *cov_inv = (float*)malloc(nc * 9 * sizeof(float));

    for(int i = 0; i < nc; i++) {
        int npj;
        scanf("%d", &npj);
        float **coords = (float**)malloc(npj * sizeof(float*));

        for(int j = 0; j < npj; j++) {
            int x, y;
            scanf("%d %d", &x, &y);

            if(x < 0 || x >= w || y < 0 || y >= h){
                fprintf(stderr, "Pixel coordinates out of bounds\n");
                free(avg);
                free(cov_inv);
                for (int k = 0; k < i; k++) free(coords[k]);
                free(coords);
                return 0;
            }

            coords[j] = (float*)malloc(3 * sizeof(float));
            uchar4 pixel = data[y * w + x];

            coords[j][0] = pixel.x;
            coords[j][1] = pixel.y; 
            coords[j][2] = pixel.z;
        }
        avg_vector(coords, npj, &avg[i * 3]);
        cov_matrix(coords, npj, &avg[i * 3], &cov[i * 9]);
        inverse_matrix(&cov[i * 9], &cov_inv[i * 9]);

        for (int j = 0; j < npj; j++) {
            free(coords[j]);
        }
        free(coords);
    }

    CSC(cudaMemcpyToSymbol(const_avg, avg, nc * 3 * sizeof(float), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(const_cov_inv, cov_inv, nc * 9 * sizeof(float), 0, cudaMemcpyHostToDevice));

    uchar4 *dev_data;
    CSC(cudaMalloc(&dev_data, w * h * sizeof(uchar4)));
    CSC(cudaMemcpy(dev_data, data, w * h * sizeof(uchar4), cudaMemcpyHostToDevice));

    kernel<<<1024, 256>>>(dev_data, w, h, nc);

    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(data, dev_data, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));

    fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open output file\n");
        free(data); free(avg); free(cov_inv);
        cudaFree(dev_data);
        return 0;
    }
    
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    free(avg);
    free(cov_inv);
    cudaFree(dev_data);

    return 0;
}