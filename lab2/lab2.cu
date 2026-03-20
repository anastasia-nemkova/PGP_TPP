#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

__device__ float getBright(uchar4 pixel) {
    return 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
}

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
   	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
    int x, y;

    for(y = idy; y < h; y += offsety)
		for(x = idx; x < w; x += offsetx) {
            uchar4 p00 = tex2D<uchar4>(tex, x, y);
            float w00 = getBright(p00);

            uchar4 p01 = tex2D<uchar4>(tex, x, y + 1);
            float w01 = getBright(p01);

            uchar4 p10 = tex2D<uchar4>(tex, x + 1, y);
            float w10 = getBright(p10);

            uchar4 p11 = tex2D<uchar4>(tex, x + 1, y + 1);
            float w11 = getBright(p11);

            float gx = w11 - w00;
            float gy = w10 - w01;

            float grad = sqrtf(gx * gx + gy * gy);

            int val = (int)fminf(grad, 255.0f);

            out[y * w + x] = make_uchar4(val, val, val, 255);
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

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    kernel<<< dim3(16, 16), dim3(32, 32) >>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	CSC(cudaDestroyTextureObject(tex));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(dev_out));

    fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open output file\n");
        return 0;
    }

	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

    free(data);
    return 0;
}