#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <cmath>

#include <stdlib.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>


const int N = 8;
const int BLOCK_SIZE = 256;

__global__ void pairs_trading_kernel(const double* stock1_prices, const double* stock2_prices, int* check, int size) {
    __shared__ double spread[1256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < 1256; i += stride) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }

    __syncthreads();

    for (int i = idx + N; i < size; i += stride) {
        double sum = 0.0;
        double sq_sum = 0.0;

        int start = i - N;

        double4 vals1 = reinterpret_cast<const double4*>(spread + start)[0];
        double4 vals2 = reinterpret_cast<const double4*>(spread + start + 4)[0];

        sum = vals1.x + vals1.y + vals1.z + vals1.w + vals2.x + vals2.y + vals2.z + vals2.w;
        sq_sum = vals1.x * vals1.x + vals1.y * vals1.y + vals1.z * vals1.z + vals1.w * vals1.w +
                 vals2.x * vals2.x + vals2.y * vals2.y + vals2.z * vals2.z + vals2.w * vals2.w;

        double mean = sum / N;
        double stddev = sqrt(sq_sum / N - mean * mean);
        double current_spread = spread[i];
        double z_score = (current_spread - mean) / stddev;

        if (z_score > 1.0) {
            atomicAdd(&check[0], 1);  // Long and Short
        } else if (z_score < -1.0) {
            atomicAdd(&check[1], 1);  // Short and Long
        } else if (fabs(z_score) < 0.8) {
            atomicAdd(&check[2], 1);  // Close positions
        } else {
            atomicAdd(&check[3], 1);  // No signal
        }
    }
}

void pairs_trading_strategy_cuda(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    int size = stock1_prices.size();

    double* d_stock1_prices;
    double* d_stock2_prices;
    int* d_check;

    size_t pitch;
    cudaMallocPitch(&d_stock1_prices, &pitch, size * sizeof(double), 1);
    cudaMallocPitch(&d_stock2_prices, &pitch, size * sizeof(double), 1);
    cudaMalloc(&d_check, 4 * sizeof(int));

    cudaMemcpy2D(d_stock1_prices, pitch, stock1_prices.data(), size * sizeof(double), size * sizeof(double), 1, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_stock2_prices, pitch, stock2_prices.data(), size * sizeof(double), size * sizeof(double), 1, cudaMemcpyHostToDevice);
    cudaMemset(d_check, 0, 4 * sizeof(int));

    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    pairs_trading_kernel<<<grid_size, BLOCK_SIZE>>>(d_stock1_prices, d_stock2_prices, d_check, size);

    cudaDeviceSynchronize();

    std::vector<int> check(4);
    cudaMemcpy(check.data(), d_check, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << check[0] << ":" << check[1] << ":" << check[2] << ":" << check[3] << std::endl;

    cudaFree(d_stock1_prices);
    cudaFree(d_stock2_prices);
    cudaFree(d_check);
}