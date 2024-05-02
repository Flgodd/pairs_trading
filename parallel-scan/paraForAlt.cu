#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <cmath>

#include <stdlib.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


const int N = 8;
const int BLOCK_SIZE = 512;


__global__ void pairs_trading_kernel(const double* stock1_prices, const double* stock2_prices, int* check, int size) {
    __shared__ double spread[1256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    /*for (int i = idx; i < 1256; i += stride) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }

    __syncthreads();*/

    for (int i = idx + N; i < size; i += stride) {
        double sum = 0.0;
        double sq_sum = 0.0;

        int start = i - N;

        /*double2 simd_sum = make_double2(0.0, 0.0);
        double2 simd_sq_sum = make_double2(0.0, 0.0);

        for (int j = 0; j < N; j += 2) {
            double2 prices = make_double2(
                    stock1_prices[start + j] - stock2_prices[start + j],
                    stock1_prices[start + j + 1] - stock2_prices[start + j + 1]
            );

            simd_sum.x += prices.x;
            simd_sum.y += prices.y;
            simd_sq_sum.x += prices.x * prices.x;
            simd_sq_sum.y += prices.y * prices.y;
        }

        sum = simd_sum.x + simd_sum.y;
        sq_sum = simd_sq_sum.x + simd_sq_sum.y;*/
//#pragma unroll
        for (int j = 0; j < N; j++) {
            double val = stock1_prices[start + j] - stock2_prices[start+j];
            sum += val;
            sq_sum += val * val;
        }

        double mean = sum / N;
        double stddev = sqrt(sq_sum / N - mean * mean);
        double current_spread = stock1_prices[i] - stock2_prices[i];
        double z_score = (current_spread - mean) / stddev;

        if (z_score > 1.0) {
            //atomicAdd(&check[0], 1);  // Long and Short
        } else if (z_score < -1.0) {
            //atomicAdd(&check[1], 1);  // Short and Long
        } else if (fabs(z_score) < 0.8) {
            //atomicAdd(&check[2], 1);  // Close positions
        } else {
            //atomicAdd(&check[3], 1);  // No signal
        }
    }
}

void pairs_trading_strategy_cuda(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices) {
    int size = stock1_prices.size();

    double* d_stock1_prices;
    double* d_stock2_prices;
    int* d_check;

    cudaMalloc(&d_stock1_prices, size * sizeof(double));
    cudaMalloc(&d_stock2_prices, size * sizeof(double));
    cudaMalloc(&d_check, 4 * sizeof(int));

    cudaMemcpy(d_stock1_prices, stock1_prices.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stock2_prices, stock2_prices.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_check, 0, 4 * sizeof(int));

    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    pairs_trading_kernel<<<grid_size, BLOCK_SIZE>>>(d_stock1_prices, d_stock2_prices, d_check, size);

    cudaDeviceSynchronize();

    //std::vector<int> check(4);
    //cudaMemcpy(check.data(), d_check, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    //std::cout << check[0] << ":" << check[1] << ":" << check[2] << ":" << check[3] << std::endl;

    cudaFree(d_stock1_prices);
    cudaFree(d_stock2_prices);
    cudaFree(d_check);
}