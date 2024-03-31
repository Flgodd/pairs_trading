#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <cmath>

const int N = 1256;
const int BLOCK_SIZE = 256;

__global__ void pairs_trading_kernel(const double* stock1_prices, const double* stock2_prices, int* check, int size) {
    __shared__ double spread[N];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        spread[i] = stock1_prices[i] - stock2_prices[i];
    }

    __syncthreads();

    for (int i = idx + N; i < size; i += stride) {
        double sum = 0.0;
        double sq_sum = 0.0;

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            double val = spread[i - N + j];
            sum += val;
            sq_sum += val * val;
        }

        double mean = sum / N;
        double stddev = sqrt(sq_sum / N - mean * mean);
        double current_spread = stock1_prices[i] - stock2_prices[i];
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

    thrust::device_vector<double> d_stock1_prices = stock1_prices;
    thrust::device_vector<double> d_stock2_prices = stock2_prices;
    thrust::device_vector<int> d_check(4, 0);

    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    pairs_trading_kernel<<<grid_size, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(d_stock1_prices.data()),
            thrust::raw_pointer_cast(d_stock2_prices.data()),
            thrust::raw_pointer_cast(d_check.data()),
            size
    );

    cudaDeviceSynchronize();

    std::vector<int> check(4);
    thrust::copy(d_check.begin(), d_check.end(), check.begin());

    std::cout << check[0] << ":" << check[1] << ":" << check[2] << ":" << check[3] << std::endl;
}