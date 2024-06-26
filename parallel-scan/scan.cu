#include <stdlib.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernels.cuh"
#include "utils.h"
#include "scan.cuh"

#define checkCudaError(o, l) _checkCudaError(o, l, __func__)

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

using namespace std;


void scanLargeDeviceArray(double *d_out, double *d_in, int length, bool bcao) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		double *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, bcao);

		add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

void scanSmallDeviceArray(double *d_out, double *d_in, int length, bool bcao) {
	int powerOfTwo = nextPowerOfTwo(length);

	if (bcao) {
		prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(double) >> >(d_out, d_in, length, powerOfTwo);
	}
	else {
		prescan_arbitrary_unoptimized<< <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(double) >> >(d_out, d_in, length, powerOfTwo);
	}
}

void scanLargeEvenDeviceArray(double *d_out, double *d_in, int length, bool bcao) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const double sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(double);

	double *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(double));
	cudaMalloc((void **)&d_incr, blocks * sizeof(double));

	if (bcao) {
		prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}
	else {
		prescan_large_unoptimized<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}

__global__ void parallelized_zscore_calculation(
        const double *stock1_prices,
        const double *stock2_prices,
        const double *spread_sum,
        const double *spread_sq_sum,
        int *check,
        int N,
        size_t size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size - N) return;

    int i = N + idx;

    const double mean = (spread_sum[i] - spread_sum[i-N])/ N;
    const double stddev = std::sqrt((spread_sq_sum[i] - spread_sq_sum[i-N])/ N - mean * mean);
    const double current_spread = stock1_prices[i] - stock2_prices[i];
    const double z_score = (current_spread - mean) / stddev;

    if (z_score > 1.0) {
        //atomicAdd(&check[0], 1); // Long and Short
    } else if (z_score < -1.0) {
        //atomicAdd(&check[1], 1); // Short and Long
    } else if (std::abs(z_score) < 0.8) {
        //atomicAdd(&check[2], 1);  // Close positions
    } else {
        //atomicAdd(&check[3], 1);  // No signal
    }
}

void calc_zz(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices,
            double spread_sum[], double spread_sq_sum[],
            std::vector<int>& check, size_t spread_size) {
    const int N = 8;
    double *d_stock1_prices, *d_stock2_prices, *d_spread_sum, *d_spread_sq_sum;
    int *d_check;

    cudaMalloc((void**)&d_stock1_prices, stock1_prices.size() * sizeof(double));
    cudaMalloc((void**)&d_stock2_prices, stock2_prices.size() * sizeof(double));
    cudaMalloc((void**)&d_spread_sum, spread_size * sizeof(double));
    cudaMalloc((void**)&d_spread_sq_sum, spread_size * sizeof(double));
    //cudaMalloc((void**)&d_check, check.size() * sizeof(int)); // Assuming 'check' has size 4

// Data Transfer to the GPU
    cudaMemcpy(d_stock1_prices, stock1_prices.data(), stock1_prices.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stock2_prices, stock2_prices.data(), stock2_prices.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spread_sum, spread_sum, spread_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spread_sq_sum, spread_sq_sum, spread_size * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 512;

    int numBlocks = (stock1_prices.size() - N - 1 + threadsPerBlock - 1) / threadsPerBlock;
    //printf("numBlocks:%d\n", numBlocks);

    parallelized_zscore_calculation<<<numBlocks, threadsPerBlock >>>(d_stock1_prices, d_stock2_prices, d_spread_sum, d_spread_sq_sum, d_check, N, stock1_prices.size());

// Copy results back
    //cudaMemcpy(check.data(), d_check, check.size() * sizeof(int), cudaMemcpyDeviceToHost);

// Print results
    //std::cout<<check[0]<<":"<<check[1]<<":"<<check[2]<<":"<<check[3]<<std::endl;
    //printf("d_check[0]:%d || d_check[1]:%d || d_check[2]:%d || d_check[3]:%d \n", check[0], check[1], check[2], check[3]);
    cudaFree(d_stock1_prices);
    cudaFree(d_stock2_prices);
    cudaFree(d_spread_sum);
    cudaFree(d_spread_sq_sum);
    //cudaFree(d_check);
}

__global__ void para_fill(const double *stock1_prices,
                          const double *stock2_prices,
                          double *spread_sum,
                          double *spread_sq_sum,
                          size_t size){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)return;
    const double current_spread = stock1_prices[idx] - stock2_prices[idx];
    spread_sum[idx] = current_spread;
    spread_sq_sum[idx] = current_spread * current_spread;
}


void fillArrays(const std::vector<double>& stock1_prices, const std::vector<double>& stock2_prices,
                             double spread_sum[], double spread_sq_sum[], size_t spread_size){
    vector<int> check (4);
    vector<double> temp(spread_size);
    double *d_stock1_prices, *d_stock2_prices, *d_spread_sum, *d_spread_sq_sum;
    int *d_check;
    int length = spread_size;
    cudaMalloc((void**)&d_stock1_prices, stock1_prices.size() * sizeof(double));
    cudaMalloc((void**)&d_stock2_prices, stock2_prices.size() * sizeof(double));
    cudaMalloc((void**)&d_spread_sum, spread_size * sizeof(double));
    cudaMalloc((void**)&d_spread_sq_sum, spread_size * sizeof(double));

    cudaMemcpy(d_stock1_prices, stock1_prices.data(), stock1_prices.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stock2_prices, stock2_prices.data(), stock2_prices.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spread_sum, spread_sum, spread_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spread_sq_sum, spread_sq_sum, spread_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_check, 4 * sizeof(int));
    int threadsPerBlock = 512;
    bool bcao = true;
    int numBlocks = (stock1_prices.size() + threadsPerBlock - 1) / threadsPerBlock;

    para_fill<<<numBlocks, threadsPerBlock >>>(d_stock1_prices, d_stock2_prices, d_spread_sum, d_spread_sq_sum, spread_size);
    //cudaMemcpy(spread_sum, d_spread_sum, spread_size * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(spread_sq_sum, d_spread_sq_sum, spread_size * sizeof(double), cudaMemcpyDeviceToHost);
    //printf("%f : %f\n" , spread_sum[0], stock1_prices[0] - stock2_prices[0]);
    double *d_out, *d_out2;
    const int arraySize = length * sizeof(double);
//
    cudaMalloc((void **)&d_out, arraySize);
    cudaMalloc((void **)&d_out2, arraySize);

    if (length > ELEMENTS_PER_BLOCK) {
        scanLargeDeviceArray(d_out, d_spread_sum, length, bcao);
        scanLargeDeviceArray(d_out2, d_spread_sq_sum, length, bcao);
    }
    else {
        scanSmallDeviceArray(d_out, d_spread_sum, length, bcao);
        scanSmallDeviceArray(d_out2, d_spread_sq_sum, length, bcao);
    }
    //cudaMemcpy(temp.data(), d_out, arraySize, cudaMemcpyDeviceToHost);


    int N =8;

     numBlocks = (stock1_prices.size() - N - 1 + threadsPerBlock - 1) / threadsPerBlock;
    parallelized_zscore_calculation<<<numBlocks, threadsPerBlock >>>(d_stock1_prices, d_stock2_prices, d_out, d_out2, d_check, N, spread_size);
    //cudaMemcpy(check.data(), d_check, check.size() * sizeof(int), cudaMemcpyDeviceToHost);
    //printf("d_check[0]:%d || d_check[1]:%d || d_check[2]:%d || d_check[3]:%d \n", check[0], check[1], check[2], check[3]);

    //cudaMemcpy(spread_sum, d_spread_sum, spread_size * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(spread_sq_sum, d_spread_sq_sum, spread_size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_check);
    cudaFree(d_out);
    cudaFree(d_out2);

    cudaFree(d_stock1_prices);
    cudaFree(d_stock2_prices);
    cudaFree(d_spread_sum);
    cudaFree(d_spread_sq_sum);
}
