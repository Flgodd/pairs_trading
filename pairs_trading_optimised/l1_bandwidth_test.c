#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define CACHE_LINE_SIZE 64 // Typical L1 cache line size in bytes
#define L1_CACHE_SIZE (32 * 1024) // Adjust the size to your CPU's L1 cache size
#define NUM_ITERATIONS (L1_CACHE_SIZE / CACHE_LINE_SIZE * 1024)

int main() {
    char array[L1_CACHE_SIZE];
    clock_t start, end;

    // Warm-up the cache
    for (int i = 0; i < L1_CACHE_SIZE; i++) {
        array[i] = i;
    }

    start = clock();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        for (int j = 0; j < L1_CACHE_SIZE; j += CACHE_LINE_SIZE) {
            array[j] += 1; // Simple operation to ensure memory access
        }
    }
    end = clock();

    double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC; // in seconds
    double bytes_accessed = (double) NUM_ITERATIONS * L1_CACHE_SIZE;
    double bandwidth = bytes_accessed / time_taken / (1024 * 1024); // MB/s

    printf("Estimated L1 Cache Bandwidth: %f MB/s\n", bandwidth);
    return 0;
}