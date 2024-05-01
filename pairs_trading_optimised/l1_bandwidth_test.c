#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define CACHE_LINE_SIZE 64 // Cache line size in bytes
#define L1_CACHE_SIZE (2 * 1024 * 1024) // Total L1 cache size (2 MiB)
#define NUM_ITERATIONS 1024

int main() {
    char *array = malloc(L1_CACHE_SIZE); // Allocate memory based on L1 cache size
    if (array == NULL) {
        perror("Failed to allocate memory");
        return 1;
    }

    clock_t start, end;

    // Warm-up the cache
    for (int i = 0; i < L1_CACHE_SIZE; i++) {
        array[i] = i % 256;
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
    double bandwidth = bytes_accessed / time_taken / (1024 * 1024); // bandwidth in MB/s

    printf("Estimated L1 Cache Bandwidth: %f MB/s\n", bandwidth);

    free(array);
    return 0;
}