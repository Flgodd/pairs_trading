#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

#define NUM_EVENTS 1
#define SIZE 1024
#define REPS 10000

int main() {
    struct timeval start_time, end_time;
    long long elapsed_time; // in microseconds
    int Events[NUM_EVENTS] = {PAPI_L1_DCM};  // Level 1 data cache misses
    long long values[NUM_EVENTS];
    int retval, EventSet = PAPI_NULL;

    // Initialize the PAPI library
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library init error!\n");
        return 1;
    }

    // Create the Event Set
    if (PAPI_create_eventset(&EventSet) != PAPI_OK) {
        fprintf(stderr, "PAPI create event set error!\n");
        return 1;
    }

    // Add L1 data cache miss event to the Event Set
    if (PAPI_add_event(EventSet, PAPI_L1_DCM) != PAPI_OK) {
        fprintf(stderr, "Error adding PAPI_L1_DCM\n");
        return 1;
    }

    // Start the timer
    gettimeofday(&start_time, NULL);

    // Start counting
    PAPI_start(EventSet);

    // Code to stress the L1 cache
    volatile int temp;
    for (int i = 0; i < REPS; i++) {
        for (int j = 0; j < SIZE; j++) {
            temp = j * i;  // Simple operation to ensure usage of L1 cache
        }
    }

    // Stop counting
    PAPI_stop(EventSet, values);

    // Stop the timer
    gettimeofday(&end_time, NULL);

    // Calculate elapsed time in microseconds
    elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000LL + (end_time.tv_usec - start_time.tv_usec);

    // Report results
    printf("L1 Data Cache Misses: %lld\n", values[0]);
    printf("Elapsed Time: %lld microseconds\n", elapsed_time);

    // Cleanup
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    // Calculate and print bandwidth
    double bandwidth = (values[0] * 64 / (elapsed_time / 1000000.0)) / (1024 * 1024); // bandwidth in MB/s
    printf("Estimated L1 Cache Bandwidth: %f MB/s\n", bandwidth);

    return 0;
}