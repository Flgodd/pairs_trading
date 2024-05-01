#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

#define NUM_EVENTS 1
#define SIZE 1024
#define REPS 10000

int main() {
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

    // Report results
    printf("L1 Data Cache Misses: %lld\n", values[0]);

    // Cleanup
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    return 0;
}