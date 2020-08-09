#include "mwc64x_rng.cl"

__kernel void piMC(uint seedOffset, uint totalSamples, __global int *acc) {
    mwc64x_state_t rng;
    ulong samplesPerStream = totalSamples / get_global_size(0);
    MWC64X_SeedStreams(&rng, seedOffset, 2*samplesPerStream);
    uint count=0;
    for(uint i=0;i<samplesPerStream;i++){
        ulong x=MWC64X_NextUint(&rng);
        ulong y=MWC64X_NextUint(&rng);
        ulong x2=x*x;
        ulong y2=y*y;
        if(x2+y2 >= x2)
            count++;
    }
    atomic_add(acc, count);
}