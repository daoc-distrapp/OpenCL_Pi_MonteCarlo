#include "mwc64x_rng.cl"

__kernel void piMC(uint seedRandom, uint totalSamples, __global ulong *acc) {
	uint counter = 0;
    mwc64x_state_t rng;
    ulong samplesPerStream = totalSamples / get_global_size(0);
    MWC64X_SeedStreams(&rng, seedRandom, 2*samplesPerStream);
	
    for(uint i=0;i<samplesPerStream;i++){
        float x=MWC64X_NextUint(&rng)/(float)UINT_MAX;
        float y=MWC64X_NextUint(&rng)/(float)UINT_MAX;
        float x2=x*x;
        float y2=y*y;
        if(x2+y2 <= 1.0)
            counter++;
    }
	
	acc[get_global_id(0)] = counter;
}
