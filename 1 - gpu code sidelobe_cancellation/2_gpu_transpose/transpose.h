
#include <cuda_runtime.h>
#include <cuda.h>


const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;


__global__ void transposeNaive(double *odata, const double *idata);


// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transposeCoalesced(double *odata, const double *idata);


// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(double *odata, const double *idata);
