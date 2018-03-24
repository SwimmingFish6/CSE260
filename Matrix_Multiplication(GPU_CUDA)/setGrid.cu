#include <math.h>

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   // set your block dimensions and grid dimensions here
  gridDim.x = ceil(((double)n / blockDim.x)/2);
  gridDim.y = ceil(((double)n / blockDim.y)/2);
//   gridDim.x = ceil((double)n / blockDim.x);
//   gridDim.y = ceil((double)n / blockDim.y);
//   if(n % blockDim.x != 0)
//   	gridDim.x++;
//   if(n % blockDim.y != 0)
//    	gridDim.y++;
}
