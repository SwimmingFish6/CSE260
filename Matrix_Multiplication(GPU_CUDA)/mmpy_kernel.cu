// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;
#define BLOCK_SIZE 32
#define ROW_BLOCK_SIZE BLOCK_SIZE
#define COL_BLOCK_SIZE BLOCK_SIZE
#define MAX_SM_SIZE 32* BLOCK_SIZE
#define NUM_SIMULTANEOUS_C_ROW_ELEMENTS 2
#define NUM_SIMULTANEOUS_C_COL_ELEMENTS 2
#define NUM_SM_TILES 1
#define min(a,b) (((a)<(b))?(a):(b))
//128 KB shared memory/L1 cache
//13 (out of 15 actual) SMXs
//3.7 compute capability
//1024 threads/ block
//64 warps/ smx
//2048 threads/smx
//16 blocks/smx
//=>
//13 * 16 blocks, 1 grid

//__global__ void matMul(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
__global__ void matMulNaive(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    int row =  blockIdx.y*blockDim.y + threadIdx.y;
    int col =  blockIdx.x*blockDim.x + threadIdx.x;

    if((row < square_dim) && (col < square_dim)){ // if within matrix bounds
        _DOUBLE_ _c = 0;

        for (unsigned int k = 0; k < square_dim; ++k) {
            _DOUBLE_ a = A[row * square_dim + k];
            _DOUBLE_ b = B[k * square_dim + col];
            _c += a * b;//
        }

        C[row * square_dim + col] = _c;
    }

}

//__global__ void matMul(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
__global__ void matMul_UNROLL2(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {



    __shared__ _DOUBLE_ A_block[ROW_BLOCK_SIZE][COL_BLOCK_SIZE];
    __shared__ _DOUBLE_ B_block[COL_BLOCK_SIZE][ROW_BLOCK_SIZE];

    __shared__ _DOUBLE_ A_block_1[ROW_BLOCK_SIZE][COL_BLOCK_SIZE];
    __shared__ _DOUBLE_ B_block_1[COL_BLOCK_SIZE][ROW_BLOCK_SIZE];

    int block_row = threadIdx.y;
    int block_col = threadIdx.x;
  //deal with block size != tile size
  int block_x_offset = min(COL_BLOCK_SIZE, blockDim.x);
  int block_y_offset = min(ROW_BLOCK_SIZE, blockDim.y);

  int row =  blockIdx.y*block_y_offset + block_row;
  int col =  blockIdx.x*block_x_offset + block_col;
//    int index = row *square_dim + col;
//    int row = blockIdx.y * blockDim.y + block_col;
    _DOUBLE_ _c = 0;

    int col_offset;
    int row_offset;
    int A_index;
    int B_index;
    int num_unrolls = 2;
#pragma unroll
    for (unsigned int kk = 0; kk < gridDim.x; kk+=num_unrolls) {

        col_offset = (kk*block_x_offset + block_col);
        row_offset = (kk*block_y_offset + block_row );
        A_index = row * square_dim + col_offset;
        B_index = row_offset * square_dim + col;

        //load sub-blocks into shared memory: each thread does one load
        if (A_index < square_dim*square_dim) {
            A_block[block_row][block_col] = A[A_index];
        } else {// this thread walks off array
            A_block[block_row][block_col] = 0;
        }

        if (B_index < square_dim*square_dim) {
            B_block[block_row][block_col] = B[B_index];
        } else {// this thread walks off array
            B_block[block_row][block_col] = 0;
        }

        col_offset = ((kk+1)*block_x_offset + block_col);
        row_offset = ((kk+1)*block_y_offset + block_row );

        A_index = row * square_dim + col_offset;
        B_index = row_offset * square_dim + col;

        //load sub-blocks into shared memory: each thread does one load
        if (A_index < square_dim*square_dim) {
            A_block_1[block_row][block_col] = A[A_index];
        } else {// this thread walks off array
            A_block_1[block_row][block_col] = 0;
        }

        if (B_index < square_dim*square_dim) {
            B_block_1[block_row][block_col] = B[B_index];
        } else {// this thread walks off array
            B_block_1[block_row][block_col] = 0;
        }
        __syncthreads();


        //each thread within the block dim loops over rows to add results
        //returning single element in C
#pragma unroll
        for (unsigned int k = 0; k < block_x_offset; ++k) {
            _c += A_block[block_row][k] *  B_block[k][block_col];//
            _c += A_block_1[block_row][k] *  B_block_1[k][block_col];//
        }
        __syncthreads();


    }

    if((row < square_dim) && (col < square_dim)) { // if within matrix bounds
        C[row * square_dim + col] = _c;
    }
}

//__global__ void matMul(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
__global__ void matMul_no_unroll(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    __shared__ _DOUBLE_ A_block[ROW_BLOCK_SIZE][COL_BLOCK_SIZE];
    __shared__ _DOUBLE_ B_block[COL_BLOCK_SIZE][ROW_BLOCK_SIZE];
    int block_row = threadIdx.y;
    int block_col = threadIdx.x;

    //deal with block size != tile size
    int block_x_offset = min(COL_BLOCK_SIZE, blockDim.x);
    int block_y_offset = min(ROW_BLOCK_SIZE, blockDim.y);

    int row =  blockIdx.y*block_y_offset + block_row;
    int col =  blockIdx.x*block_x_offset + block_col;


//    int index = row *square_dim + col;
//    int row = blockIdx.y * blockDim.y + block_col;
    _DOUBLE_ _c = 0;
    int col_offset;
    int row_offset;
    int A_index;
    int B_index;
#pragma unroll
    for (unsigned int kk = 0; kk < gridDim.x; ++kk) {

        col_offset = (kk*block_x_offset + block_col);
        row_offset = (kk*block_y_offset + block_row );
        A_index = row * square_dim + col_offset;
        B_index = row_offset * square_dim + col;

        //load sub-blocks into shared memory: each thread does one load
        if (A_index < square_dim*square_dim) {
            A_block[block_row][block_col] = A[A_index];
        } else {// this thread walks off array
            A_block[block_row][block_col] = 0;
        }

        if (B_index < square_dim*square_dim) {
            B_block[block_row][block_col] = B[B_index];
        } else {// this thread walks off array
            B_block[block_row][block_col] = 0;
        }
        __syncthreads();


        //each thread within the block dim loops over rows to add results
        //returning single element in C
#pragma unroll
        for (unsigned int k = 0; k < block_x_offset; ++k) {
            _c += A_block[block_row][k] *  B_block[k][block_col];//
        }
        __syncthreads();

    }


    if((row < square_dim) && (col < square_dim)) { // if within matrix bounds
        C[row * square_dim + col] = _c;
    }
}

//__global__ void matMul(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
__global__ void matMul_naiveSM(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    __shared__ _DOUBLE_ A_block[ROW_BLOCK_SIZE][COL_BLOCK_SIZE];
    __shared__ _DOUBLE_ B_block[COL_BLOCK_SIZE][ROW_BLOCK_SIZE];
    int block_row = threadIdx.y;
    int block_col = threadIdx.x;
    int row =  blockIdx.y*ROW_BLOCK_SIZE + block_row;
    int col =  blockIdx.x*COL_BLOCK_SIZE + block_col;

    _DOUBLE_ _c = 0;
    int col_offset;
    int row_offset;
    int A_index;
    int B_index;
#pragma unroll
    for (unsigned int kk = 0; kk < gridDim.x; ++kk) {

        col_offset = (kk*COL_BLOCK_SIZE + block_col);
        row_offset = (kk*ROW_BLOCK_SIZE + block_row );

        A_index = row * square_dim + col_offset;
        B_index = row_offset * square_dim + col;

        //load sub-blocks into shared memory: each thread does one load
        if (A_index < square_dim*square_dim) {
            A_block[block_row][block_col] = A[A_index];
        } else {// this thread walks off array
            A_block[block_row][block_col] = 0;
        }

        if (B_index < square_dim*square_dim) {
            B_block[block_col][block_row] = B[B_index];
        } else {// this thread walks off array
            B_block[block_col][block_row] = 0;
        }
        __syncthreads();


        //each thread within the block dim loops over rows to add results
        //returning single element in C
#pragma unroll
        for (unsigned int k = 0; k < COL_BLOCK_SIZE; ++k) {
            _c += A_block[block_row][k] *  B_block[block_col][k];//
        }
        __syncthreads();

    }


    if((row < square_dim) && (col < square_dim)) { // if within matrix bounds
        C[row * square_dim + col] = _c;
    }
}

__global__ void matMul(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
    int num_elements = square_dim*square_dim;
    int block_row = threadIdx.y;
    int block_col = threadIdx.x;
    
    //since we are operating on adjacent squares
    int block_index_y = (blockIdx.y*NUM_SIMULTANEOUS_C_COL_ELEMENTS);
    int block_index_x = (blockIdx.x*NUM_SIMULTANEOUS_C_ROW_ELEMENTS);

    __shared__ _DOUBLE_ A_block_0y[ROW_BLOCK_SIZE][COL_BLOCK_SIZE];
    __shared__ _DOUBLE_ A_block_1y[ROW_BLOCK_SIZE][COL_BLOCK_SIZE];

    __shared__ _DOUBLE_ B_block_0x[COL_BLOCK_SIZE][ROW_BLOCK_SIZE];
    __shared__ _DOUBLE_ B_block_1x[COL_BLOCK_SIZE][ROW_BLOCK_SIZE];
    //deal with block size != tile size
    int block_x_offset = COL_BLOCK_SIZE;//min(COL_BLOCK_SIZE, blockDim.x);//
    int block_y_offset = ROW_BLOCK_SIZE;//min(ROW_BLOCK_SIZE, blockDim.y); //


    int row0y =  (block_index_y)*block_y_offset+ block_row;
    int row1y =  (block_index_y+1)*block_y_offset + block_row;
    int col0x =  (block_index_x)*block_x_offset + block_col;
    int col1x =  (block_index_x+1)*block_x_offset + block_col;
    
    _DOUBLE_ _c0y0x = 0;
    _DOUBLE_ _c0y1x = 0;
    _DOUBLE_ _c1y0x = 0;
    _DOUBLE_ _c1y1x = 0;

    int B_increment = block_y_offset*square_dim;

    //if incrementing inside for loop
    int B_index_0x = col0x + block_row* square_dim  ;
    int B_index_1x = col1x + block_row* square_dim ;

    int A_index_0y = row0y * square_dim + block_col;
    int A_index_1y = row1y * square_dim + block_col ;
//  unsigned int bank_conflict_avoidance_index;
#pragma unroll
    for (unsigned int stride = 0;
         stride < gridDim.x*NUM_SIMULTANEOUS_C_COL_ELEMENTS;
         ++stride
            ,A_index_0y += block_x_offset
            ,A_index_1y += block_x_offset
            ,B_index_0x += B_increment
            ,B_index_1x += B_increment
            ) {

//Note: could probably avoid checks for  a good portion of loop if we
// split it up to do 1 loop over indexes we know will be inbound,
// then this will be the second loop over the rest

////////////load sub-blocks into shared memory: each thread does one load to each array//////////////////

      //check if both A indices inbounds
        if (A_index_1y < num_elements) {
            A_block_0y[block_row][block_col] = A[A_index_0y];
            A_block_1y[block_row][block_col] = A[A_index_1y];
        } else {// this thread walks off array
            A_block_1y[block_row][block_col] = 0;
            //check if A0y direction inbounds
            if (A_index_0y < num_elements) {
                A_block_0y[block_row][block_col] = A[A_index_0y];
            } else {// this thread walks off array
                A_block_0y[block_row][block_col] = 0;
            }
        }

        //check if both B indices inbounds
        if (B_index_1x < num_elements) {
            B_block_0x[block_row][block_col] = B[B_index_0x];
            B_block_1x[block_row][block_col] = B[B_index_1x];
        } else {// this thread walks off array
            B_block_1x[block_row][block_col] = 0;
            //check if B0x index inbounds

            if (B_index_0x < num_elements) {
                B_block_0x[block_row][block_col] = B[B_index_0x];
            } else {// this thread walks off array
                B_block_0x[block_row][block_col] = 0;
            }
        }
        __syncthreads();


//        each thread within the block dim loops over rows to add results


// Note: due to thread divergence, produced marginally better results than having all threads compute

      if((row1y < square_dim) ) { // if within row bound
        if (col1x < square_dim){//all fit
#pragma unroll
          for (unsigned int k = 0; k < block_x_offset; ++k) {
            _c0y0x += A_block_0y[block_row][k] *  B_block_0x[k][block_col];//
            _c0y1x += A_block_0y[block_row][k] *  B_block_1x[k][block_col];//
            _c1y0x += A_block_1y[block_row][k] *  B_block_0x[k][block_col];//
            _c1y1x += A_block_1y[block_row][k] *  B_block_1x[k][block_col];//
          }
        } else if(col0x < square_dim){//if within col bound
#pragma unroll
          for (unsigned int k = 0; k < block_x_offset; ++k) {
            _c0y0x += A_block_0y[block_row][k] *  B_block_0x[k][block_col];//
            _c1y0x += A_block_1y[block_row][k] *  B_block_0x[k][block_col];//
          }
        }
      }
      else if((row0y < square_dim) ) { // if within matrix bounds
        if (col1x < square_dim){//both cols fit
#pragma unroll
          for (unsigned int k = 0; k < block_x_offset; ++k) {
            _c0y0x += A_block_0y[block_row][k] *  B_block_0x[k][block_col];//
            _c0y1x += A_block_0y[block_row][k] *  B_block_1x[k][block_col];//
          }
        } else if(col0x < square_dim){//only 1 col fits
#pragma unroll
          for (unsigned int k = 0; k < block_x_offset; ++k) {
            _c0y0x += A_block_0y[block_row][k] *  B_block_0x[k][block_col];//
          }        }
      }
      __syncthreads();

//        #pragma unroll
//        for (k = 0; k < block_x_offset; ++k) {
//
//            _c0y0x += A_block_0y[block_row][k] *  B_block_0x[k][block_col];//
//            _c0y1x += A_block_0y[block_row][k] *  B_block_1x[k][block_col];//
//            _c1y0x += A_block_1y[block_row][k] *  B_block_0x[k][block_col];//
//            _c1y1x += A_block_1y[block_row][k] *  B_block_1x[k][block_col];//
//        }
//        __syncthreads();

    }

//    if((row0y < square_dim) && (col0x < square_dim)) { // if within matrix bounds
//        C[row0y * square_dim + col0x] = _c0y0x;
//    }
//
//    if((row0y < square_dim) && (col1x < square_dim)) { // if within matrix bounds
//        C[row0y * square_dim + col1x] = _c0y1x;
//    }
//
//    if((row1y < square_dim) && (col0x < square_dim)) { // if within matrix bounds
//        C[row1y * square_dim + col0x] = _c1y0x;
//    }
//
//    if((row1y < square_dim) && (col1x < square_dim)) { // if within matrix bounds
//        C[row1y * square_dim + col1x] = _c1y1x;
//    }

        if((row1y < square_dim) ) { // if within row bound

        if (col1x < square_dim){//all fit
            C[row0y * square_dim + col0x] = _c0y0x;
            C[row0y * square_dim + col1x] = _c0y1x;
            C[row1y * square_dim + col0x] = _c1y0x;
            C[row1y * square_dim + col1x] = _c1y1x;
        } else if(col0x < square_dim){//if within col bound
            C[row0y * square_dim + col0x] = _c0y0x;
            C[row1y * square_dim + col0x] = _c1y0x;
        }
    }
    else if((row0y < square_dim) ) { // if within matrix bounds
        if (col1x < square_dim){//both cols fit
            C[row0y * square_dim + col0x] = _c0y0x;
            C[row0y * square_dim + col1x] = _c0y1x;
        } else if(col0x < square_dim){//only 1 col fits
            C[row0y * square_dim + col0x] = _c0y0x;
        }
    }


    
    
}
//__global__ void matMul(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
__global__ void matMul_1sync_element(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
  int num_elements = square_dim*square_dim;
  int block_row = threadIdx.y;
  int block_col = threadIdx.x;
  int nextTile = 0;
  //since we are operating on adjacent squares
  int block_index_y = (blockIdx.y*NUM_SIMULTANEOUS_C_COL_ELEMENTS);
  int block_index_x = (blockIdx.x*NUM_SIMULTANEOUS_C_ROW_ELEMENTS);

  __shared__ _DOUBLE_ A_block_0y[NUM_SM_TILES][ROW_BLOCK_SIZE][COL_BLOCK_SIZE];
  __shared__ _DOUBLE_ A_block_1y[NUM_SM_TILES][ROW_BLOCK_SIZE][COL_BLOCK_SIZE];

  __shared__ _DOUBLE_ B_block_0x[NUM_SM_TILES][COL_BLOCK_SIZE][ROW_BLOCK_SIZE];
  __shared__ _DOUBLE_ B_block_1x[NUM_SM_TILES][COL_BLOCK_SIZE][ROW_BLOCK_SIZE];
  //deal with block size != tile size
  int block_x_offset = min(COL_BLOCK_SIZE, blockDim.x);
  int block_y_offset = min(ROW_BLOCK_SIZE, blockDim.y);


  int row0y =  (block_index_y)*block_y_offset+ block_row;
  int row1y =  (block_index_y+1)*block_y_offset + block_row;
  int col0x =  (block_index_x)*block_x_offset + block_col;
  int col1x =  (block_index_x+1)*block_x_offset + block_col;

  _DOUBLE_ _c0y0x = 0;
  _DOUBLE_ _c0y1x = 0;
  _DOUBLE_ _c1y0x = 0;
  _DOUBLE_ _c1y1x = 0;


  int B_increment = block_y_offset*square_dim;


  //if incrementing inside for loop
  int B_index_0x = col0x + block_row* square_dim ;
  int B_index_1x = col1x + block_row* square_dim ;

  int A_index_0y = row0y * square_dim + block_col ;
  int A_index_1y = row1y * square_dim + block_col ;


  //Load tile for 1st math calculation loop
  if (A_index_1y < num_elements) {
    A_block_0y[nextTile][block_row][block_col] = A[A_index_0y];
    A_block_1y[nextTile][block_row][block_col] = A[A_index_1y];
  } else {// this thread walks off array
    A_block_1y[nextTile][block_row][block_col] = 0;
    //check if A0y direction inbounds
    if (A_index_0y < num_elements) {
      A_block_0y[nextTile][block_row][block_col] = A[A_index_0y];
    } else {// this thread walks off array
      A_block_0y[nextTile][block_row][block_col] = 0;
    }
  }

  //check if both B indices inbounds
  if (B_index_1x < num_elements) {
    B_block_0x[nextTile][block_row][block_col] = B[B_index_0x];
    B_block_1x[nextTile][block_row][block_col] = B[B_index_1x];
  } else {// this thread walks off array
    B_block_1x[nextTile][block_row][block_col] = 0;
    //check if B0x index inbounds

    if (B_index_0x < num_elements) {
      B_block_0x[nextTile][block_row][block_col] = B[B_index_0x];
    } else {// this thread walks off array
      B_block_0x[nextTile][block_row][block_col] = 0;
    }
  }
  A_index_0y += block_x_offset;
  A_index_1y += block_x_offset;
  B_index_0x += B_increment;
  B_index_1x += B_increment;
  __syncthreads();

#pragma unroll
  for (unsigned int stride = 0;
       stride < gridDim.x*NUM_SIMULTANEOUS_C_COL_ELEMENTS -1;
       ++stride
               ,A_index_0y += block_x_offset
               ,A_index_1y += block_x_offset
               ,B_index_0x += B_increment
               ,B_index_1x += B_increment
          ) {
    //each thread within the block dim loops over rows to add results
    //returning single element in C
#pragma unroll
    for (unsigned int k = 0; k < block_x_offset; ++k) {
      _c0y0x += A_block_0y[nextTile][block_row][k] *  B_block_0x[nextTile][k][block_col];//
      _c0y1x += A_block_0y[nextTile][block_row][k] *  B_block_1x[nextTile][k][block_col];//
      _c1y0x += A_block_1y[nextTile][block_row][k] *  B_block_0x[nextTile][k][block_col];//
      _c1y1x += A_block_1y[nextTile][block_row][k] *  B_block_1x[nextTile][k][block_col];//
    }

    //Load the tile for the next calculation
    nextTile = !nextTile;
    if (A_index_1y < num_elements) {
      A_block_0y[nextTile][block_row][block_col] = A[A_index_0y];
      A_block_1y[nextTile][block_row][block_col] = A[A_index_1y];
    } else {// this thread walks off array
      A_block_1y[nextTile][block_row][block_col] = 0;
      //check if A0y direction inbounds
      if (A_index_0y < num_elements) {
        A_block_0y[nextTile][block_row][block_col] = A[A_index_0y];
      } else {// this thread walks off array
        A_block_0y[nextTile][block_row][block_col] = 0;
      }
    }

    //check if both B indices inbounds
    if (B_index_1x < num_elements) {
      B_block_0x[nextTile][block_row][block_col] = B[B_index_0x];
      B_block_1x[nextTile][block_row][block_col] = B[B_index_1x];
    } else {// this thread walks off array
      B_block_1x[nextTile][block_row][block_col] = 0;
      //check if B0x index inbounds

      if (B_index_0x < num_elements) {
        B_block_0x[nextTile][block_row][block_col] = B[B_index_0x];
      } else {// this thread walks off array
        B_block_0x[nextTile][block_row][block_col] = 0;
      }
    }
    __syncthreads();

  }

  //perform final calculation
#pragma unroll
  for (unsigned int k = 0; k < block_x_offset; ++k) {
    _c0y0x += A_block_0y[nextTile][block_row][k] *  B_block_0x[nextTile][k][block_col];//
    _c0y1x += A_block_0y[nextTile][block_row][k] *  B_block_1x[nextTile][k][block_col];//
    _c1y0x += A_block_1y[nextTile][block_row][k] *  B_block_0x[nextTile][k][block_col];//
    _c1y1x += A_block_1y[nextTile][block_row][k] *  B_block_1x[nextTile][k][block_col];//
  }
  __syncthreads();

//perform assignment
  if((row0y < square_dim) && (col0x < square_dim)) { // if within matrix bounds
    C[row0y * square_dim + col0x] = _c0y0x;
  }

  if((row0y < square_dim) && (col1x < square_dim)) { // if within matrix bounds
    C[row0y * square_dim + col1x] = _c0y1x;
  }

  if((row1y < square_dim) && (col0x < square_dim)) { // if within matrix bounds
    C[row1y * square_dim + col0x] = _c1y0x;
  }

  if((row1y < square_dim) && (col1x < square_dim)) { // if within matrix bounds
    C[row1y * square_dim + col1x] = _c1y1x;
  }


}

//__global__ void matMul(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
__global__ void matMul_altSync(int square_dim, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

  int nextTile = 0;


  __shared__ _DOUBLE_ A_block[2][ROW_BLOCK_SIZE][COL_BLOCK_SIZE];
  __shared__ _DOUBLE_ B_block[2][COL_BLOCK_SIZE][ROW_BLOCK_SIZE];


  int block_row = threadIdx.y;
  int block_col = threadIdx.x;
  //deal with block size != tile size
  int block_x_offset = min(COL_BLOCK_SIZE, blockDim.x);
  int block_y_offset = min(ROW_BLOCK_SIZE, blockDim.y);

  int row =  blockIdx.y*block_y_offset + block_row;
  int col =  blockIdx.x*block_x_offset + block_col;
//    int index = row *square_dim + col;
//    int row = blockIdx.y * blockDim.y + block_col;
  _DOUBLE_ _c = 0;

  int col_offset;
  int row_offset;
  int A_index;
  int B_index;
  col_offset = (block_col);
  row_offset = (block_row );
  A_index = row * square_dim + col_offset;
  B_index = row_offset * square_dim + col;

  //load sub-blocks into shared memory: each thread does one load
  if (A_index < square_dim*square_dim) {
    A_block[nextTile][block_row][block_col] = A[A_index];
  } else {// this thread walks off array
    A_block[nextTile][block_row][block_col] = 0;
  }

  if (B_index < square_dim*square_dim) {
    B_block[nextTile][block_row][block_col] = B[B_index];
  } else {// this thread walks off array
    B_block[nextTile][block_row][block_col] = 0;
  }
  __syncthreads();


#pragma unroll
  for (unsigned int kk = 1; kk < gridDim.x; ++kk) {
    //each thread within the block dim loops over rows to add results
    //returning single element in C

#pragma unroll
    for (unsigned int k = 0; k < block_x_offset; ++k) {
      _c += A_block[nextTile][block_row][k] *  B_block[nextTile][k][block_col];//
    }

    //load next tile
    nextTile = !nextTile;
    col_offset = (kk*block_x_offset + block_col);
    row_offset = (kk*block_y_offset + block_row );
    A_index = row * square_dim + col_offset;
    B_index = row_offset * square_dim + col;

    //load sub-blocks into shared memory: each thread does one load
    if (A_index < square_dim*square_dim) {
      A_block[nextTile][block_row][block_col] = A[A_index];
    } else {// this thread walks off array
      A_block[nextTile][block_row][block_col] = 0;
    }

    if (B_index < square_dim*square_dim) {
      B_block[nextTile][block_row][block_col] = B[B_index];
    } else {// this thread walks off array
      B_block[nextTile][block_row][block_col] = 0;
    }
    __syncthreads();

  }
#pragma unroll
  for (unsigned int k = 0; k < block_x_offset; ++k) {
    _c += A_block[nextTile][block_row][k] *  B_block[nextTile][k][block_col];//
  }
  __syncthreads();

  if((row < square_dim) && (col < square_dim)) { // if within matrix bounds
    C[row * square_dim + col] = _c;
  }
}