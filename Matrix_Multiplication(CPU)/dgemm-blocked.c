/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */
#include <x86intrin.h>
#include <string.h>
const char* dgemm_desc = "Simple blocked dgemm.";
#define TRANSPOSE //true

#define min(a,b) (((a)<(b))?(a):(b))

//Shared cache characteristics
#define CACHE_LINE_SIZE 64
#define DOUBLES_PER_LINE 8 // 64 BYTES / 64 BITS = 8
#define DOUBLES_PER_REGISTER 2 //SIMD == 2

//L2 cache characteristics
#define L2_ASSOCIATIVITY 16//16 *8 = 128 doubles per block
#define L2_DOUBLES_PER_BLOCK L2_ASSOCIATIVITY * DOUBLES_PER_LINE
#define NUM_L2_BLOCKS 4096
#define L2_SIZE 4096 *1024 //KB
#define L2_BLOCKS NUM_L2_BLOCKS/100 // for A[i][:], B[j:j+1][:]*
#define A_BLOCK_SIZE_L2 543//(int)min(lda, L2_BLOCKS*L2_DOUBLES_PER_BLOCK) // 1X SZ ROW
#define B_BLOCK_SIZE_L2 A_BLOCK_SIZE_L2//min(lda*lda, DOUBLES_PER_REGISTER * L2_BLOCKS) // 2 x SZ' Rectangle


//L1 cache characteristics
#define L1_ASSOCIATIVITY 8
#define L1_DOUBLES_PER_BLOCK L1_ASSOCIATIVITY * DOUBLES_PER_LINE
#define NUM_L1_BLOCKS 64
#define L1_SIZE 32 *1024 //KB
#define L1_BLOCKS NUM_L1_BLOCKS/10

#define A_BLOCK_SIZE_L1 543//(int)min(lda, 1000)// L1_BLOCKS * L1_DOUBLES_PER_BLOCK)
#define B_BLOCK_SIZE_L1 A_BLOCK_SIZE_L1//min(B_BLOCK_SIZE_L2, DOUBLES_PER_REGISTER * L1_BLOCKS)



#if !defined(BLOCK_SIZE)
//#define BLOCK_SIZE 37
//#define BLOCK_SIZE 256
//#define BLOCK_SIZE 719
#define BLOCK_SIZE B_BLOCK_SIZE_L2
#define BLOCK_SIZE_MIDDLE B_BLOCK_SIZE_L1
#define BLOCK_SIZE_INNER_A 8//TODO: do this
#define BLOCK_SIZE_INNER_B 16//TODO: do this
//#define BLOCK_SIZE_TLB B_BLOCK_SIZE_L1
#endif


static inline void do_block_SIMD (int lda,
                      int M,
                      int N,
                      int K,
                      double *restrict  A,  double *restrict  B, double *restrict C)
{

    /* For each row i of A */
    for (int i = 0; i < M; ++i) {

        /* For each column j of B */
        for (int j = 0; j < N; ++j) {
            /* Compute C(i,j) */
           register double cij = C[i * lda + j];
            int k = 0;
            int num_paired_loads = 12;
            int num_total_loads = num_paired_loads;
            for (; k < K-(num_total_loads-1); k+=num_total_loads) {//increment by 16s until we get to the last <16 elements
//              for (; k < K-1; k+=2) {//increment by 16s until we get to the last <16 elements

//                num_total_loads = min(K-k-1, num_paired_loads);//make sure the offset doesn't walk off the array
                __m128d a_reg1, a_reg2, a_reg3, a_reg4, a_reg5, a_reg6;
                __m128d b_reg1, b_reg2, b_reg3, b_reg4, b_reg5, b_reg6;

                //separate hadds to accumulate into a sum, hadd that at the end
                a_reg1 = _mm_loadu_pd(&A[(i * lda + k )]); //pd1 loads 1 double into both slots
                b_reg1 = _mm_loadu_pd(&B[(j * lda + k )]);
                a_reg1 = _mm_mul_pd(a_reg1, b_reg1);// store back in a to save space


                a_reg2 = _mm_loadu_pd(&A[(i * lda + k + 2 )]); //pd1 loads 1 double into both slots
                b_reg2 = _mm_loadu_pd(&B[(j * lda + k + 2 )]);
                a_reg2 = _mm_mul_pd(a_reg2, b_reg2);// store back + 2 in a to save space


                a_reg3 = _mm_loadu_pd(&A[(i * lda + k + 4 )]); //pd1 loads 1 double into both slots
                b_reg3 = _mm_loadu_pd(&B[(j * lda + k + 4 )]);
                a_reg3 = _mm_mul_pd(a_reg3, b_reg3);// store back + 4 in a to save space

                ////start summing
                a_reg1 = _mm_add_pd(a_reg1, a_reg2);

                a_reg4 = _mm_loadu_pd(&A[(i * lda + k + 6 )]); //pd1 loads 1 double into both slots
                b_reg4 = _mm_loadu_pd(&B[(j * lda + k + 6 )]);
                a_reg4 = _mm_mul_pd(a_reg4, b_reg4);// store back + 6 in a to save space

                a_reg1 = _mm_add_pd(a_reg1, a_reg3);

                a_reg5 = _mm_loadu_pd(&A[(i * lda + k + 8 )]); //pd1 loads 1 double into both slots
                b_reg5 = _mm_loadu_pd(&B[(j * lda + k + 8 )]);
                a_reg5 = _mm_mul_pd(a_reg5, b_reg5);// store back + 8 in a to save space

                a_reg1 = _mm_add_pd(a_reg1, a_reg4);


                a_reg6 = _mm_loadu_pd(&A[(i * lda + k + 10 )]); //pd1 loads 1 double into both slots
                b_reg6 = _mm_loadu_pd(&B[(j * lda + k + 10 )]);
                a_reg6 = _mm_mul_pd(a_reg6, b_reg6);// store back + 10 in a to save space


                a_reg1 = _mm_add_pd(a_reg1, a_reg5);
                a_reg1 = _mm_add_pd(a_reg1, a_reg6);

                //horizontally add
                a_reg1 = _mm_hadd_pd(a_reg1, a_reg1);
                cij += _mm_cvtsd_f64(a_reg1);//store into cij

            }
            for (; k < K; ++k) {
                cij += A[i * lda + k] * B[j* lda + k];
            }

            C[i * lda + j] = cij;
        }
    }//else, would have dealt with even M  in preceding loop.
}


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda,
                      int M,
                      int N,
                      int K,
                      double *restrict  A,  double *restrict  B, double *restrict C)
{

    /* For each row i of A */
  for (int i = 0; i < M; ++i) {

      /* For each column j of B */
      for (int j = 0; j < N; ++j) {

          /* Compute C(i,j) */
          register double cij = C[i * lda + j];
          for (int k = 0; k < K; ++k) {
            #ifdef TRANSPOSE
              cij += A[i * lda + k] * B[j * lda + k];
            #else
              cij += A[i*lda+k] * B[k*lda+j];
            #endif
          }

          C[i * lda + j] = cij;
      }
  }
}

static inline void do_reg_B_loop (int lda,
                                 int i,
                                 int j,
                                  int N,
                                 int K,
                                 double *restrict  A,  double *restrict  B, double *restrict C){
    for (; j < N; ++j) {

        /* Compute C(i,j) */
        register double cij = C[i * lda + j];
        for (int k = 0; k < K; ++k) {
            cij += A[i * lda + k] * B[j * lda + k];
        }
        C[i * lda + j] = cij;
    }
    }

static inline void do_2_B_loop (int lda,
                                  int i,
                                  int j,
                                  int N,
                                  int K,
                                  double *restrict  A,  double *restrict  B, double *restrict C){

        for (; j < N-1; j+=2) {

            /* Compute C(i,j) */
            register double cij = C[i * lda + j];
            register double cij_1 = C[i * lda + j+1];
            double a;
            for (int k = 0; k < K; ++k) {
                a = A[i * lda + k];
                cij += a * B[j* lda + k];
                cij_1 += a * B[(j+1) * lda + k];
            }
//
            C[i * lda + j] = cij;
            C[i * lda + (j+1)] = cij_1;
        }
    if(N%2){//finish off
        do_reg_B_loop(lda, i, j, N, K, A, B, C);
    }
}

static inline void do_4_B_loop (int lda,
                                int i,
                                int j,
                                int N,
                                int K,
                                double *restrict  A,  double *restrict  B, double *restrict C){

    for (; j < N-3; j+=4) {

        /* Compute C(i,j) */
        register double cij = C[i * lda + j];
        register double cij_1 = C[i * lda + j+1];
        register double cij_2 = C[i * lda + j+2];
        register double cij_3 = C[i * lda + j+3];


        double a, a1;
        int k = 0;
        for (; k < K; ++k) {
            a = A[i * lda + k];
            cij += a * B[j* lda + k];
            cij_1 += a * B[(j+1) * lda + k];
            cij_2 += a * B[(j+2) * lda + k];
            cij_3 += a * B[(j+3) * lda + k];


        }
//
        C[i * lda + j] = cij;
        C[i * lda + (j+1)] = cij_1;
        C[i * lda + (j+2)] = cij_2;
        C[i * lda + (j+3)] = cij_3;


    }
    do_2_B_loop(lda, i, j, N, K, A, B, C);//2 #simultaneous sums
}

static inline void do_2_A_reg_B_loop (int lda,
                                  int i,
                                  int j,
                                  int N,
                                  int K,
                                  double *restrict  A,  double *restrict  B, double *restrict C){
    for (; j < N; ++j) {

        /* Compute C(i,j) */
        register double cij = C[i * lda + j];
        register double ci1j = C[(i+1) * lda + j];

        for (int k = 0; k < K; ++k) {
            cij += A[i * lda + k] * B[j * lda + k];
            ci1j += A[(i+1) * lda + k] * B[j* lda + k];

        }
        C[i * lda + j] = cij;
        C[(i+1) * lda + j] = ci1j;
    }
}

static inline void do_2_A_2_B_loop (int lda,
                                int i,
                                int j,
                                int N,
                                int K,
                                double *restrict  A,  double *restrict  B, double *restrict C){

    for (; j < N-1; j+=2) {

        /* Compute C(i,j) */
        register double cij = C[i * lda + j];
        register double cij_1 = C[i * lda + j+1];

        register double ci1j = C[(i+1) * lda + j];
        register double ci1j_1 = C[(i+1) * lda + j+1];

        double a, a1;
        for (int k = 0; k < K; ++k) {
            a = A[i * lda + k];
            cij += a * B[j* lda + k];
            cij_1 += a * B[(j+1) * lda + k];

            a1 = A[(i+1) * lda + k];
            ci1j += a1 * B[j* lda + k];
            ci1j_1 += a1 * B[(j+1) * lda + k];
        }
//
        C[i * lda + j] = cij;
        C[i * lda + (j+1)] = cij_1;

        C[(i+1) * lda + j] = ci1j;
        C[(i+1) * lda + (j+1)] = ci1j_1;
    }
    if(N%2){//finish off
        do_2_A_reg_B_loop(lda, i, j, N, K, A, B, C);
    }
}

static inline void do_2_A_4_B_loop (int lda,
                                int i,
                                int j,
                                int N,
                                int K,
                                double *restrict  A,  double *restrict  B, double *restrict C){

    for (; j < N-3; j+=4) {

        /* Compute C(i,j) */
        register double cij = C[i * lda + j];
        register double cij_1 = C[i * lda + j+1];
        register double cij_2 = C[i * lda + j+2];
        register double cij_3 = C[i * lda + j+3];

        register double ci1j = C[(i+1) * lda + j];
        register double ci1j_1 = C[(i+1) * lda + j+1];
        register double ci1j_2 = C[(i+1) * lda + j+2];
        register double ci1j_3 = C[(i+1) * lda + j+3];


        double a, a1;
        int k = 0;
        for (; k < K; ++k) {
            a = A[i * lda + k];
            cij += a * B[j* lda + k];
            cij_1 += a * B[(j+1) * lda + k];
            cij_2 += a * B[(j+2) * lda + k];
            cij_3 += a * B[(j+3) * lda + k];

            a1 = A[(i+1) * lda + k];
            ci1j += a1 * B[j* lda + k];
            ci1j_1 += a1 * B[(j+1) * lda + k];
            ci1j_2 += a1 * B[(j+2) * lda + k];
            ci1j_3 += a1 * B[(j+3) * lda + k];


        }
        C[i * lda + j] = cij;
        C[i * lda + (j+1)] = cij_1;
        C[i * lda + (j+2)] = cij_2;
        C[i * lda + (j+3)] = cij_3;

        C[(i+1) * lda + j] = ci1j;
        C[(i+1) * lda + (j+1)] = ci1j_1;
        C[(i+1) * lda + (j+2)] = ci1j_2;
        C[(i+1) * lda + (j+3)] = ci1j_3;


    }
    do_2_A_2_B_loop(lda, i, j, N, K, A, B, C);//2 #simultaneous sums
}

static inline void do_unrolled_B_loop (int lda,
                                  int i,
                                  int j,
                                       int N,
                                  int K,

                                    int increment,
                                  double *restrict  A,  double *restrict  B, double *restrict C){
    for (; j < N-(increment-1); j+=increment) {
    /* Compute C(i,j) */
    double cijs[increment];

    for (int inc_iter=0; inc_iter<increment; ++inc_iter){
        cijs[inc_iter] = C[i * lda + (j+inc_iter)];
    }

    double a;
    for (int k = 0; k < K; ++k) {
        a = A[i * lda + k];
        for (int inc_iter=0; inc_iter<increment; ++inc_iter){
            cijs[inc_iter] += a * B[j* lda + (k+ inc_iter)];
        }

    }
        for (int inc_iter=0; inc_iter<increment; ++inc_iter){
            C[i * lda + (j+inc_iter)] = cijs[inc_iter];
        }
    }
    do_2_B_loop(lda, i, j, N, K, A, B, C);//2 #simultaneous sums
}



static inline void do_block_alt (int lda,
                             int M,
                             int N,
                             int K,
                             double *restrict  A,  double *restrict  B, double *restrict C)
{

    /* 2 rows of A */
    int i = 0, j=0;
    for (; i < M-1; i+=2) {
        /* 4 rows of B^T */
        do_2_A_4_B_loop(lda, i, j, N, K, A, B, C);
    }if (M%2){
        do_4_B_loop(lda, i, j, N, K, A, B, C);//do the rest
    }

}

static inline void do_outer_block (int lda, int M, int N, int K,
                            double *restrict  A, double *restrict  B, double *restrict C
)
{
    /* For each row i of A */
    for (int i = 0; i < M; i += BLOCK_SIZE_MIDDLE) {
        /* For each column j of B */
        for (int j = 0; j < N; j += BLOCK_SIZE_MIDDLE) {
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < K; k += BLOCK_SIZE_MIDDLE) {
/* Correct block dimensions if block "goes off edge of" the matrix */
                int M_inner = min (BLOCK_SIZE_MIDDLE, M - i);
                int N_inner = min (BLOCK_SIZE_MIDDLE, N - j);
                int K_inner = min (BLOCK_SIZE_MIDDLE, K - k);

                /* Perform individual block dgemm */
                #ifdef TRANSPOSE
                do_block_alt(lda, M_inner, N_inner, K_inner, \
                         A + i*lda + k, \
                         B + j*lda + k, \
                         C + i * lda + j);
                #else
               do_block(lda, M_inner, N_inner, K_inner, \
                         A + i*lda + k, \
                         B + k*lda + j, \
                         C + i * lda + j);
                #endif
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double *restrict old_A, double *restrict  old_B, double *restrict C
) {
//Transpose B once entering function

#ifdef TRANSPOSE
        for (int i = 0; i < lda; ++i)
        for (int j = i + 1; j < lda; ++j) {
            double t = old_B[i * lda + j];
            old_B[i * lda + j] = old_B[j * lda + i];
            old_B[j * lda + i] = t;
        }
#endif

    /* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        int M = min (BLOCK_SIZE, lda - i);

        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            int N = min (BLOCK_SIZE, lda - j);

            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int K = min (BLOCK_SIZE, lda - k);

                /* Perform individual block dgemm */
                #ifdef TRANSPOSE
                do_outer_block(lda, M, N, K,
                               old_A + i*lda +  k, //single row now being passed in
                               old_B + j *lda + k,//single row now being pass in
                               C + i * lda + j
                );
                #else
                do_outer_block(lda, M, N, K,
                old_A + i*lda + k,
                old_B + k*lda + j,
                C + i*lda + j
                );
                #endif
            }
        }
    }

//transpose B back
#ifdef TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = old_B[i*lda+j];
        old_B[i*lda+j] = old_B[j*lda+i];
        old_B[j*lda+i] = t;
  }
#endif
}
