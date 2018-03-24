/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 *
 * MPI communication code written by Teofilo Erin Zosa IV
 * March 12, 2018
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include "cblock.h"
#ifdef _MPI_
#include <mpi.h>
#endif
#include "omp.h"
using namespace std;

void printMat(const char mesg[], double *E, int m, int n);
double *alloc1D(int m,int n);
extern control_block cb;



//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
    int i;

    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2);//first comp row
         i < (m+1)*(n+2);//last comp row
         i++) {
	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
	if(colIndex == 0 || //ghost cell (left)
          colIndex == (n+1) || //ghost cell (right)
          colIndex < ((n+1)/2+1)) {//[0, midpoint] <=> [0, 128]if n==255
        continue;
    }

        E_prev[i] = 1.0;
    }

    for (i = 0; //first ghost row
         i < (m+2)*(n+2); //last ghost row
         i++) {
      int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
      int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

            // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
      if(colIndex == 0 ||//ghost cell (left)
              colIndex == (n+1) ||//ghost cell (right)
              rowIndex < ((m+1)/2+1)){// 1st half of array 0s
        continue;
      }

        R[i] = 1.0;
    }
    // We only print the meshes if they are small enough
#if 0
    printMat("E_prev",E_prev,m,n);
    printMat("R",R,m,n);
#endif

#ifdef _MPI_
  int row_dim = cb.m, col_dim=cb.n, //m x n; row_dim == col_dim
          proc_col_dim=cb.px, proc_row_dim=cb.py; //
  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
//    int any_tag = 0, curr_tag = 1, prev_tag = 2, R_tag = 3;
        int any_tag = 0; int curr_tag = any_tag, prev_tag = any_tag, R_tag = any_tag;



////////////////////////////////////////////////
    /*TRUE COMMUNICATION*/
//    double *submesh_curr = alloc1D(row_dim, col_dim);
//        double *submesh_prev = alloc1D(row_dim, col_dim);
//        double *submesh_R = alloc1D(row_dim, col_dim);

    int proc_row = myrank / proc_col_dim; //this proc's row
    int proc_col = myrank % proc_col_dim; // this proc's column
    assert((proc_row * proc_col_dim + proc_col) == myrank);//sanity check
//
// //srd x scd submesh computational boxes
        int submesh_row_comp_dim = (row_dim) / proc_row_dim;
        int submesh_col_comp_dim = (col_dim) / proc_col_dim;

        int leftover_rows = row_dim - submesh_row_comp_dim * proc_row_dim;
        int leftover_cols = col_dim - submesh_col_comp_dim * proc_col_dim;



    if (myrank == 0) {
//        for (int recvr_proc_row = 0; recvr_proc_row < proc_row_dim; recvr_proc_row++) {
//            for (int recvr_proc_col = 0; recvr_proc_col < proc_col_dim; recvr_proc_col++) {
        for (int rank = nprocs-1; rank >=0 ; rank--){
            int recvr_proc_row = rank/proc_col_dim;
            int recvr_proc_col = rank % proc_col_dim;

                int dest_proc = recvr_proc_row * proc_col_dim + recvr_proc_col;
                assert (dest_proc >=0 && dest_proc <nprocs);
            assert (dest_proc == rank);



                //srd x scd submesh computational boxes
                int submesh_row_comp_dim_rcvr = submesh_row_comp_dim;
                int submesh_col_comp_dim_rcvr = submesh_col_comp_dim;

                //how many before this had extra rows?
                int row_offset = min(leftover_rows, recvr_proc_row);
                int col_offset = min(leftover_cols, recvr_proc_col);



//  //offset for procs with extra rows/cols
                int prev_submesh_row_comp_dims = (submesh_row_comp_dim + 1) * row_offset;
                int prev_submesh_col_comp_dims = (submesh_col_comp_dim + 1) * col_offset;

                int remainder_rows = (recvr_proc_row - row_offset);
                int remainder_cols = (recvr_proc_col - col_offset);

                assert(remainder_cols >= 0 && remainder_rows >= 0);


//  //generate indices
                int recvr_proc_start_row = ((remainder_rows * submesh_row_comp_dim) + prev_submesh_row_comp_dims);
                int recvr_proc_start_col = ((remainder_cols * submesh_col_comp_dim) + prev_submesh_col_comp_dims);



//  assign any extra row to procs, 1 row/proc
                if (recvr_proc_row < leftover_rows) {
                    submesh_row_comp_dim_rcvr++;

                }

//  assign any extra cols to procs, 1 col/proc
                if (recvr_proc_col < leftover_cols) {
                    submesh_col_comp_dim_rcvr++;


                }


                int submesh_row_dim = submesh_row_comp_dim_rcvr + 2;
                int submesh_col_dim = submesh_col_comp_dim_rcvr + 2;


                double *submesh_curr = alloc1D(submesh_row_dim, submesh_col_dim);
                double *submesh_prev = alloc1D(submesh_row_dim, submesh_col_dim);
                double *submesh_R = alloc1D(submesh_row_dim, submesh_col_dim);

                /*Loop to fill local submesh*/
                int recvr_proc_start_index = (recvr_proc_start_row) * (col_dim + 2) + recvr_proc_start_col;

                for (int submesh_row = 0; submesh_row < submesh_row_dim; submesh_row++) {
                    int index_row = (submesh_row) * (col_dim + 2);

                    for (int submesh_col = 0; submesh_col < submesh_col_dim; submesh_col++) {

                        int index = (recvr_proc_start_row + submesh_row)*(n + 2)  + (recvr_proc_start_col + submesh_col);
assert ((recvr_proc_start_col + submesh_col) >= 0 && (recvr_proc_start_col + submesh_col)< col_dim + 2) ;
//      top/bottom procs should have 0 at ghost cells
                        if ((recvr_proc_row == 0 && submesh_row == 0) ||
                            (recvr_proc_row == proc_row_dim - 1 && submesh_row == submesh_row_dim - 1)) {
                            bool top_bottom_ERROR = E_prev[index] == 0;
                            assert(top_bottom_ERROR);
                        }



                        if ((recvr_proc_col == proc_col_dim - 1
                            && submesh_col == submesh_col_dim - 1)) {


                              int colIndex = index % (n+2);
                              bool col_ERROR = 	(colIndex == (n+1)) && E_prev[index] == 0;

                            assert(col_ERROR);


                            assert(R[index] == 0);
                            bool right_ERROR = E_prev[index] == 0;
                            assert(right_ERROR);


                        }

                            submesh_curr[submesh_row * submesh_col_dim + submesh_col] = E[index];
                            submesh_prev[submesh_row * submesh_col_dim + submesh_col] = E_prev[index];
                            submesh_R[submesh_row * submesh_col_dim + submesh_col] = R[index];



                    }
                }/*end loop to fill submesh*/

                MPI_Request send_request[3];
                MPI_Status send_status[3];

                if (dest_proc != 0) {//not rank 0's data; send to proc
                    MPI_Isend(submesh_curr, submesh_row_dim * submesh_col_dim, MPI_DOUBLE, dest_proc,
                              curr_tag, MPI_COMM_WORLD, &send_request[0]);

                    MPI_Isend(submesh_prev, submesh_row_dim * submesh_col_dim, MPI_DOUBLE, dest_proc,
                              prev_tag, MPI_COMM_WORLD, &send_request[1]);

                    MPI_Isend(submesh_R, submesh_row_dim * submesh_col_dim, MPI_DOUBLE, dest_proc,
                              R_tag, MPI_COMM_WORLD, &send_request[2]);

                    MPI_Wait(&send_request[0], &send_status[0]);
                    MPI_Wait(&send_request[1], &send_status[1]);
                    MPI_Wait(&send_request[2], &send_status[2]);


                } else{//rank 0 doesn't need to send data over, already has it
                    for (int submesh_row = 0; submesh_row < submesh_row_dim; submesh_row++) {
                        for (int submesh_col = 0; submesh_col < submesh_col_dim; submesh_col++) {

                            ///*copy elements over to computational window*/
                            E[submesh_row * submesh_col_dim + submesh_col] = submesh_curr[submesh_row * submesh_col_dim + submesh_col];
                            E_prev[submesh_row * submesh_col_dim + submesh_col] = submesh_prev[submesh_row * submesh_col_dim + submesh_col];
                            R[submesh_row * submesh_col_dim + submesh_col] = submesh_R[submesh_row * submesh_col_dim + submesh_col];


                        }
                    }/*end loop to fill submesh*/
                }//else,  rank 0's data already here


        }
    }else{ //receive submesh from master proc

        proc_row = myrank / proc_col_dim; //this proc's row
        proc_col = myrank % proc_col_dim; // this proc's column
        assert((proc_row * proc_col_dim + proc_col) == myrank);//sanity check


    //how many before me had extra rows?
    int row_offset = min(leftover_rows, proc_row);
    int col_offset = min(leftover_cols, proc_col);

//  //offset for procs with extra rows/cols
    int prev_submesh_row_comp_dims = (submesh_row_comp_dim + 1) * row_offset;
    int prev_submesh_col_comp_dims = (submesh_col_comp_dim + 1) * col_offset;

    int remainder_rows = (proc_row - row_offset);
    int remainder_cols = (proc_col - col_offset);

    assert(remainder_cols >= 0 && remainder_rows >= 0);
//  //generate indices
    int recvr_proc_start_row = ((remainder_rows * submesh_row_comp_dim) + prev_submesh_row_comp_dims);
    int recvr_proc_start_col = ((remainder_cols * submesh_col_comp_dim) + prev_submesh_col_comp_dims);

//  assign any extra row to procs, 1 row/proc
    if (proc_row < leftover_rows) {
        submesh_row_comp_dim++;

    }

//  assign any extra cols to procs, 1 col/proc
    if (proc_col < leftover_cols) {
        submesh_col_comp_dim++;
    }


    int submesh_row_dim = submesh_row_comp_dim + 2;
    int submesh_col_dim = submesh_col_comp_dim + 2;



        double *submesh_curr = alloc1D(submesh_row_dim, submesh_col_dim);
        double *submesh_prev = alloc1D(submesh_row_dim, submesh_col_dim);
        double *submesh_R = alloc1D(submesh_row_dim, submesh_col_dim);
        MPI_Request recv_request[3];
        MPI_Status recv_status[3];

        int src_proc = 0;
        MPI_Irecv(E, submesh_row_dim * submesh_col_dim, MPI_DOUBLE, src_proc,
                  curr_tag, MPI_COMM_WORLD, &recv_request[0]);
        MPI_Irecv(E_prev, submesh_row_dim * submesh_col_dim, MPI_DOUBLE, src_proc,
                  prev_tag, MPI_COMM_WORLD, &recv_request[1]);
        MPI_Irecv(R, submesh_row_dim * submesh_col_dim, MPI_DOUBLE, src_proc,
                  R_tag, MPI_COMM_WORLD, &recv_request[2]);
        MPI_Wait(&recv_request[0], &recv_status[0]);
        MPI_Wait(&recv_request[1], &recv_status[1]);
        MPI_Wait(&recv_request[2], &recv_status[2]);
}





#if 0
  printMat("E_prev",E_prev,submesh_row_comp_dim,submesh_col_comp_dim);
  printMat("R",R,submesh_row_comp_dim,submesh_col_comp_dim);
#endif
#endif

}



double *alloc1D(int m,int n){
    int nx=n, ny=m;
    double *E;
    // Ensures that allocated memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}
