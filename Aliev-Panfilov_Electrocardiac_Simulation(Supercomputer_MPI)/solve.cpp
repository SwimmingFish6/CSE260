/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 *
 * MPI communication code and optimizations by Teofilo Erin Zosa IV on March 12, 2018
 * 
 */
#define NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <string.h>

#ifdef _MPI_
#include <mpi.h>
#endif
#include "omp.h"

using namespace std;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
double *alloc1D(int m,int n);

extern control_block cb;

#ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
__attribute__((optimize("no-tree-vectorize")))
#endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve_single_proc(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {

  // Simulated time is different from the integer timestep number
  double t = 0.0;

  double *E = *_E, *E_prev = *_E_prev;
  double *R_tmp = R;
  double *E_tmp = *_E;
  double *E_prev_tmp = *_E_prev;
  double mx, sumSq;
  int niter;
  int row_dim = cb.m, col_dim=cb.n, //m x n; row_dim == col_dim
          block_col_dim=cb.px, block_row_dim=cb.py; //
  bool noComm=cb.noComm;//use to measure communication overhead
  int innerBlockRowStartIndex = (col_dim+2)+1;
  int innerBlockRowEndIndex = (((row_dim+2)*(col_dim+2) - 1)
                               - (col_dim))
                              - (col_dim+2);


  // We continue to sweep over the mesh until the simulation has reached
  // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){

    if  (cb.debug && (niter==0)){
      stats(E_prev,row_dim,col_dim,&mx,&sumSq);
      double l2norm = L2Norm(sumSq);
      repNorms(l2norm,mx,dt,row_dim,col_dim,-1, cb.stats_freq);
      if (cb.plot_freq)
        plotter->updatePlot(E,  -1, row_dim+1, col_dim+1);
    }

    /*
     * Copy data from boundary of the computational box to the
     * padding region, set up for differencing computational box's boundary
     *
     * These are physical boundary conditions, and are not to be confused
     * with ghost cells that we would use in an MPI implementation
     *
     * The reason why we copy boundary conditions is to avoid
     * computing single sided differences at the boundaries
     * which increase the running time of solve()
     *
     */

    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int index,inner_block_row;


    // Fills in the TOP Ghost Cells
    /* Top stencil point == bottom stencil point*/
    for (index = 0;
         index < (col_dim+2);
         index++) {
      E_prev[index] = E_prev[index + (col_dim+2)*2];
    }


    // Fills in the RIGHT Ghost Cells
    /* RIGHT stencil point == LEFT stencil point*/

    for (index = (col_dim+1);
         index < (row_dim+2)*(col_dim+2);
         index+=(col_dim+2)) {
      E_prev[index] = E_prev[index-2];
    }


    // Fills in the LEFT Ghost Cells
    for (index = 0;
         index < (row_dim+2)*(col_dim+2);
         index+=(col_dim+2)) {
      E_prev[index] = E_prev[index+2];
    }

    // Fills in the BOTTOM Ghost Cells
    for (index = ((row_dim+2)*(col_dim+2)-(col_dim+2));
         index < (row_dim+2)*(col_dim+2);
         index++) {
      E_prev[index] = E_prev[index - (col_dim+2)*2];
    }


//////////////////////////////////////////////////////////////////////////////

//#define FUSED 1


    //NOTE: intel compiler may vectorize loop better (i.e. AVX)
#ifdef FUSED
    // Solve for the excitation, a PDE

    //unroll and jam
    for(inner_block_row = innerBlockRowStartIndex;
        inner_block_row <= innerBlockRowEndIndex;
        inner_block_row+=(col_dim+2)) {
      E_tmp = E + inner_block_row;
      E_prev_tmp = E_prev + inner_block_row;
      R_tmp = R + inner_block_row;

      for(index = 0; index < col_dim; index++) {
        //PDE solver
        E_tmp[index] = E_prev_tmp[index]+
                     alpha*(E_prev_tmp[index+1]+E_prev_tmp[index-1]-4*E_prev_tmp[index]+E_prev_tmp[index+(col_dim+2)]+E_prev_tmp[index-(col_dim+2)]);

        //ODE solver
        E_tmp[index] += -dt*(kk*E_prev_tmp[index]*(E_prev_tmp[index]-a)*(E_prev_tmp[index]-1)
                           +E_prev_tmp[index]*R_tmp[index]);
        R_tmp[index] += dt*(epsilon+M1* R_tmp[index]/( E_prev_tmp[index]+M2))*(-R_tmp[index]-kk*E_prev_tmp[index]*(E_prev_tmp[index]-b-1));
      }
    }
#else
    // Solve for the excitation, a PDE
    for(inner_block_row = innerBlockRowStartIndex; inner_block_row <= innerBlockRowEndIndex; inner_block_row+=(col_dim+2)) {
        E_tmp = E + inner_block_row;
            E_prev_tmp = E_prev + inner_block_row;
            for(index = 0; index < col_dim; index++) {
                E_tmp[index] = E_prev_tmp[index]+alpha*(E_prev_tmp[index+1]+E_prev_tmp[index-1]-4*E_prev_tmp[index]+E_prev_tmp[index+(col_dim+2)]+E_prev_tmp[index-(col_dim+2)]);
            }
    }

    /*
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(inner_block_row = innerBlockRowStartIndex; inner_block_row <= innerBlockRowEndIndex; inner_block_row+=(col_dim+2)) {
        E_tmp = E + inner_block_row;
        R_tmp = R + inner_block_row;
	E_prev_tmp = E_prev + inner_block_row;
        for(index = 0; index < col_dim; index++) {
	  E_tmp[index] += -dt*(kk*E_prev_tmp[index]*(E_prev_tmp[index]-a)*(E_prev_tmp[index]-1)+E_prev_tmp[index]*R_tmp[index]);
	  R_tmp[index] += dt*(epsilon+M1* R_tmp[index]/( E_prev_tmp[index]+M2))*(-R_tmp[index]-kk*E_prev_tmp[index]*(E_prev_tmp[index]-b-1));
        }
    }
#endif
    /////////////////////////////////////////////////////////////////////////////////

    /*Update Stats*/
    if (cb.stats_freq){
      if ( !(niter % cb.stats_freq)){
        stats(E,row_dim,col_dim,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,row_dim,col_dim,niter, cb.stats_freq);
      }
    }

    /*Plot data*/
    if (cb.plot_freq){
      if (!(niter % cb.plot_freq)){
        plotter->updatePlot(E,  niter, row_dim, col_dim);
      }
    }

    // Swap current and previous meshes
    double *tmp = E; E = E_prev; E_prev = tmp;

  } //end of 'niter' loop at the beginning

  // return the L2 and infinity norms via in-out parameters
  stats(E_prev,row_dim,col_dim,&Linf,&sumSq);
  L2 = L2Norm(sumSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}

static void inline solve_MPI(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {

  // Simulated time is different from the integer timestep number
  double t = 0.0;

  double *E = *_E,
          *E_prev = *_E_prev;


  double *R_tmp = R;
  double *E_tmp = *_E;
  double *E_prev_tmp = *_E_prev;

  //for spatial locality in kernel
   register double *E_prev_tmp_top,*E_prev_tmp_bottom;

  double mx, sumSq;
  int niter;
  int row_dim = cb.m, col_dim=cb.n, //m x n; row_dim == col_dim
          proc_col_dim=cb.px, proc_row_dim=cb.py; //
  bool noComm=cb.noComm;//use to measure communication overhead


  int nprocs, myrank;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  int root_proc = 0;
  //proc's position in processor geometry
  int proc_row = myrank/proc_col_dim; //this proc's row
  int proc_col = myrank%proc_col_dim; // this proc's column

  assert((proc_row * proc_col_dim + proc_col) == myrank);//sanity check

  int submesh_row_comp_dim = (row_dim)/proc_row_dim;
  int submesh_col_comp_dim = (col_dim)/proc_col_dim;

  int leftover_rows = row_dim - submesh_row_comp_dim * proc_row_dim;
  int leftover_cols = col_dim - submesh_col_comp_dim * proc_col_dim;

//  assign any extra row to procs, 1 row/proc
  if (proc_row < leftover_rows){
    submesh_row_comp_dim++;
  }

//  assign any extra cols to procs, 1 col/proc
  if (proc_col < leftover_cols){
    submesh_col_comp_dim++;
  }


  int submesh_row_dim = submesh_row_comp_dim + 2;//top&btm ghost cells
  int submesh_col_dim = submesh_col_comp_dim + 2;//right&lft ghost cells

    double e_prev_sum[submesh_col_comp_dim];
  //for temporal locality
    register double e_prev_tmp_curr, e_prev_tmp_next, e_prev_tmp_prev, r_tmp, e_prev_tmp_kk;



  int innerBlockRowStartIndex = (submesh_col_dim)+1;//first computational box

  int innerBlockRowEndIndex = ((submesh_row_dim)*(submesh_col_dim) - 1)//last index
                              - (submesh_col_comp_dim) //
                                 -submesh_col_dim;
  //= last computational row

  int submesh_size = submesh_row_dim * submesh_col_dim;

  double *send_left_ghost = alloc1D(submesh_row_dim, 1 ); //
  double *recv_left_ghost = alloc1D(submesh_row_dim, 1 ); //

  double *send_right_ghost = alloc1D(submesh_row_dim, 1 ); //
  double *recv_right_ghost = alloc1D(submesh_row_dim, 1 ); //

  // We continue to sweep over the mesh until the simulation has reached
  // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){


    if  (cb.debug && (niter==0)){
      stats(E_prev,submesh_row_comp_dim,submesh_col_comp_dim,&mx,&sumSq);
      double l2norm = L2Norm(sumSq);
      repNorms(l2norm,mx,dt,submesh_row_comp_dim,submesh_col_comp_dim,-1, cb.stats_freq);
      if (cb.plot_freq)
        plotter->updatePlot(E,  -1, submesh_row_comp_dim+1, submesh_col_comp_dim+1);
    }



    int index,inner_block_row;
    int buf_ind;


    if (noComm){//do nothing

        //account for computation we would have done if we were communicating to other processes

        // put computational column into buffer
        for (index = 1, buf_ind = 0;
             index < submesh_size;
             index += (submesh_col_dim), ++buf_ind) {
            send_left_ghost[buf_ind] = E_prev[index];
        }

        // put computational column into buffer
        for (index = (submesh_col_comp_dim), buf_ind = 0;
             index < (submesh_row_dim) * (submesh_col_dim);
             index += (submesh_col_dim), ++buf_ind) {
            send_right_ghost[buf_ind] = E_prev[index];
        }
    }
    else {
      int sendR_recvR_tag = 3;
      int sendL_recvL_tag = 4;
      int sendUp_recvUp_tag = 5;
      int sendBtm_recvBtm_tag = 6;
      int any_tag = 0;
      MPI_Request recv_request[4];
      MPI_Request send_request[4];



//////////////////////////////////////////////////////////////////////////////
      /*BEGIN sends/receives*/

/*LEFT: send/rcv*/
      if (0 < proc_col) {
        int dest_proc = myrank - 1;
//        int dest_proc = (proc_row) * proc_col_dim + (proc_col-1);

        int src_proc = dest_proc;
        //ready to receive ASAP
        MPI_Irecv(recv_left_ghost, submesh_row_dim, MPI_DOUBLE, src_proc,
                  any_tag, MPI_COMM_WORLD, &recv_request[0]);

        // put computational column into buffer
        for (index = 1, buf_ind = 0;
             index < submesh_size;
             index += (submesh_col_dim), ++buf_ind) {
          send_left_ghost[buf_ind] = E_prev[index];
        }


        MPI_Isend(send_left_ghost, submesh_row_dim, MPI_DOUBLE, dest_proc,
                  any_tag, MPI_COMM_WORLD, &send_request[0]);//async send or send_Rcv?


      }//else already have own ghost cells

/*RIGHT: send/rcv*/

      if (proc_col < proc_col_dim - 1) {
        int dest_proc = myrank + 1;
//        int dest_proc = (proc_row) * proc_col_dim + (proc_col+1);

        int src_proc = dest_proc;
        //ready to receive ASAP
        MPI_Irecv(recv_right_ghost, submesh_row_dim, MPI_DOUBLE, src_proc,
                  any_tag, MPI_COMM_WORLD, &recv_request[1]);

        // put computational column into buffer
        for (index = (submesh_col_comp_dim), buf_ind = 0;
             index < (submesh_row_dim) * (submesh_col_dim);
             index += (submesh_col_dim), ++buf_ind) {
          send_right_ghost[buf_ind] = E_prev[index];
        }

        MPI_Isend(send_right_ghost, submesh_row_dim, MPI_DOUBLE, dest_proc,
                  any_tag, MPI_COMM_WORLD, &send_request[1]);//async send or send_Rcv?

      }

/*TOP: send/rcv*/
      if (0 < proc_row) {

        int dest_proc =  myrank - proc_col_dim;//(proc_row-1) * proc_col_dim + proc_col;//previous row
        int src_proc = dest_proc;

        MPI_Isend(E_prev + innerBlockRowStartIndex - 1, submesh_col_dim, MPI_DOUBLE, dest_proc,
                  any_tag, MPI_COMM_WORLD, &send_request[2]);//async send or send_Rcv?
        MPI_Irecv(E_prev, submesh_col_dim, MPI_DOUBLE, src_proc,
                  any_tag, MPI_COMM_WORLD, &recv_request[2]);

          assert (innerBlockRowStartIndex - 1 == submesh_col_dim);


      }

/*BOTTOM: send/rcv*/
      if (proc_row < proc_row_dim - 1) {
        int dest_proc = myrank + proc_col_dim;//(proc_row+1) * proc_col_dim + proc_col;//next row
        int src_proc = dest_proc;

        MPI_Isend(E_prev + innerBlockRowEndIndex - 1, submesh_col_dim, MPI_DOUBLE, dest_proc,
                  any_tag, MPI_COMM_WORLD, &send_request[3]);//async send or send_Rcv?
        MPI_Irecv(E_prev + (innerBlockRowEndIndex-1) + submesh_col_dim , submesh_col_dim, MPI_DOUBLE, src_proc,
                  any_tag, MPI_COMM_WORLD, &recv_request[3]);

          assert(((submesh_row_dim)*(submesh_col_dim)-(submesh_col_dim)) == ((innerBlockRowEndIndex-1) + submesh_col_dim ));
          assert(((submesh_row_dim)*(submesh_col_dim)-(submesh_col_dim*2)) == innerBlockRowEndIndex - 1);
      }


//////*END sends/receives*/////////////////////////////////////////////////






//////////////*BEGIN Wait for sends/receives*/////////////////////////////////

      MPI_Status send_status[4];
      MPI_Status recv_status[4];

/*LEFT: Wait*/
      if (0 < proc_col) {
        MPI_Wait(&recv_request[0], &recv_status[0]);
        MPI_Wait(&send_request[0], &send_status[0]);
      }
/*RIGHT: Wait*/

      if (proc_col < proc_col_dim - 1) {
        MPI_Wait(&recv_request[1], &recv_status[1]);
        MPI_Wait(&send_request[1], &send_status[1]);
      }

/*TOP: Wait*/
      if (0 < proc_row) {
        MPI_Wait(&send_request[2], &send_status[2]);
        MPI_Wait(&recv_request[2], &recv_status[2]);
      }

/*BOTTOM: wait*/
      if (proc_row < proc_row_dim - 1) {
        MPI_Wait(&send_request[3], &send_status[3]);
        MPI_Wait(&recv_request[3], &recv_status[3]);
      }


////*END Wait for sends/receives*/////////////////////////////////////////////////

    }

////*BEGIN filling in local ghost cells*//////////////////////////////////////

    /*
     * Copy data from boundary of the computational box to the
     * padding region, set up for differencing computational box's boundary
     *
     * These are physical boundary conditions, and are not to be confused
     * with ghost cells that we would use in an MPI implementation
     *
     * The reason why we copy boundary conditions is to avoid
     * computing single sided differences at the boundaries
     * which increase the running time of solve()
     *
     */

    // 4 FOR LOOPS set up the padding needed for the boundary conditions

    // Fills in the LEFT Ghost Cells
    if (proc_col == 0 ){//already has left ghost cells
      for (index = 0;
           index < submesh_size;
           index+=(submesh_col_dim)) {
        E_prev[index] = E_prev[index+2];
      }
    } else{//LEFT ghost cells received
      for (index = 0,buf_ind = 0;
           index < submesh_size;
           index+=(submesh_col_dim), ++buf_ind) {
        E_prev[index] = recv_left_ghost[buf_ind];//left's "send_right"
      }
    }

    // Fills in the RIGHT Ghost Cells
    /* RIGHT stencil point == LEFT stencil point*/
    if (proc_col == proc_col_dim-1){//already has right ghost cells
      for (index = (submesh_col_dim-1);
           index < submesh_size;
           index+=(submesh_col_dim)) {
        E_prev[index] = E_prev[index-2];
      }
    } else{//RIGHT ghost cells received
      for (index = (submesh_col_dim-1),buf_ind = 0;
           index < submesh_size;
           index+=(submesh_col_dim), ++buf_ind) {
        E_prev[index] = recv_right_ghost[buf_ind];
      }
    }

    // Fills in the TOP Ghost Cells
    /* Top stencil point == bottom stencil point*/
    if (proc_row == 0){//already has top ghost cells
      for (index = 0;
           index < (submesh_col_dim);
           index++) {
        E_prev[index] = E_prev[index + (submesh_col_dim)*2];
      }
    }//else, TOP ghost cells received (already in E_prev)


    // Fills in the BOTTOM Ghost Cells
    if (proc_row == proc_row_dim-1){//already has bottom ghost cells
      for (index = submesh_size-(submesh_col_dim);
           index < submesh_size;
           index++) {
        E_prev[index] = E_prev[index - (submesh_col_dim)*2];
      }
    }//else, BOTTOM ghost cells received (already in E_prev)


/////*END filling in local ghost cells*////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////

//#define FUSED 1


    //NOTE: intel compiler may vectorize loop better (i.e. AVX)
#ifdef FUSED
    // Solve for the excitation, a PDE

    //unroll and jam
    for(inner_block_row = innerBlockRowStartIndex;
        inner_block_row <= innerBlockRowEndIndex;
        inner_block_row +=(submesh_col_dim)) {
        E_prev_tmp = E_prev + inner_block_row;
        E_tmp = E + inner_block_row;
        E_prev_tmp_bottom = E_prev + inner_block_row - submesh_col_dim;
        E_prev_tmp_top = E_prev + inner_block_row + submesh_col_dim;
        R_tmp = R + inner_block_row;
            for (index = 0; index < submesh_col_comp_dim; index++) {//index is first comp box,
                e_prev_tmp_prev = E_prev_tmp[index-1];
                e_prev_tmp_curr = E_prev_tmp[index];
                e_prev_tmp_next = E_prev_tmp[index + 1];
                r_tmp = R_tmp[index];

              //PDE solver
//
//                E_tmp[index] = e_prev_tmp_curr +
//                               alpha * (e_prev_tmp_next//right
//                                        + E_prev_tmp[index - 1]//left
//                                        - 4 * e_prev_tmp_curr//curr
//                                        + E_prev_tmp_top[index]//top
//                                        + E_prev_tmp_bottom[index]);//bottom

                e_prev_tmp_kk = kk*e_prev_tmp_curr;


                //ODE solver
//                E_tmp[index] += -dt*(e_prev_tmp_kk*(e_prev_tmp_curr-a)*(e_prev_tmp_curr-1)
//                                     +e_prev_tmp_curr*r_tmp);
                E_tmp[index] = (-dt*(e_prev_tmp_kk*(e_prev_tmp_curr-a)*(e_prev_tmp_curr-1)
                                     +e_prev_tmp_curr*r_tmp)) + (e_prev_tmp_curr +
                                                                alpha * (e_prev_tmp_next//right
                                                                         + e_prev_tmp_prev//left
                                                                         - 4 * e_prev_tmp_curr//curr
                                                                         + E_prev_tmp_top[index]//top
                                                                         + E_prev_tmp_bottom[index]));//bottom;
                R_tmp[index] += dt*(epsilon+M1* r_tmp/( e_prev_tmp_curr+M2))
                                *(-r_tmp-e_prev_tmp_kk*(e_prev_tmp_curr-b-1));
            }
        }

#else
      for(inner_block_row = innerBlockRowStartIndex;
          inner_block_row <= innerBlockRowEndIndex;
          inner_block_row+=(submesh_col_dim)) {
          E_tmp = E + inner_block_row;
          E_prev_tmp = E_prev + inner_block_row;

          E_prev_tmp_bottom = E_prev + inner_block_row - submesh_col_dim;
          E_prev_tmp_top = E_prev + inner_block_row + submesh_col_dim;

          R_tmp = R + inner_block_row;


          for(index = 0; index < submesh_col_comp_dim; index++) {
              e_prev_tmp_prev = E_prev_tmp[index-1];
              e_prev_tmp_curr = E_prev_tmp[index];
              e_prev_tmp_next = E_prev_tmp[index + 1];

                E_tmp[index] = e_prev_tmp_curr +
               alpha * (e_prev_tmp_next//right
                        + e_prev_tmp_prev//left
                        - 4 * e_prev_tmp_curr//curr
                        + E_prev_tmp_top[index]//top
                        + E_prev_tmp_bottom[index]);//bottom
          }
/*
       * Solve the ODE, advancing excitation and recovery variables
       *     to the next timtestep
       */

          for(index = 0; index < submesh_col_comp_dim; index++) {
              e_prev_tmp_curr = E_prev_tmp[index];
              r_tmp = R_tmp[index];
              e_prev_tmp_kk = kk*e_prev_tmp_curr;

              E_tmp[index] += -dt*(e_prev_tmp_kk*(e_prev_tmp_curr-a)*(e_prev_tmp_curr-1)
                                     +e_prev_tmp_curr*r_tmp);

              R_tmp[index] += dt*(epsilon+M1* r_tmp/( e_prev_tmp_curr+M2))
                              *(-r_tmp-e_prev_tmp_kk*(e_prev_tmp_curr-b-1));

          }
      }

#endif
    /////////////////////////////////////////////////////////////////////////////////

    /*Update Stats*/
    if (cb.stats_freq){
      if ( !(niter % cb.stats_freq)){
        stats(E,submesh_row_comp_dim,submesh_col_comp_dim,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,submesh_row_comp_dim,submesh_col_comp_dim,niter, cb.stats_freq);
      }
    }


    /*Plot data*/
    if (cb.plot_freq){
      if (!(niter % cb.plot_freq)){
        plotter->updatePlot(E,  niter, submesh_row_comp_dim, submesh_col_comp_dim);
      }
    }

    // Swap current and previous meshes
    double *tmp = E; E = E_prev; E_prev = tmp;

  } //end of 'niter' loop at the beginning


  // return the L2 and infinity norms via in-out parameters
  stats(E_prev,submesh_row_comp_dim,submesh_col_comp_dim,&Linf,&sumSq);

  if (noComm) {//do nothing
  }else{
    /*return cumulative results*/
    double _Linf, _sumSq;
    MPI_Reduce(&Linf, &_Linf, 1, MPI_DOUBLE, MPI_MAX, root_proc, MPI_COMM_WORLD);
    MPI_Reduce(&sumSq, &_sumSq, 1, MPI_DOUBLE, MPI_SUM, root_proc, MPI_COMM_WORLD);
    Linf = _Linf;
    sumSq = _sumSq;
  }

  L2 = L2Norm(sumSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {
#ifdef _MPI_
  solve_MPI(_E ,_E_prev, R, alpha, dt, plotter, L2, Linf);
#else
  solve_single_proc(_E ,_E_prev, R, alpha, dt, plotter, L2, Linf);
#endif
}
