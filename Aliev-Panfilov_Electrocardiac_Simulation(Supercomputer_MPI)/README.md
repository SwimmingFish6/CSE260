# Scaling Aliev-Panfilov Electro-Cardiac Simulation Using OpenMPI

Optimized the Aliev-Panfilov
electro-cardiac simulation algorithm using OpenMPI to run on the COMET
supercomputer at the San Diego Super Computer Center. This program achieves a 
scaling efficiency > 1.0 and peak performance of 1.82 Tflops/s using 480 cores via the use of
asynchronous non-blocking communication and the exploitation of spatial and temporal locality 
inherent to computed elements.