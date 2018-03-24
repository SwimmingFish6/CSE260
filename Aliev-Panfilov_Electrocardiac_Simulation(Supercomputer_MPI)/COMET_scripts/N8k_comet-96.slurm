#!/bin/bash
#### this a bash script which we write out and then submit to the batch queue
# This script is intepreted by the Bourne Shell, sh
#
#SBATCH --account=csd562
#SBATCH --job-name=96_8k
#SBATCH --output="96_8k.out"
#SBATCH --partition="compute"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu
#SBATCH -t 00:03:00
# Commands go here

#ibrun -np 96 ./apf -n 8000 -i 2000 -x 1 -y 96
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 2 -y 48
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 4 -y 24
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 8 -y 12
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 16 -y 6

#ibrun -np 96 ./apf -n 8000 -i 2000 -x 32 -y 3




echo ">>> __APF-REF__"

ibrun -np 96 ./apf-ref/apf -n 8000 -i 2000 -x 1 -y 96
ibrun -np 96 ./apf-ref/apf -n 8000 -i 2000 -x 2 -y 48
ibrun -np 96 ./apf-ref/apf -n 8000 -i 2000 -x 4 -y 24
ibrun -np 96 ./apf-ref/apf -n 8000 -i 2000 -x 8 -y 12
ibrun -np 96 ./apf-ref/apf -n 8000 -i 2000 -x 16 -y 6
#ibrun -np 96 ./apf-ref/apf -n 8000 -i 2000 -x 32 -y 3
