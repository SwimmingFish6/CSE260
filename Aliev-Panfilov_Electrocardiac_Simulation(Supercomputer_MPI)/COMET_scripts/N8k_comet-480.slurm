#!/bin/bash
#### this a bash script which we write out and then submit to the batch queue
# This script is intepreted by the Bourne Shell, sh
#
#SBATCH --account=csd562
#SBATCH --job-name=480_8k
#SBATCH --output="480_8k.out"
#SBATCH --partition="compute"
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu
#SBATCH -t 00:03:00
# Commands go here

ibrun -np 480 ./apf -n 8000 -i 2000 -x 8 -y 60
ibrun -np 480 ./apf -n 8000 -i 2000 -x 16 -y 30
ibrun -np 480 ./apf -n 8000 -i 2000 -x 20 -y 24
ibrun -np 480 ./apf -n 8000 -i 2000 -x 32 -y 15
ibrun -np 480 ./apf -n 8000 -i 2000 -x 40 -y 12


#echo ">>> __APF-REF__"

##ibrun -np 480 ./apf-ref/apf -n 8000 -i 2000 -x 4 -y 70
#ibrun -np 480 ./apf-ref/apf -n 8000 -i 2000 -x 8 -y 60
#ibrun -np 480 ./apf-ref/apf -n 8000 -i 2000 -x 16 -y 30
#ibrun -np 480 ./apf-ref/apf -n 8000 -i 2000 -x 20 -y 24
##ibrun -np 480 ./apf-ref/apf -n 8000 -i 2000 -x 24 -y 20
#ibrun -np 480 ./apf-ref/apf -n 8000 -i 2000 -x 32 -y 15
#ibrun -np 480 ./apf-ref/apf -n 8000 -i 2000 -x 40 -y 12



