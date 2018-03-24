#!/bin/bash
#### this a bash script which we write out and then submit to the batch queue
# This script is intepreted by the Bourne Shell, sh
#
#SBATCH --account=csd562
#SBATCH --job-name=48_18k
#SBATCH --output="48_18k.out"
#SBATCH --partition="compute"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu
#SBATCH -t 00:03:00

# Commands go here
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 1 -y 48 -k
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 2 -y 24 -k
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 4 -y 12 -k
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 6 -y 8 -k
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 8 -y 6 -k
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 12 -y 4 -k
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 24 -y 2 -k
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 48 -y 1 -k


ibrun -np 48 ./apf -n 1800 -i 2000 -x 8 -y 6
echo ">>> __APF_NOCOMM__"
ibrun -np 48 ./apf -n 1800 -i 2000 -x 8 -y 6 -k


echo ">>> __APF-REF__"



#ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 1 -y 48 -k
#ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 2 -y 24 -k
#ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 4 -y 12 -k
#ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 6 -y 8 -k
#ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 8 -y 6 -k
#ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 12 -y 4 -k
#ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 24 -y 2 -k
#ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 48 -y 1 -k


ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 8 -y 6
echo ">>> __NOCOMM APF-REF__"

ibrun -np 48 ./apf-ref/apf -n 1800 -i 2000 -x 8 -y 6 -k

