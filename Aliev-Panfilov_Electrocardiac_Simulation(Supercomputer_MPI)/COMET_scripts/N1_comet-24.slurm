#!/bin/bash
#### this a bash script which we write out and then submit to the batch queue
# This script is intepreted by the Bourne Shell, sh
#
#SBATCH --account=csd562
#SBATCH --job-name=24_18k
#SBATCH --output="24_18k.out"
#SBATCH --partition="compute"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu
#SBATCH -t 00:03:00
# Commands go here
ibrun -np 24 ./apf -n 1800 -i 2000 -x 6 -y 4
echo ">>> __APF_NOCOMM__"

ibrun -np 24 ./apf -n 1800 -i 2000 -x 6 -y 4 -k
#
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 1 -y 24 -k
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 2 -y 12 -k
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 4 -y 6 -k
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 6 -y 4 -k
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 8 -y 3 -k
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 12 -y 2 -k
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 24 -y 1 -k



echo ">>> __APF-REF__"
ibrun -np 24 ./apf-ref/apf -n 1800 -i 2000 -x 6 -y 4

echo ">>> __NOCOMM APF-REF__"
ibrun -np 24 ./apf-ref/apf -n 1800 -i 2000 -x 6 -y 4 -k
#
#ibrun -np 24 ./apf-ref/apf -n 1800 -i 2000 -x 1 -y 24 -k
#ibrun -np 24 ./apf-ref/apf -n 1800 -i 2000 -x 2 -y 12 -k
#ibrun -np 24 ./apf-ref/apf -n 1800 -i 2000 -x 4 -y 6 -k
#ibrun -np 24 ./apf-ref/apf -n 1800 -i 2000 -x 6 -y 4 -k
#ibrun -np 24 ./apf-ref/apf -n 1800 -i 2000 -x 8 -y 3 -k
#ibrun -np 24 ./apf-ref/apf -n 1800 -i 2000 -x 12 -y 2 -k
#ibrun -np 24 ./apf-ref/apf -n 1800 -i 2000 -x 24 -y 1 -k
#
