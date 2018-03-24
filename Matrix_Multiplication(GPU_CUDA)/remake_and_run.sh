#!/usr/bin/env bash
git pull origin master
make clean
make bx=32 by=32


JOB_NUM=$(sbatch ./single_GPU.scr |grep -Eo '[0-9]{1,9}')
string='#!/usr/bin/env bash \ncat '
string=$string"./MMPY-CUDA.o$JOB_NUM"
string=$string' | sed -n -e '\''/printenv/,$p'\'
echo -e $string > ./print_last_test.sh
chmod a+x ./print_last_test.sh
echo "Job ("$JOB_NUM") started"
echo "run [./print_last_test.sh] for output (note: results will be appended as job progresses)"