#!/usr/bin/env bash
git pull origin master
make clean
make MY_OPT="-O4 -ffast-math -msse3 -mfpmath=sse -ftree-vectorize -funroll-loops -funroll-all-loops -DTEOZOSA_COMPILER_FLAGS"
JOB_NUM=$(qsub ./batch.sh|grep -Eo '[0-9]{1,9}')
string='#!/usr/bin/env bash \ncat '
string=$string"./DGEMM.o$JOB_NUM"
string=$string' | sed -n -e '\''/>>>/,$p'\'
echo -e $string > ./print_last_test.sh
chmod a+x ./print_last_test.sh
echo "Job ("$JOB_NUM") started"
echo "run [./print_last_test.sh] for output (note: results will be appended as job progresses)"
