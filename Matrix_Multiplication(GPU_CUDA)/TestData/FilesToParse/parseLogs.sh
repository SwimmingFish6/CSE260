#!/usr/bin/env bash


find ./ -name "MMPY-CUDA*" -print | while read f; do
    grep -E 'Device computation|@' "$f" | sed 'N;s/.*\[\([0-9][0-9]*\.[0-9][0-9]*\).*\n@[^0-9]*\([0-9][0-9][0-9]*\).*/\1, /g'  > ./parsed/"$f".Parsed
   # grep -E 'gflops|@' "$f"  > ./parsed/"$f".Parsed
    #sed 'N;s/.*\[\([0-9][0-9]*\.[0-9][0-9]*\).*\n@[^0-9]*\([0-9][0-9][0-9]*\).*/\1	\2 /g' ./parsed/"$f".Parsed 

done