cmake_minimum_required(VERSION 3.9)
project(pa2_tzosa)

set(CMAKE_CXX_STANDARD 11)

add_executable(pa2_tzosa
        batch-stampede.scr
        cmdLine.cpp
        DECLARATION
        genMatrix.cpp
        Makefile
        mmpy.cu
        mmpy_host.cpp
        mmpy_kernel.cu
        README
        remake_and_run.sh
        Report.cpp
        run.sh
        setGrid.cu
        single_GPU.scr
        teameval.txt
        Timer.cpp
        types.h
        utils.cu
        utils.h)
