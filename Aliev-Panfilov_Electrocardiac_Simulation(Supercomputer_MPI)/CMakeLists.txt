cmake_minimum_required(VERSION 3.9)
project(pa3_pa3_tzosa)

set(CMAKE_CXX_STANDARD 11)

add_executable(pa3_pa3_tzosa
        apf.cpp
        apf.h
        cblock.h
        cmdLine.cpp
        helper.cpp
        Plotting.cpp
        Plotting.h
        Report.cpp
        solve.cpp
        Timer.cpp
        utils.cpp)
