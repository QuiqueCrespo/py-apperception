#!/bin/bash

case $(expr $1 + 1) in
    1 )
        echo "Solving sokoban example e2d_0_0..."
        time python3 code/solve.py sokoban e2d_0_0
        ;;
    2 )
        echo "Solving sokoban example e2d_1_0..."
        time python3 code/solve.py sokoban e2d_1_0
        ;;
    3 )
        echo "Solving sokoban example e2d_2_0..."
        time python3 code/solve.py sokoban e2d_2_0
        ;;
esac
