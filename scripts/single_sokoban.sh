#!/bin/bash

case $(expr $1 + 1) in
    1 )
        echo "Solving sokoban example e2d_0_0..."
        time code/solve sokoban e2d_0_0
        ;;
    2 )
        echo "Solving sokoban example e2d_1_0..."
        time code/solve sokoban e2d_1_0
        ;;
    3 )
        echo "Solving sokoban example e2d_2_0..."
        time code/solve sokoban e2d_2_0
        ;;
esac
