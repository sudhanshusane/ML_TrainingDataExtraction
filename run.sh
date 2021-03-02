#!/bin/bash

#FILEPATH="/home/sci/ssane/data/ML_Gerris/"
FILEPATH="/home/sci/ssane/data/DoubleGyre/"
FILENO="1"
EXT=".vtk"
SLICE=0
OUTPUT="extract_"$SLICE
END_TIME=3


DIMX=256
DIMY=128
DIMZ=1001


./ExtractTrainingData $FILEPATH $FILENO $EXT $OUTPUT $SLICE $END_TIME $DIMX $DIMY $DIMZ
