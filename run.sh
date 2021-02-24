#!/bin/bash

FILEPATH="/home/sci/ssane/data/ML_Gerris/"
FILENO="2000"
EXT=".vtk"
SLICE=0
OUTPUT="extract_"$SLICE
END_TIME=5

./ExtractTrainingData $FILEPATH $FILENO $EXT $OUTPUT $SLICE $END_TIME
