#!/usr/bin/env bash

mkdir -p boxoutput
echo 'Stage 1, extract ROIs'
python reduce.py
echo 'Stage 2, classify ROIs'
python classify.py  -W ignore #Tensorflow may raise some warnings if you arent using GPU


#boxoutput is a temporary file where all the ROIs are stored
#saving ROIs will make a new directory and save boxoutput and CoordsROI.txt
read -r -p "Would you like to save ROIs [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    echo 'name of new directory'
    read dname
    mkdir "$dname"
    cp -r boxoutput $dname
    mv CoordsROI.txt $dname
    cd boxoutput
    rm *
    cd ..

else
    cd boxoutput
    rm *
    cd ..
fi

rm CoordsROI.txt
