#!/bin/bash

echo "ProcId"
echo $1
echo "HEPML_path"
echo $2
echo "HEP_OUTPATH"
echo $3
echo "MACHINES"
echo $4
echo "SIGNAL"
echo $5
echo "MODE"
echo $6

export HEP_OUTPATH=$3
export MACHINES=$4

if [ "$4" == "CERN" ]; then
cp -r $2 .
fi    
    
if [ "$4" == "DESY" ]; then
cd ..
fi

ls
source /cvmfs/sft.cern.ch/lcg/views/LCG_100/x86_64-centos7-gcc9-opt/setup.sh
python -m venv hepenv
source hepenv/bin/activate
cd HEPML
 
python train.py -j $1 -s $5 -m $6

