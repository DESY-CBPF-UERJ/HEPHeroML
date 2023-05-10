#!/bin/bash

flavor=$1
N_models=$2

#listMode="keras torch"
listMode="torch"

for j in $listMode; do
    #./submit_jobs.sh ${flavor} ${N_models} Signal_parameterized "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_all "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_all_xsec "$j"
    #./submit_jobs.sh ${flavor} ${N_models} Signal_one_relevant "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_1000_100 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_1000_200 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_1000_300 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_1000_400 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_1000_600 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_1000_800 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_400_100 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_400_200 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_500_100 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_500_200 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_500_300 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_600_100 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_600_300 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_600_400 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_800_100 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_800_200 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_800_300 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_800_600 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_600_200 "$j"
    ./submit_jobs.sh ${flavor} ${N_models} Signal_800_400 "$j"
done

