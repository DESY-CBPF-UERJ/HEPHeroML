#!/bin/bash
import os
import sys
import argparse


#======GET SETUP FILE==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--analysis", type=str, default="GEN")
args = parser.parse_args()
   
sys.path.insert(0, 'setups')
sm = __import__(args.analysis)

trainer_name = 'train_' + args.analysis + '_' + sm.tag + '.py'

print('Generate ' + trainer_name)
print('')
   

#======CREATE NEW SETUP============================================================================
with open(trainer_name, "w") as newfile:
    newfile.write("import sys\n")
    newfile.write("import numpy as np\n")
    newfile.write("import pandas as pd\n")
    newfile.write("import os\n")
    newfile.write("import time\n")
    newfile.write("import concurrent.futures as cf\n")
    newfile.write("import argparse\n")
    newfile.write("from statsmodels.stats.weightstats import DescrStatsW\n")
    newfile.write("import matplotlib.pyplot as plt\n")
    newfile.write("import matplotlib.gridspec as gs\n")
    newfile.write("from matplotlib.ticker import AutoMinorLocator\n")
    newfile.write("import json\n")
    newfile.write("from functions import read_files\n")
    newfile.write("from functions import join_datasets\n")
    newfile.write("from functions import train_model\n")
    newfile.write("from functions import step_plot\n")
    newfile.write("import functions as func\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# General Setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("selection = '" + sm.selection +"'\n")
    newfile.write("analysis = '" + sm.analysis +"'\n")
    newfile.write("periods = " + str(sm.periods) +"\n")
    newfile.write("tag = '" + sm.tag +"'\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Models setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("library = '" + sm.library +"'\n")
    newfile.write("NN_type = " + str(sm.NN_type) +"\n")
    newfile.write("num_layers = " + str(sm.num_layers) +"\n")
    newfile.write("num_nodes = " + str(sm.num_nodes) +"\n")
    newfile.write("activation_func = " + str(sm.activation_func) +"\n")
    newfile.write("optimizer = " + str(sm.optimizer) +"\n")
    newfile.write("loss_func = " + str(sm.loss_func) +"\n")
    newfile.write("learning_rate = " + str(sm.learning_rate) +"\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Training setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("batch_size = " + str(sm.batch_size) +"\n")
    newfile.write("load_size = " + str(sm.load_size) +"\n")
    newfile.write("train_frac = " + str(sm.train_frac) +"\n")
    newfile.write("eval_step_size = " + str(sm.eval_step_size) +"\n")
    newfile.write("num_max_iterations = " + str(sm.num_max_iterations) +"\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Inputs setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("input_mode = '" + sm.input_mode +"'\n")
    newfile.write("feature_info = " + str(sm.feature_info) +"\n")
    newfile.write("\n")
    newfile.write("input_variables = " + str(sm.input_variables) +"\n")
    newfile.write("\n")
    newfile.write("input_parameters = " + str(sm.input_parameters) +"\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("# Classes setup\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")
    newfile.write("classes = " + str(sm.classes) +"\n")
    newfile.write("\n")
    newfile.write("\n")
    newfile.write("#-------------------------------------------------------------------------------------\n")


#======ADD FIXED PART==============================================================================
linelist = open("setups/config_train.py").readlines()

with open(trainer_name, "a") as newfile:
    flag = 0
    for line in linelist:
        if line.startswith("# [DO NOT TOUCH THIS PART]"):
            flag = 1
        if flag:
            newfile.write(line)
        
