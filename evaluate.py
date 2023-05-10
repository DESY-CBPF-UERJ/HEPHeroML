import numpy as np
import pandas as pd
import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--period", default="17")
parser.add_argument("-s", "--selection", default="ML")
parser.add_argument("--check", dest='check_flag', action='store_true')
parser.set_defaults(check_flag=False)
args = parser.parse_args()


analysis = "HHDM"
selection = args.selection
period = args.period
directory = "datasets_DeepJet"

outpath = os.environ.get("HEP_OUTPATH")

outpath_base = os.path.join(outpath, analysis, selection, directory)

best_models_path = os.path.join(outpath_base, period, "ML", "best_models")
if not os.path.exists(best_models_path):
    os.makedirs(best_models_path)

#list_modes = [ "keras", "torch" ]
list_modes = [ "torch" ]
#list_loss_func = [ "bce", "cce" ] 
list_loss_func = [ "bce" ]

list_signals = []
for mode in list_modes:
    list_signals_temp = os.listdir(os.path.join(outpath_base, period, "ML", mode))
    if len(list_signals_temp) > len(list_signals):
        list_signals = list_signals_temp

torch_bce_list = np.zeros(len(list_signals))
torch_cce_list = np.zeros(len(list_signals))
keras_bce_list = np.zeros(len(list_signals))
keras_cce_list = np.zeros(len(list_signals))

for mode in list_modes:
    print("#########################################################################################")
    print(mode)
    print("#########################################################################################")
    
    ml_outpath = os.path.join(outpath_base, period, "ML", mode)
    os.system("rm -rf " + os.path.join(best_models_path, mode))
    print("outpath = ", ml_outpath)
    
    for loss in list_loss_func:
        list_best_models = []
        for signal in list_signals:
            #print(signal)
            list_models = os.listdir(os.path.join(ml_outpath, signal, "models"))
            if mode == "keras":
                list_models.remove('preprocessing.json')
            models_loss = []
            models_accuracy = []
            models_iterations = []
            models_name = []
            for model in list_models:
                #print(model)
                if loss in model:
                    training_file = os.path.join(ml_outpath, signal, "models", model, "training.csv")
                    if os.path.isfile(training_file):
                        df_training = pd.read_csv(training_file)
                        if len(df_training) > 0 and len(df_training["test_loss"]) > 0 and len(df_training["test_acc"]) > 0:
                            min_loss = np.amin(df_training["test_loss"])
                            if not math.isnan(min_loss):
                                models_loss.append(min_loss)
                                models_accuracy.append(np.array(df_training[df_training["test_loss"] == min_loss]["test_acc"])[-1])
                                models_iterations.append(np.array(df_training[df_training["test_loss"] == min_loss]["iteration"])[-1])
                                models_name.append(model)
            df_training = pd.DataFrame({"Model": models_name, "Loss": models_loss, "Accuracy": models_accuracy, "Iterations": models_iterations})
            df_training = df_training.sort_values("Loss")
            df_training = df_training.reset_index()
            
            #best_model_dir = os.path.join(ml_outpath, signal, "models", df_training.loc[0]["Model"])
            #signal_dir = os.path.join(ml_outpath, signal)
            #copyCommand = "cp -rf " + best_model_dir + " " + signal_dir
            #os.system(copyCommand)
            #print(signal)
            #print("XXXXXXXXXXXXXXXXXXXXXXXXXX")
            #print(df_training)
            
            list_best_models.append(df_training.loc[0]["Model"])
            
            pd.set_option("display.precision", 15)
            print("============================================================================================================")
            print(signal)
            print("============================================================================================================")
            print(df_training)
            print("")
            save_path = os.path.join(best_models_path, mode, loss, signal)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df_training.to_csv(os.path.join(save_path, "training_result.csv"))
            
            os.system("cp -rf " + os.path.join(ml_outpath, signal, 'features') + " " + save_path)
            
            models_path = os.path.join(save_path, 'models')
            if not os.path.exists(models_path):
                os.makedirs(models_path)
            if mode == "keras":    
                copyCommand = "cp -rf " + os.path.join(ml_outpath, signal, 'models', 'preprocessing.json') + " " + models_path
                os.system(copyCommand)
            if len(df_training) <= 3:
                for model in list_models:
                    copyCommand = "cp -rf " + os.path.join(ml_outpath, signal, 'models', model) + " " + models_path
                    os.system(copyCommand)
            else:
                for model in list_models:
                    if( model == df_training.loc[0]["Model"] or model == df_training.loc[1]["Model"] or model == df_training.loc[2]["Model"]):
                        copyCommand = "cp -rf " + os.path.join(ml_outpath, signal, 'models', model) + " " + models_path
                        os.system(copyCommand)
        
        model_type = mode + "_" + loss
        
        if model_type == "torch_bce":
            torch_bce_list = list_best_models
        elif model_type == "torch_cce":
            torch_cce_list = list_best_models
        elif model_type == "keras_bce":
            keras_bce_list = list_best_models
        elif model_type == "keras_cce":
            keras_cce_list = list_best_models
    
    
df_result = pd.DataFrame({"signal": list_signals, "torch_bce": torch_bce_list, "torch_cce": torch_cce_list, "keras_bce": keras_bce_list, "keras_cce": keras_cce_list})
df_result = df_result.reset_index()
df_result.to_csv(os.path.join(best_models_path, "best_models.csv"))






