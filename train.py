import sys
import numpy as np
import pandas as pd
import os
import time
import concurrent.futures as cf
import argparse
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.ticker import AutoMinorLocator
import json
from functions import read_files
from functions import join_datasets
from functions import train_model
from functions import step_plot
import functions as func


parser = argparse.ArgumentParser()
parser.add_argument("-j", "--job", type=int, default=0)
parser.add_argument("-s", "--signal", default="Signal_1000_100")
parser.add_argument("-m", "--mode", default="torch")
parser.add_argument("--check", dest='check_flag', action='store_true')
parser.set_defaults(check_flag=False)
parser.add_argument("--clean", dest='clean_flag', action='store_true')
parser.set_defaults(clean_flag=False)

args = parser.parse_args()
outpath = os.environ.get("HEP_OUTPATH")

#=============================DATASETS=============================================================
analysis = "HHDM"
selection = "ML"
directory = "datasets_DeepJet"
period = '16'

outpath_base = os.path.join(outpath, analysis, selection, directory)


input_variables = [
    ["LeadingLep_pt",           r"$\mathrm{leading}\,p_\mathrm{T}^\mathrm{l}$"], 
    ["LepLep_pt",               r"$p_\mathrm{T}^\mathrm{ll}$"], 
    ["LepLep_deltaR",           r"$\Delta R^\mathrm{ll}$"], 
    ["LepLep_deltaM",           r"$\Delta M^\mathrm{ll}$"], 
    ["MET_pt",                  r"$E_\mathrm{T}^\mathrm{miss}$"], 
    ["MET_LepLep_Mt",           r"$M_\mathrm{T}^\mathrm{ll,MET}$"], 
    ["MET_LepLep_deltaPhi",     r"$\Delta \phi^\mathrm{ll,MET}$"],
    ["TrailingLep_pt",          r"$\mathrm{trailing}\,p_\mathrm{T}^\mathrm{l}$"],
    ["MT2LL",                   r"$M_\mathrm{T2}^\mathrm{ll}$"],
    #["Nbjets",                  r"$N_\mathrm{b\,jets}$"],
    #["Njets_forward",           r"Njets_forward"],
    #["Dijet_deltaEta",          r"Dijet_deltaEta"],
    #["HT30",                    r"HT30"],
    #["MHT30",                   r"MHT30"],
    #["OmegaMin30",              r"OmegaMin30"],
    #["FMax30",                  r"FMax30"],
    ]

input_parameters = [
    ["m_H", r"$m_H$"], 
    ["m_a", r"$m_a$"],
    ]


#--------------------------------------------------------------------------------------------------

signal_list = [ 
    "Signal_1000_100", 
    "Signal_1000_200", 
    "Signal_1000_300", 
    "Signal_1000_400", 
    "Signal_1000_600", 
    "Signal_1000_800", 
    "Signal_400_100", 
    "Signal_400_200", 
    "Signal_500_100", 
    "Signal_500_200", 
    "Signal_500_300", 
    "Signal_600_100",
    "Signal_600_200",
    "Signal_600_300", 
    "Signal_600_400", 
    "Signal_800_100", 
    "Signal_800_200", 
    "Signal_800_300", 
    "Signal_800_400",
    "Signal_800_600", 
    ] 

backgrounds = [
    "DYJetsToLL_Pt-0To3",
    "DYJetsToLL_PtZ-3To50",
    "DYJetsToLL_PtZ-50To100",
    "DYJetsToLL_PtZ-100To250",
    "DYJetsToLL_PtZ-250To400",
    "DYJetsToLL_PtZ-400To650",
    "DYJetsToLL_PtZ-650ToInf",    
    "TTTo2L2Nu", 
    "TTToSemiLeptonic",
    "ST_tW_antitop", 
    "ST_tW_top", 
    #"ST_s-channel",
    "ST_t-channel_top", 
    "ST_t-channel_antitop",
    "WZTo3LNu",
    "WZ_Others",
    "ZZTo4L",
    "ZZTo2L2Nu",
    "ZZ_Others",
    "WW",
    "WZZ", 
    "WWZ", 
    "ZZZ", 
    "WWW", 
    "TTWZ",
    "TTZZ",
    "TTWW",
    "TWZToLL_thad_Wlept",
    #"TWZToLL_tlept_Whad",
    "TWZToLL_tlept_Wlept",
    "TTWJetsToLNu",
    "TTWJetsToQQ",
    "TTZToQQ",
    "TTZToLL",
    #"TTZToNuNu",
    "tZq_ll",
    "ttHTobb", 
    #"ttH_HToZZ",
    #"ttHToTauTau",
    #"GluGluHToWWTo2L2Nu", 
    #"GluGluHToZZTo4L",
    "WplusH_HToZZTo4L", 
    #"WminusH_HToZZTo4L",
    "ZH_HToBB_ZToLL", 
    #"ZH_HToZZ",
    ]


#=============================MODELS AND TRAINING==================================================
train_frac = 0.5
eval_frac = 1
num_max_iterations = 2000


NN_type = [ "NN" ]  # [ "NN", "DANN", "PNN" ]
num_layers = [ 2, 3, 4 ]
num_nodes = [ 20, 100, 500, 1000 ] #, 500, 1000, 2000, 5000 ]
activation_func = [ "elu" ] # [ "elu", "relu", "tanh" ]
optimizer = [ "adam", "sgd" ] # [ "adam", "sgd" ]
loss_func = [ "bce" ] # [ "bce", "cce" ]
batch_size = [ 1000 ] # [ 100, 1000 ]
learning_rate = [ 0.1, 0.01 ] #, 0.001 ]





#--------------------------------------------------------------------------------------------------------------------------------------------------
# [DO NOT TOUCH THIS PART] 
#--------------------------------------------------------------------------------------------------------------------------------------------------

outpath = os.environ.get("HEP_OUTPATH")
if args.clean_flag:
    os.system("rm -rf " + os.path.join(outpath_base, period, "ML", args.mode, args.signal))
    sys.exit()


#=====================================================================================================================
# CHECK ARGUMENT
#=====================================================================================================================
#inpath = "files"

modelName = [] 
model = []
for i_NN_type in NN_type:
    for i_num_layers in num_layers:
        for i_num_nodes in num_nodes:
            for i_activation_func in activation_func:
                for i_optimizer in optimizer:
                    for i_loss_func in loss_func:
                        for i_batch_size in batch_size:
                            for i_lr in learning_rate:
                                modelName.append(i_NN_type+"_"+str(i_num_layers)+"_"+str(i_num_nodes)+"_"+i_activation_func+"_"+i_optimizer+"_"+i_loss_func+"_"+str(i_batch_size)+"_"+str(i_lr).replace(".", "p"))
                                model.append([i_NN_type] + [[i_num_nodes for i in range(i_num_layers)]] + [i_activation_func] + [i_optimizer] + [i_loss_func] + [i_batch_size] + [i_lr])

N = int(args.job)
if N == -1:
    print("")
    sys.exit("Number of jobs: " + str(len(model)))
if N == -2:
    for i in range(len(model)):
        print(str(i)+"  "+str(model[i])+",")
    sys.exit("")
if N <= -3:
    sys.exit(">> Enter an integer >= -1")
if N >= len(model):
    sys.exit("There are only " + str(len(model)) + " models")


#=====================================================================================================================
import torch   
                           
    
variables = [input_variables[i][0] for i in range(len(input_variables))]
var_names = [input_variables[i][1] for i in range(len(input_variables))]

signal_parameters = [input_parameters[i][0] for i in range(len(input_parameters))]
signal_parameters_names = [input_parameters[i][1] for i in range(len(input_parameters))] 


if args.signal == "Signal_parameterized":
    if len(signal_parameters) > 2:
        sys.exit("Code does not support more than 2 signal parameters!")
        # It can be extended for more than 2 variables
    else:
        variables = variables + signal_parameters
        var_names = var_names + signal_parameters_names

n_var = len(variables)


#=============================CLASSES AND INPUT VARIABLES==========================================     [THIS PART MUST BE CHANGED IF I USE MORE THAN 2 CLASSES]
classes = [args.signal, "Bkg"]    
class_names = [args.signal, 'Background']
colors = ['green', 'red']

# Example with more than 2 classes:
#classes = [args.signal, "DYJetsToLL", "TT", "Others"]
#class_names = [args.signal, 'Drell-Yan', r'$t\bar{t}$', "Others"]
#colors = ['green', 'darkgoldenrod', 'skyblue', 'grey']


#=====================================================================================================================
# Define signal and backgrounds datasets
#=====================================================================================================================

datasets = read_files(outpath_base, period, features=variables+["evtWeight"])

if args.signal == "Signal_parameterized":
    for dataset in datasets:
        if dataset not in signal_list:
            datasets[dataset]["m_H"] = 600  # dumb number
            datasets[dataset]["m_a"] = 300  # dumb number
    
    for sgn in signal_list:
        signal_info = sgn.split("_")
        datasets[sgn]["m_H"] = int(signal_info[1])
        datasets[sgn]["m_a"] = int(signal_info[2])
        datasets[sgn].loc[:,"evtWeight"] = datasets[sgn]["evtWeight"]/datasets[sgn]["evtWeight"].sum()
    signal_list.remove("Signal_500_200")                                                                            # hard code
    signal_list.remove("Signal_800_400")                                                                            # hard code
    print(signal_list)

    join_datasets(datasets, "Signal_parameterized", signal_list)
    
elif args.signal == "Signal_all":
    for sgn in signal_list:
        datasets[sgn].loc[:,"evtWeight"] = datasets[sgn]["evtWeight"]/datasets[sgn]["evtWeight"].sum()
    #signal_list.remove("Signal_500_200")                                                                            # hard code
    #signal_list.remove("Signal_800_400")                                                                            # hard code

    join_datasets(datasets, "Signal_all", signal_list)
    
elif args.signal == "Signal_all_xsec":

    join_datasets(datasets, "Signal_all_xsec", signal_list)
    
elif args.signal == "Signal_one_relevant":
    for sgn in signal_list:
        datasets[sgn].loc[:,"evtWeight"] = datasets[sgn]["evtWeight"]/datasets[sgn]["evtWeight"].sum()
    datasets["Signal_800_600"].loc[:,"evtWeight"] = datasets["Signal_800_600"]["evtWeight"]*2000000               # hard code
    
    join_datasets(datasets, "Signal_one_relevant", signal_list)
    
    
join_datasets(datasets, "Bkg", backgrounds)


#=====================================================================================================================
# Output setup
#=====================================================================================================================
ml_outpath = os.path.join(outpath_base, period, "ML")
if not os.path.exists(ml_outpath):
    os.makedirs(ml_outpath)

signal_outpath = os.path.join(ml_outpath, args.mode, args.signal)
if not os.path.exists(signal_outpath):
    os.makedirs(signal_outpath)

plots_outpath = os.path.join(signal_outpath, "features")    
if not os.path.exists(plots_outpath):
    os.makedirs(plots_outpath)
    
if not os.path.exists(os.path.join(signal_outpath, "models")):
    os.makedirs(os.path.join(signal_outpath, "models")) 
    
model_outpath = os.path.join(signal_outpath, "models", modelName[int(args.job)])
if not os.path.exists(model_outpath):
    os.makedirs(model_outpath)
    
print('Results will be stored in ' + ml_outpath)


#=====================================================================================================================
# Preprocessing input data
#=====================================================================================================================
print("")
print("Preprocessing input data...")

df = {}
for i in range(len(classes)):
    df[i] = datasets[classes[i]].copy()

df_train = {}
df_test = {}
for key in df.keys():
    dataset = df[key] 
    #dataset = dataset[(dataset["RecoLepID"] < 1000) & (dataset["Nbjets"] > 0)]
    if len(dataset) > 0 :
        dataset = dataset.sample(frac=1)
        dataset = dataset.reset_index(drop=True)
        dataset["class"] = key

        train_limit = int(train_frac*len(dataset))
        df_train_i = dataset.loc[0:(train_limit-1),:].copy()
        df_test_i = dataset.loc[train_limit:,:].copy()
    
        sum_weights = dataset["evtWeight"].sum()
        train_factor = dataset["evtWeight"].sum()/df_train_i["evtWeight"].sum()
        test_factor = dataset["evtWeight"].sum()/df_test_i["evtWeight"].sum()
        df_train_i["evtWeight"] = df_train_i["evtWeight"]*train_factor 
        df_test_i["evtWeight"] = df_test_i["evtWeight"]*test_factor
        df_train_i['mvaWeight'] = df_train_i['evtWeight']/df_train_i['evtWeight'].sum()
        df_test_i['mvaWeight'] = df_test_i['evtWeight']/df_test_i['evtWeight'].sum()

        df_train[key] = df_train_i
        df_test[key] = df_test_i


list_train = [df_train[key] for key in df.keys()]
df_mva = pd.concat(list_train).reset_index(drop=True)

mean = []
std = []
for i in range(len(variables)):
    weighted_stats = DescrStatsW(df_mva[variables[i]], weights=df_mva["mvaWeight"], ddof=0)
    mean.append(weighted_stats.mean)
    std.append(weighted_stats.std)
print("mean: " + str(mean))
print("std: " + str(std))


stat_values={"mean": mean, "std": std}

if args.mode == "keras":
    with open(os.path.join(signal_outpath, "models", 'preprocessing.json'), 'w') as json_file:
        json.dump(stat_values, json_file)

    for key in df.keys():
        df_train[key][variables] = (df_train[key][variables] - mean) / std
        df_test[key][variables] = (df_test[key][variables] - mean) / std

    

control = True
for key in df.keys():
    if control:
        df_full_train = df_train[key].copy()
        df_full_test = df_test[key].copy()
        control = False
    else:
        df_full_train = pd.concat([df_full_train, df_train[key]])
        df_full_test = pd.concat([df_full_test, df_test[key]])

#df_full_train.to_csv("files/train.csv", index=False)
#df_full_test.to_csv("files/test.csv", index=False)
#del df_full_train, df_full_test


#=====================================================================================================================
# Plot training and test distributions
#=====================================================================================================================
for ivar in range(len(variables)):
    
    fig1 = plt.figure(figsize=(10,7))
    gs1 = gs.GridSpec(1, 1)
    #==================================================
    ax1 = plt.subplot(gs1[0])            
    #==================================================
    var = variables[ivar]
    if args.mode == "keras":
        bins = np.linspace(-2.5,2.5,51)
    elif args.mode == "torch":
        bins = np.linspace(mean[ivar]-2.5*std[ivar],mean[ivar]+2.5*std[ivar],51)
    for key in df.keys():
        step_plot( ax1, var, df_train[key], label=class_names[key]+" (train)", color=colors[key], weight="mvaWeight", bins=bins, error=True )
        step_plot( ax1, var, df_test[key], label=class_names[key]+" (test)", color=colors[key], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )
    ax1.set_xlabel(var_names[ivar], size=14, horizontalalignment='right', x=1.0)
    ax1.set_ylabel("Events normalized", size=14, horizontalalignment='right', y=1.0)
    
    ax1.tick_params(which='major', length=8)
    ax1.tick_params(which='minor', length=4)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.margins(x=0)
    ax1.legend(numpoints=1, ncol=2, prop={'size': 10.5}, frameon=False)

    plt.savefig(os.path.join(plots_outpath, var + '.png'))

if args.check_flag:    
    sys.exit()


#=====================================================================================================================
# Load Datasets
#=====================================================================================================================

#df_full_train = pd.read_csv(os.path.join(inpath,"train.csv"))
df_full_train = df_full_train.sample(frac=1)
train_x = df_full_train[variables]
train_x = train_x.values
train_y = np.array(df_full_train['class']).ravel()
train_w = np.array(df_full_train['mvaWeight']).ravel()                    # weight to signal x bkg comparison
print("Variables shape = " + str(train_x.shape))
print("Labels shape = " + str(train_y.shape))
print("Weights shape = " + str(train_w.shape))

#df_full_test = pd.read_csv(os.path.join(inpath,"test.csv"))
df_full_test = df_full_test.sample(frac=1)
test_x = df_full_test[variables]
test_x = test_x.values
test_y = np.array(df_full_test['class']).ravel()
test_w = np.array(df_full_test['mvaWeight']).ravel()                      # weight to signal x bkg comparison

#df_source = pd.read_csv(os.path.join(inpath,"source.csv"))
df_source = df_full_train.copy()
df_source = df_source.sample(frac=1)
source_x = df_source[variables]
source_x = source_x.values
source_w = np.array(df_source['mvaWeight']).ravel()                  # weight to source x target comparison

#df_target = pd.read_csv(os.path.join(inpath,"target.csv"))
df_target = df_full_test.copy()
df_target = df_target.sample(frac=1)
target_x = df_target[variables]
target_x = target_x.values
target_w = np.array(df_target['mvaWeight']).ravel()                  # weight to source x target comparison
         

n_classes = len(df_full_train["class"].unique())

signal_param = []
if args.signal == "Signal_parameterized":
    model[N][0] = "P" + model[N][0]
    
    for param in signal_parameters:
        signal_param.append(np.sort(np.unique(df_train[0][param])))
    
    #print(signal_param)
    #print(df_full_train[variables])
    
    
#=====================================================================================================================
# RUN TRAINING
#=====================================================================================================================
print("")
print("Training...")

start = time.time()
        
  
class_model, iteration, train_acc, test_acc, train_loss, test_loss, adv_source_acc, adv_target_acc = train_model(
    train_x, 
    train_y, 
    train_w, 
    test_x, 
    test_y, 
    test_w,
    source_x, 
    source_w, 
    target_x, 
    target_w, 
    model[N], 
    n_var,
    n_classes,
    n_iterations = num_max_iterations, 
    signal_param = signal_param,
    mode = args.mode,
    stat_values = stat_values,
    eval_frac = eval_frac
    )

if args.mode == "keras":
    class_model.save(os.path.join(model_outpath, "model.h5"))
elif args.mode == "torch":
    #torch.save(class_model, os.path.join(model_outpath, "model.pt"))
    model_scripted = torch.jit.script(class_model) # Export to TorchScript
    model_scripted.save(os.path.join(model_outpath, "model_scripted.pt"))


#=====================================================================================================================
# SAVE TRAINING INFORMATION 
#=====================================================================================================================
df_training = pd.DataFrame(list(zip(iteration, train_acc, test_acc, train_loss, test_loss, adv_source_acc, adv_target_acc)),columns=["iteration", "train_acc", "test_acc", "train_loss", "test_loss", "adv_source_acc", "adv_target_acc"])

df_training.to_csv(os.path.join(model_outpath, 'training.csv'), index=False)

iteration = df_training['iteration']
train_acc = df_training['train_acc']
test_acc = df_training['test_acc'] 
train_loss = df_training['train_loss']
test_loss = df_training['test_loss']
adv_source_acc = df_training['adv_source_acc'] 
adv_target_acc = df_training['adv_target_acc']
adv_sum_acc = np.array(df_training['adv_source_acc']) + np.array(df_training['adv_target_acc'])

min_loss = np.amin(test_loss)
position = np.array(iteration[test_loss == min_loss])[0]


fig1 = plt.figure(figsize=(18,5))
grid = [1, 2]
gs1 = gs.GridSpec(grid[0], grid[1])
#-----------------------------------------------------------------------------------------------------------------
# Accuracy
#-----------------------------------------------------------------------------------------------------------------
ax1 = plt.subplot(gs1[0])
plt.axvline(position, color='grey')
plt.plot(iteration, train_acc, "-", color='red', label='Train (Class Accuracy)')
plt.plot(iteration, test_acc, "-", color='blue', label='Test (Class Accuracy)')
#plt.plot(iteration, adv_target_acc, "-", color='orange', label='Target (Domain Accuracy)')
#plt.plot(iteration, adv_source_acc, "-", color='green', label='Source (Domain Accuracy)')
#plt.plot(iteration, adv_sum_acc, "-", color='orchid', label='Sum (Domain Accuracy)')
plt.axhline(1, color='grey', linestyle='--')
ax1.set_xlabel("iterations", size=14, horizontalalignment='right', x=1.0)
ax1.set_ylabel("Accuracy", size=14, horizontalalignment='right', y=1.0)
ax1.tick_params(which='major', length=8)
ax1.tick_params(which='minor', length=4)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
#ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
ax1.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.margins(x=0)
ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='lower right')

#-----------------------------------------------------------------------------------------------------------------
# Loss
#-----------------------------------------------------------------------------------------------------------------
ax2 = plt.subplot(gs1[1])
plt.axvline(position, color='grey')
plt.plot(iteration, train_loss, "-", color='red', label='Train (Class Loss)')
plt.plot(iteration, test_loss, "-", color='blue', label='Test (Class Loss)')
plt.yscale('log')
ax2.set_xlabel("iterations", size=14, horizontalalignment='right', x=1.0)
ax2.set_ylabel("Loss", size=14, horizontalalignment='right', y=1.0)
ax2.tick_params(which='major', length=8)
ax2.tick_params(which='minor', length=4)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
#ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
ax2.grid(which='major', axis='y', linewidth=0.2, linestyle='-', color='0.75')
ax2.spines['bottom'].set_linewidth(1)
ax2.spines['top'].set_linewidth(1)
ax2.spines['left'].set_linewidth(1)
ax2.spines['right'].set_linewidth(1)
ax2.margins(x=0)
ax2.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False)

plt.subplots_adjust(left=0.055, bottom=0.115, right=0.990, top=0.95, wspace=0.18, hspace=0.165)
plt.savefig(os.path.join(model_outpath, "training.png"))


#=====================================================================================================================
# CHECK OVERTRAINING AND ROC
#=====================================================================================================================
for key in df.keys():
    train_x = df_train[key][variables]
    train_x = train_x.values
    test_x = df_test[key][variables]
    test_x = test_x.values
    
    if args.mode == "keras":
        train_class_pred = class_model.predict(train_x)
        test_class_pred = class_model.predict(test_x)
    elif args.mode == "torch":
        train_class_pred = model_scripted(torch.FloatTensor(train_x)).detach().numpy()
        test_class_pred = model_scripted(torch.FloatTensor(test_x)).detach().numpy()
    
    if model[N][4] == "cce":
        n_outputs = n_classes
        for i in range(n_outputs):
            pred_name = 'score_C'+str(i)
            df_test[key][pred_name] = test_class_pred[:,i]
            df_train[key][pred_name] = train_class_pred[:,i]
    if model[N][4] == "bce":
        n_outputs = 1
        pred_name = 'score_C0'
        df_test[key][pred_name] = 1 - test_class_pred[:,0]
        df_train[key][pred_name] = 1 - train_class_pred[:,0]
    
for i in range(n_outputs):
    fig1 = plt.figure(figsize=(20,7))
    gs1 = gs.GridSpec(1,1)
    #==================================================
    ax1 = plt.subplot(gs1[0])            
    #==================================================
    var = 'score_C'+str(i)
    bins = np.linspace(0,1,201)
    for key in df.keys():
        step_plot( ax1, var, df_train[key], label=class_names[key]+" (train)", color=colors[key], weight="mvaWeight", bins=bins, error=True )
        step_plot( ax1, var, df_test[key], label=class_names[key]+" (test)", color=colors[key], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )
    ax1.set_xlabel(classes[i] + " score", size=14, horizontalalignment='right', x=1.0)
    ax1.set_ylabel("Events normalized", size=14, horizontalalignment='right', y=1.0)

    ax1.tick_params(which='major', length=8)
    ax1.tick_params(which='minor', length=4)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.margins(x=0)
    ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='upper center')
    
    plt.savefig(os.path.join(model_outpath, var + ".png"))
    

    
fig1 = plt.figure(figsize=(18,5))
grid = [1, 2]
gs1 = gs.GridSpec(grid[0], grid[1])
#==================================================
ax1 = plt.subplot(gs1[0])            
#==================================================
var = 'score_C0'
signal_train_roc = []
signal_test_roc = []
bkg_train_roc = []
bkg_test_roc = []
ikey = 0
for key in df.keys():
    if ikey == 0:
        signal_train_roc.append(df_train[key])
        signal_test_roc.append(df_test[key])
        ikey += 1
    else:
        bkg_train_roc.append(df_train[key])
        bkg_test_roc.append(df_test[key])
        
ctr_train = func.control( var, signal_train_roc, bkg_train_roc, weight="evtWeight", bins=np.linspace(0,1,1001) )
ctr_train.roc_plot(label='ROC (train)', color='blue', linestyle="-")
ctr_test = func.control( var, signal_test_roc, bkg_test_roc, weight="evtWeight", bins=np.linspace(0,1,1001) )
ctr_test.roc_plot(label='ROC (test)', color='blue', linestyle="--")

ax1.set_xlabel("Background rejection", size=14, horizontalalignment='right', x=1.0)
ax1.set_ylabel("Signal efficiency", size=14, horizontalalignment='right', y=1.0)

ax1.tick_params(which='major', length=8)
ax1.tick_params(which='minor', length=4)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.spines['bottom'].set_linewidth(1)
ax1.spines['top'].set_linewidth(1)
ax1.spines['left'].set_linewidth(1)
ax1.spines['right'].set_linewidth(1)
ax1.margins(x=0)
ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='lower left')

plt.savefig(os.path.join(model_outpath, "ROC.png"))
    
  
   
#=====================================================================================================================   
end = time.time()
hours = int((end - start)/3600)
minutes = int(((end - start)%3600)/60)
seconds = int(((end - start)%3600)%60)

print("")
print("-----------------------------------------------------------------------------------")
print("Total process duration: " + str(hours) + " hours " + str(minutes) + " minutes " + str(seconds) + " seconds")  
print("-----------------------------------------------------------------------------------")
print("")




