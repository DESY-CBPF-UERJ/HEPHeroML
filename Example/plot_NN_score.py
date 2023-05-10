# %%
#==================================================================================================
# Import packages
#==================================================================================================
import os
import numpy as np
import pandas as pd
import json
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from tensorflow.keras.models import load_model
import anatools.data as data
import anatools.analysis as ana
ana.start()


# %%
#==================================================================================================
# Read files
#==================================================================================================
basedir = "/home/gcorreia/ML/datasets"
period = '17'
luminosity = '41.5'

ds = data.read_files(basedir, period, mode="normal")

data.join_datasets(ds, "ST", [
    "ST_tW_antitop", 
    "ST_tW_top", 
    "ST_s-channel",
    "ST_t-channel_top", 
    "ST_t-channel_antitop",
    ], mode="normal")

data.join_datasets(ds, "TT", [
    "TTTo2L2Nu", 
    "TTToSemiLeptonic",
    ], mode="normal")

data.join_datasets(ds, "DYJetsToLL", [
    "DYJetsToLL_Pt-Inclusive",
    ], mode="normal")

data.join_datasets(ds, "Residual", [
    "WZZ", 
    "WWZ", 
    "ZZZ", 
    "WWW", 
    "ZGToLLG", 
    #"TTGamma", 
    "TTGJets", 
    "WW",
    #"WGToLNuG", 
    #"TTZToQQ", 
    #"TTZToNuNu", 
    #"TWZToLL_thad_Wlept", 
    #"TWZToLL_tlept_Whad", 
    #"TWZToLL_tlept_Wlept", 
    #"WJetsToLNu", 
    #"TTWZ", 
    #"TTZZ",
    ], mode="normal")


def signal_label(param_0, param_1):
    label = r'$m_H=$' + str(param_0) + r', $m_\mathit{a}=$' + str(param_1)
    return label



variables = ["LeadingLep_pt", "LepLep_pt", "LepLep_deltaR", "LepLep_deltaM", "MET_pt", "MET_LepLep_Mt", "MET_LepLep_deltaPhi"]

model = load_model('NN_3_20_elu_adam_soft/model.h5')

with open("preprocessing.json") as prep_file:
    preprocessing = json.load(prep_file) 
mean = preprocessing["mean"]
std = preprocessing["std"]

df = ds.copy()
for key in df.keys():
    df_input = df[key][variables]
    df_input = df_input.values
    df_input = (df_input - mean) / std
    prediction = model.predict(df_input)
    df[key]["Signal_NN_score"] = prediction[:,0]


# %%
#==================================================================================================
# Plot
#==================================================================================================
fig1 = plt.figure(figsize=(20,7))
grid = [1, 2]
gs1 = gs.GridSpec(grid[0], grid[1])


colors = ['gainsboro', 'orchid', 'limegreen', 'red', 'skyblue', 'darkgoldenrod']
labels = [r'Residual SM', r'$WZ$', r'$ZZ$', 'Single top', r'$t\bar{t}$', 'Drell-Yan']
dataframes = [df["Residual"], df["WZ"], df["ZZ"], df["ST"], df["TT"], df["DYJetsToLL"]]
dataframes, labels, colors, sizes = data.order_datasets(dataframes, labels, colors)


#=================================================================================================================
N = 1
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "Signal_NN_score"
bins = np.linspace(0,1,101)
ybkg, errbkg = ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
ysgn1, errsgn1 = ana.step_plot( ax1, var, df["Signal_1000_100"], label=signal_label(1000,100), color='blue', weight="evtWeight", bins=bins, error=True )
ana.labels(ax1, xlabel=r"$\mathrm{NN\ score}$", ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=luminosity, year=period[-2:], ylog=True, legend_ncol=4, ylim=[1.e-1,1.e6]) # Set up the plot style and information on top


#=================================================================================================================
N = 2
#=================================================================================================================
#==================================================
ax1 = plt.subplot(ana.position(gs1,grid,N,1))              # Positioning at subplot 1 of the plot number 1
#==================================================
var = "Signal_NN_score"
bins = np.linspace(0.9,1,101)
ybkg, errbkg = ana.stacked_plot( ax1, var, dataframes, labels, colors, weight="evtWeight", bins=bins )  # Produce the stacked plot
ysgn1, errsgn1 = ana.step_plot( ax1, var, df["Signal_1000_100"], label=signal_label(1000,100), color='blue', weight="evtWeight", bins=bins, error=True )
ana.labels(ax1, xlabel=r"$\mathrm{NN\ score}$", ylabel="Events")  # Set up the label names
ana.style(ax1, lumi=luminosity, year=period[-2:], ylog=True, legend_ncol=4, ylim=[1.e-1,1.e6]) # Set up the plot style and information on top


#=================================================================================================================
# Make final setup, save and show plots
#=================================================================================================================
plt.subplots_adjust(left=0.045, bottom=0.085, right=0.97, top=0.965, wspace=0.2, hspace=0.09)
plt.savefig('NN_score_'+period+'.pdf')
plt.savefig('NN_score_'+period+'.png')
plt.show()

