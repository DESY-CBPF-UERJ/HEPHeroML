import sys
import pandas as pd
import os
import concurrent.futures as cf
from operator import itemgetter
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.patches as pat
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
plt.style.use('classic')
from tqdm import tqdm
import json
import h5py
import math

seed = 16
import numpy as np
numpy_random = np.random.RandomState(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
import torch.nn as nn

from custom_opts.ranger import Ranger
from models.NN import *
from models.PNN import *
from models.PNET import *

"""
-> model training minimize classification and domain at the same time (affect all weights)
-> The weights of the domain branch are recovered
-> domain model training minimize domain (classification branch weights are not affected)
-> The weights of the comum branch are recovered (the weights of the classification model are effectivily updated in the "model training")

-> The target samples don't influence the classification training (sample_weights)
-> During the model training, the weights are modified to improved the accuaracy of the domain and classification prediction
-> During the domain model training, the weights are modified to make the network doesn't be able to distinguish the domains
"""





















#=====================================================================================================================
def get_sample(basedir, period, classes, n_signal, train_frac, load_size, load_it, reweight_info, features=[], vec_features=[], verbose=False, normalization_method="evtsum"):

    has_weights = False
    if len(reweight_info) > 0:
        reweight_vars = [reweight_info[i][0] for i in range(len(reweight_info))]
        reweight_limits = [reweight_info[i][1] for i in range(len(reweight_info))]
        if reweight_vars[-1] == "var_weights":
            var_weights = reweight_limits[-1]
            reweight_vars = reweight_vars[:-1]
            reweight_limits = reweight_limits[:-1]
            has_weights = True
        else:
            var_weights = {}

    class_names = []
    class_labels = []
    class_colors = []
    control = True

    if not verbose:
            print("Loading datasets entries")

    for class_key in classes: # for each class
        info_list = []

        if len(reweight_info) > 0 and not has_weights:
            if len(reweight_vars) == 1:
                var_weights[class_key] = np.ones((2,len(reweight_limits[0])-1))
            elif len(reweight_vars) == 2:
                var_weights[class_key] = np.ones((2,len(reweight_limits[0])-1,len(reweight_limits[1])-1))
            elif len(reweight_vars) == 3:
                var_weights[class_key] = np.ones((2,len(reweight_limits[0])-1,len(reweight_limits[1])-1,len(reweight_limits[2])-1))

        if class_key[:14] == "Signal_samples":
            class_name = classes[class_key][0][n_signal]
            class_label = classes[class_key][0][n_signal]
            input_list = [classes[class_key][0][n_signal]]
        else:
            class_name = class_key
            class_label = classes[class_key][3]
            input_list = classes[class_name][0]
        class_color = classes[class_key][4]

        mode = classes[class_key][1]
        combination = classes[class_key][2]

        #print(" ")
        if verbose:
            print("Loading datasets entries of class", class_key)
        #print("load_it", load_it)

        datasets_dir = os.path.join(basedir, period)
        datasets_abspath = [(f, os.path.join(datasets_dir, f)) for f in os.listdir(datasets_dir)]

        #=================================================================================
        datasets_length = []
        datasets_evtWsum = []
        datasets_names = []
        n_datasets = 0
        for dataset, abspath in datasets_abspath: # for each dataset in a class
            dataset_name = dataset.split(".")[0]

            if dataset.endswith(".h5") and dataset_name in input_list:
                with h5py.File(abspath) as f:
                    datasets_length.append(len(np.array(f["scalars/evtWeight"]))) # number of entries
                    datasets_evtWsum.append(np.array(f["scalars/evtWeight"]).sum()) # sum of weights
                    datasets_names.append(dataset_name)
                n_datasets += 1

        if combination == "balanced" or combination == "equal":
            datasets_frac = np.ones(n_datasets)*(1./n_datasets)
        elif combination == "evtsum":
            datasets_evtWsum = np.array(datasets_evtWsum)
            total_evtWsum = datasets_evtWsum.sum()
            datasets_frac = datasets_evtWsum/total_evtWsum

        #=================================================================================
        class_load_size = int(load_size/len(classes))

        datasets_entries = datasets_frac*class_load_size
        datasets_pdf = datasets_entries/datasets_entries.sum()
        datasets_entries = np.array([ int(i) if int(i) >= 2 else 2 for i in datasets_entries])
        datasets_assigned_entries = datasets_entries.copy()
        datasets_entries = np.minimum(datasets_entries, datasets_length)

        for itry in range(3):
            datasets_needed_entries = datasets_length - datasets_entries
            total_remaining_entries = datasets_assigned_entries.sum() - datasets_entries.sum()
            if total_remaining_entries > 0 and datasets_needed_entries.sum() > 0:
                datasets_pdf = np.array([datasets_pdf[i] if datasets_needed_entries[i] > 0 else 0 for i in range(len(datasets_pdf))])
                datasets_pdf = datasets_pdf/datasets_pdf.sum()
                datasets_provided_entries = total_remaining_entries*datasets_pdf
                datasets_provided_entries = [int(i) for i in datasets_provided_entries]

                datasets_entries = np.array([ datasets_entries[i]+datasets_provided_entries[i] if datasets_needed_entries[i] >= datasets_provided_entries[i] else datasets_entries[i]+datasets_needed_entries[i] for i in range(len(datasets_entries))])

        datasets_train_entries = train_frac*datasets_entries
        datasets_train_entries = np.array([ int(i) for i in datasets_train_entries])
        datasets_test_entries = datasets_entries - datasets_train_entries

        datasets_nSlices = np.array([int(datasets_length[i]/datasets_entries[i]) if datasets_entries[i] > 0 else 0 for i in range(len(datasets_length))])

        datasets_slices = load_it%datasets_nSlices
        #print("datasets_slices", datasets_slices)

        datasets_train_limits = [[datasets_slices[i]*datasets_entries[i], datasets_train_entries[i]+datasets_slices[i]*datasets_entries[i]] for i in range(len(datasets_slices))]
        datasets_test_limits = [[datasets_train_entries[i]+datasets_slices[i]*datasets_entries[i], (datasets_slices[i]+1)*datasets_entries[i]] for i in range(len(datasets_slices))]

        #print("datasets_train_limits", datasets_train_limits)
        #print("datasets_test_limits", datasets_test_limits)


        if verbose:
            info_df = pd.DataFrame({
                'Dataset Name': datasets_names,
                'Available': datasets_length,
                'Loaded': datasets_entries,
                'Train': datasets_train_entries,
                'Test': datasets_test_entries,
                'nSlices': datasets_nSlices,
            })
            print(info_df)
            print(" ")


        #=================================================================================

        for it in range(2):
            datasets = {}
            datasets_vec = {}
            ids = 0
            #for dataset, abspath in tqdm(datasets_abspath):
            for dataset, abspath in datasets_abspath:
                dataset_name = dataset.split(".")[0]
                if dataset.endswith(".h5") and dataset_name in input_list:

                    # Getting % usage of virtual_memory ( 3rd field)
                    #print('RAM memory % used:', psutil.virtual_memory()[2])
                    # Getting usage of virtual_memory in GB ( 4th field)
                    #print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

                    variables_dict = {}
                    with h5py.File(abspath) as f:
                        if "scalars" in f.keys():
                            group = "scalars"
                            for variable in f[group].keys():
                                if len(features) == 0 or variable in features:
                                    if it == 0:
                                        variables_dict[variable] = np.array(f[group+"/"+variable][datasets_train_limits[ids][0]:datasets_train_limits[ids][1]])
                                    elif it == 1:
                                        variables_dict[variable] = np.array(f[group+"/"+variable][datasets_test_limits[ids][0]:datasets_test_limits[ids][1]])
                            datasets[dataset_name] = variables_dict
                        else:
                            print("Warning: Dataset " + dataset_name + " is empty!")

                    if len(vec_features) > 0:
                        variables_dict = {}
                        with h5py.File(abspath) as f:
                            if "vectors" in f.keys():
                                group = "vectors"
                                for variable in f[group].keys():
                                    if variable in vec_features:
                                        if it == 0:
                                            variables_dict[variable] = np.array(f[group+"/"+variable][datasets_train_limits[ids][0]:datasets_train_limits[ids][1]])
                                        elif it == 1:
                                            variables_dict[variable] = np.array(f[group+"/"+variable][datasets_test_limits[ids][0]:datasets_test_limits[ids][1]])
                                datasets_vec[dataset_name] = variables_dict
                            else:
                                print("Warning: Dataset " + dataset_name + " is empty!")

                    if len(datasets[dataset_name]["evtWeight"]) > 0:
                        if combination == "equal":
                            datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]/datasets[dataset_name]["evtWeight"].sum()
                        elif combination == "evtsum" or combination == "balanced":
                            if datasets[dataset_name]["evtWeight"].sum() != 0:
                                ds_factor = datasets_evtWsum[ids]/datasets[dataset_name]["evtWeight"].sum()
                                datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]*ds_factor
                            else:
                                datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]*0.

                    ids += 1

            #==========================================================================
            check_list = [True if len(datasets[input_name]["evtWeight"]) > 0 else False for input_name in input_list]
            #print("check_list", check_list)

            if len(input_list) > 0:
                join_datasets(datasets, class_name, input_list, check_list, mode="scalars", combination=combination)
                if len(vec_features) > 0:
                    join_datasets(datasets_vec, class_name, input_list, check_list, mode="vectors", combination=combination)
            #==========================================================================

            ikey = 0
            for key in classes:
                if key == class_key:
                    break
                ikey += 1

            n_entries = len(datasets[class_name]['evtWeight'])
            p_idx = np.random.permutation(n_entries)

            dataset = {}
            for variable in datasets[class_name].keys():
                dataset[variable] = datasets[class_name][variable][p_idx]
                datasets[class_name][variable] = 0
            del datasets

            dataset["class"] = np.ones(n_entries)*ikey
            dataset['mvaWeight'] = dataset['evtWeight']/dataset['evtWeight'].sum()

            dataset_vec = {}
            if len(vec_features) > 0:
                for variable in datasets_vec[class_name].keys():
                    dataset_vec[variable] = datasets_vec[class_name][variable][p_idx]
                    datasets_vec[class_name][variable] = 0
                del datasets_vec










            #==========================================================================
            # REWEIGHT PART
            if len(reweight_info) > 0:
                if len(reweight_vars) == 1:
                    split1 = np.array(reweight_limits[0])
                    var1 = reweight_vars[0]
                    data_var = {var1: dataset[var1], 'mvaWeight': dataset['mvaWeight']}
                    data_var = pd.DataFrame.from_dict(data_var)
                    for j in range(len(split1)-1):
                        if has_weights:
                            fac = var_weights[class_key][it,j]
                        else:
                            bin_Wsum = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1]))]['mvaWeight'].sum()
                            bin_base = split1[j+1]-split1[j]
                            if normalization_method == "evtsum":
                                fac = 1/bin_Wsum
                            elif normalization_method == "area":
                                fac = 1/(bin_Wsum*bin_base)
                            if math.isnan(fac):
                                fac = 1
                            var_weights[class_key][it,j] = fac
                        data_var.loc[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])), 'mvaWeight'] = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1]))]['mvaWeight']*fac

                    dataset['mvaWeight'] = np.array(data_var['mvaWeight']/data_var['mvaWeight'].sum())
                    del data_var

                elif len(reweight_vars) == 2:
                    split1 = reweight_limits[0]
                    var1 = reweight_vars[0]
                    split2 = reweight_limits[1]
                    var2 = reweight_vars[1]
                    data_var = {var1: dataset[var1], var2: dataset[var2], 'mvaWeight': dataset['mvaWeight']}
                    data_var = pd.DataFrame.from_dict(data_var)
                    for j in range(len(split1)-1):
                        for i in range(len(split2)-1):
                            if has_weights:
                                fac = var_weights[class_key][it,j,i]
                            else:
                                bin_Wsum = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1]))]['mvaWeight'].sum()
                                bin_base = (split1[j+1]-split1[j])*(split2[i+1]-split2[i])
                                if normalization_method == "evtsum":
                                    fac = 1/bin_Wsum
                                elif normalization_method == "area":
                                    fac = 1/(bin_Wsum*bin_base)
                                if math.isnan(fac):
                                    fac = 1
                                var_weights[class_key][it,j,i] = fac
                            data_var.loc[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])), 'mvaWeight'] = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1]))]['mvaWeight']*fac

                    dataset['mvaWeight'] = np.array(data_var['mvaWeight']/data_var['mvaWeight'].sum())
                    del data_var

                elif len(reweight_vars) == 3:
                    split1 = reweight_limits[0]
                    var1 = reweight_vars[0]
                    split2 = reweight_limits[1]
                    var2 = reweight_vars[1]
                    split3 = reweight_limits[2]
                    var3 = reweight_vars[2]
                    data_var = {var1: dataset[var1], var2: dataset[var2], var3: dataset[var3], 'mvaWeight': dataset['mvaWeight']}
                    data_var = pd.DataFrame.from_dict(data_var)
                    for j in range(len(split1)-1):
                        for i in range(len(split2)-1):
                            for k in range(len(split3)-1):
                                if has_weights:
                                    fac = var_weights[class_key][it,j,i,k]
                                else:
                                    bin_Wsum = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])) & ((data_var[var3] >= split3[k]) & (data_var[var3] < split3[k+1]))]['mvaWeight'].sum()
                                    bin_base = (split1[j+1]-split1[j])*(split2[i+1]-split2[i])*(split3[k+1]-split3[k])
                                    if normalization_method == "evtsum":
                                        fac = 1/bin_Wsum
                                    elif normalization_method == "area":
                                        fac = 1/(bin_Wsum*bin_base)
                                    if math.isnan(fac):
                                        fac = 1
                                    var_weights[class_key][it,j,i,k] = fac
                                data_var.loc[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])) & ((data_var[var3] >= split3[k]) & (data_var[var3] < split3[k+1])), 'mvaWeight'] = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])) & ((data_var[var3] >= split3[k]) & (data_var[var3] < split3[k+1]))]['mvaWeight']*fac

                    dataset['mvaWeight'] = np.array(data_var['mvaWeight']/data_var['mvaWeight'].sum())
                    del data_var

                if not has_weights:
                    if it == 0:
                        print("Train weights for "+class_key+":")
                    elif it == 1:
                        print("Test weights for "+class_key+":")
                    #print("reweight_vars", reweight_vars)
                    #print("reweight_limits", reweight_limits)
                    print(var_weights[class_key][it])
                    print(" ")
            #dataset['mvaWeight'] = dataset['mvaWeight']/dataset['mvaWeight'].sum()









            #==========================================================================
            if it == 0:
                dataset_train = dataset.copy()
                dataset_vec_train = dataset_vec.copy()
            elif it == 1:
                dataset_test = dataset.copy()
                dataset_vec_test = dataset_vec.copy()
            del dataset, dataset_vec

        if len(reweight_info) > 0 and not has_weights:
            var_weights[class_key][0] = (var_weights[class_key][0] + var_weights[class_key][1])*0.5
            var_weights[class_key][1] = var_weights[class_key][0]
            #print("var_weights_after", var_weights[class_key][0])
            #print("var_weights_after", var_weights[class_key][1])

        class_names.append(class_name)
        class_labels.append(class_label)
        class_colors.append(class_color)

        if control:
            ds_full_train = dataset_train.copy()
            ds_full_test = dataset_test.copy()
            vec_full_train = dataset_vec_train.copy()
            vec_full_test = dataset_vec_test.copy()
            control = False
        else:
            for variable in ds_full_train.keys():
                ds_full_train[variable] = np.concatenate((ds_full_train[variable], dataset_train[variable]), axis=0)
                ds_full_test[variable] = np.concatenate((ds_full_test[variable], dataset_test[variable]), axis=0)
            for variable in vec_full_train.keys():
                #print(variable)
                #print(vec_full_train[variable].shape)
                #print(dataset_vec_train[variable].shape)

                out_size = len(vec_full_train[variable][0])
                dataset_size = len(dataset_vec_train[variable][0])
                diff_size = abs(out_size-dataset_size)
                if out_size > dataset_size:
                    number_of_events = len(dataset_vec_train[variable])
                    for i in range(diff_size):
                        dataset_vec_train[variable] = np.c_[ dataset_vec_train[variable], np.zeros(number_of_events) ]
                elif dataset_size > out_size:
                    number_of_events = len(vec_full_train[variable])
                    for i in range(diff_size):
                        vec_full_train[variable] = np.c_[ vec_full_train[variable], np.zeros(number_of_events) ]
                vec_full_train[variable] = np.concatenate((vec_full_train[variable], dataset_vec_train[variable]), axis=0)

                out_size = len(vec_full_test[variable][0])
                dataset_size = len(dataset_vec_test[variable][0])
                diff_size = abs(out_size-dataset_size)
                if out_size > dataset_size:
                    number_of_events = len(dataset_vec_test[variable])
                    for i in range(diff_size):
                        dataset_vec_test[variable] = np.c_[ dataset_vec_test[variable], np.zeros(number_of_events) ]
                elif dataset_size > out_size:
                    number_of_events = len(vec_full_test[variable])
                    for i in range(diff_size):
                        vec_full_test[variable] = np.c_[ vec_full_test[variable], np.zeros(number_of_events) ]
                vec_full_test[variable] = np.concatenate((vec_full_test[variable], dataset_vec_test[variable]), axis=0)

        if verbose:
            print(" ")
            print(" ")

    if len(reweight_info) > 0 and not has_weights:
        reweight_info.append(["var_weights", var_weights])
    del dataset_train, dataset_test, dataset_vec_train, dataset_vec_test

    return ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, class_colors, reweight_info
























def check_scalars(train_data, variables, var_names, var_use, var_bins, class_names, class_labels, class_colors, plots_outpath):

    train_data = pd.DataFrame.from_dict(train_data)

    for ivar in range(len(variables)):
        if var_bins[ivar] is not None:
            fig1 = plt.figure(figsize=(9,5))
            gs1 = gs.GridSpec(1, 1)
            #==================================================
            ax1 = plt.subplot(gs1[0])
            #==================================================
            var = variables[ivar]
            bins = var_bins[ivar]
            for ikey in range(len(class_names)):
                yHist, errHist = tools.step_plot( ax1, var, train_data[train_data["class"] == ikey], label=class_labels[ikey]+" (train)", color=class_colors[ikey], weight="mvaWeight", bins=bins, error=True )
                #print(variables[ivar], class_names[ikey])
                #print("yHist", np.round(yHist,6).tolist())
                #print("errHist", np.round(errHist,6).tolist())

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

            plt.subplots_adjust(left=0.09, bottom=0.115, right=0.97, top=0.95, wspace=0.18, hspace=0.165)
            plt.savefig(os.path.join(plots_outpath, var + '_check_.png'), dpi=400)
            plt.savefig(os.path.join(plots_outpath, var + '_check_.pdf'))
            plt.close()




#=====================================================================================================================
def join_datasets(ds, new_name, input_list, check_list, mode="scalars", combination="evtsum"):

    datasets_list = []
    for i in range(len(input_list)):
        if check_list[i]:
            if mode == "scalars" and combination == "equal":
                ds[input_list[i]]["evtWeight"] = ds[input_list[i]]["evtWeight"]/ds[input_list[i]]["evtWeight"].sum()
            datasets_list.append(ds[input_list[i]])

    good_list = False
    if mode == "normal":
        ds[new_name] = pd.concat(datasets_list).reset_index(drop=True)
        good_list = True
    elif mode == "syst":
        ds[new_name] = datasets_list
        good_list = True
    elif mode == "scalars" or mode == "vectors":
        ds[new_name] = {}
        first = True
        for dataset in datasets_list:
            if first:
                for variable in dataset.keys():
                    ds[new_name][variable] = dataset[variable].copy()
            else:
                for variable in dataset.keys():
                    if mode == "vectors":
                        out_size = len(ds[new_name][variable][0])
                        dataset_size = len(dataset[variable][0])
                        diff_size = abs(out_size-dataset_size)
                        if out_size > dataset_size:
                            number_of_events = len(dataset[variable])
                            for i in range(diff_size):
                                dataset[variable] = np.c_[ dataset[variable], np.zeros(number_of_events) ]
                        elif dataset_size > out_size:
                            number_of_events = len(ds[new_name][variable])
                            for i in range(diff_size):
                                ds[new_name][variable] = np.c_[ ds[new_name][variable], np.zeros(number_of_events) ]
                    ds[new_name][variable] = np.concatenate((ds[new_name][variable],dataset[variable]))
            first = False
        good_list = True





    else:
        print("Type of the items is not supported!")

    if good_list:
        for input_name in input_list:
            if input_name != new_name:
                del ds[input_name]

    del datasets_list


#=====================================================================================================================
class control:
    """
    Produce control information to assist in the defition of cuts
    """
    def __init__(self, var, signal_list, others_list, weight=None, bins=np.linspace(0,100,5), above=True):
        self.bins = bins
        self.var = var
        self.signal_list = signal_list
        self.others_list = others_list
        self.weight = weight

        use_bins = [np.array([-np.inf]), np.array(bins), np.array([np.inf])]
        use_bins = np.concatenate(use_bins)

        hist_signal_list = []
        for signal in signal_list:
            if weight is not None:
                hist, hbins = np.histogram( signal[var], weights=signal[weight], bins=use_bins )
            else:
                hist, hbins = np.histogram( signal[var], bins=use_bins )
            if not above:
                hist = np.cumsum(hist)
                hist = hist[:-1]
            else:
                hist = np.cumsum(hist[::-1])[::-1]
                hist = hist[1:]
            hist_signal_list.append(hist)
        hist_signal = hist_signal_list[0]
        for i in range(len(signal_list)-1):
            hist_signal = hist_signal + hist_signal_list[i+1]
        self.hist_signal = hist_signal

        hist_others_list = []
        for others in others_list:
            if weight is not None:
                hist, hbins = np.histogram( others[var], weights=others[weight], bins=use_bins )
            else:
                hist, hbins = np.histogram( others[var], bins=use_bins )
            if not above:
                hist = np.cumsum(hist)
                hist = hist[:-1]
            else:
                hist = np.cumsum(hist[::-1])[::-1]
                hist = hist[1:]
            hist_others_list.append(hist)
        hist_others = hist_others_list[0]
        for i in range(len(others_list)-1):
            hist_others = hist_others + hist_others_list[i+1]
        self.hist_others = hist_others

        signal_sum_list = []
        for signal in signal_list:
            if weight is not None:
                signal_sum = signal[weight].sum()
            else:
                signal_sum = len(signal[var])
            signal_sum_list.append(signal_sum)
        full_signal = signal_sum_list[0]
        for i in range(len(signal_list)-1):
            full_signal = full_signal + signal_sum_list[i+1]
        self.full_signal = full_signal

        others_sum_list = []
        for others in others_list:
            if weight is not None:
                others_sum = others[weight].sum()
            else:
                others_sum = len(others[var])
            others_sum_list.append(others_sum)
        full_others = others_sum_list[0]
        for i in range(len(others_list)-1):
            full_others = full_others + others_sum_list[i+1]
        self.full_others = full_others

        self.purity = self.hist_signal/(self.hist_signal + self.hist_others)
        self.eff_signal = self.hist_signal/self.full_signal
        self.eff_others = self.hist_others/self.full_others
        self.rej_others = 1 - self.eff_others

    def roc_plot(self, label='Signal-bkg ROC', color='blue', linestyle="-", version=1):
        if version == 1:
            plt.plot(self.rej_others, self.eff_signal, color=color, label=label, linestyle=linestyle)
        elif version == 2:
            plt.plot(self.eff_signal, self.eff_others, color=color, label=label, linestyle=linestyle)


#======================================================================================================================
def step_plot( ax, var, dataframe, label, color='black', weight=None, error=False, normalize=False, bins=np.linspace(0,100,5), linestyle='solid', overflow=False, underflow=False ):


    if weight is None:
        W = None
        W2 = None
    else:
        W = dataframe[weight]
        W2 = dataframe[weight]*dataframe[weight]

    eff_bins = bins[:]
    if overflow:
        eff_bins[-1] = np.inf
    if underflow:
        eff_bins[0] = -np.inf

    counts, binsW = np.histogram(
        dataframe[var],
        bins=eff_bins,
        weights=W
    )
    yMC = np.array(counts)

    countsW2, binsW2 = np.histogram(
        dataframe[var],
        bins=eff_bins,
        weights=W2
    )
    errMC = np.sqrt(np.array(countsW2))

    if normalize:
        if weight is None:
            norm_factor = len(dataframe[var])
        else:
            norm_factor = dataframe[weight].sum()
        yMC = yMC/norm_factor
        errMC = errMC/norm_factor

    ext_yMC = np.append([yMC[0]], yMC)

    plt.step(bins, ext_yMC, color=color, label=label, linewidth=1.5, linestyle=linestyle)

    if error:
        x = np.array(bins)
        dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
        x = x[:-1]

        ax.errorbar(
            x+0.5*dx,
            yMC,
            yerr=[errMC, errMC],
            fmt=',',
            color=color,
            elinewidth=1
        )

    return yMC, errMC


#======================================================================================================================
def ratio_plot( ax, ynum, errnum, yden, errden, bins=np.linspace(0,100,5), color='black', numerator="data" ):
    x = np.array(bins)
    dx = np.array([ (x[i+1]-x[i]) for i in range(x.size-1)])
    x = x[:-1]
    yratio = np.zeros(ynum.size)
    yeratio = np.zeros(ynum.size)
    y2ratio = np.zeros(ynum.size)
    ye2ratio = np.zeros(ynum.size)
    for i in range(ynum.size):
        if yden[i] == 0:
            yratio[i] = 99999
            yeratio[i] = 0
            ye2ratio[i] = 0
        else:
            yratio[i] = ynum[i]/yden[i]
            yeratio[i] = errnum[i]/yden[i]
            y2ratio[i] = yden[i]/yden[i]
            ye2ratio[i] = errden[i]/yden[i]

    if numerator == "data":
        yl = (yden - errden)/yden
        yh = (yden + errden)/yden
        dy = yh - yl
        pats = [ pat.Rectangle( (x[i], yl[i]), dx[i], dy[i], hatch='/////', fill=False, linewidth=0, edgecolor='grey' ) for i in range(len(x)-1) ]
        pats.append(pat.Rectangle( (x[len(x)-1], yl[len(x)-1]), dx[len(x)-1], dy[len(x)-1], hatch='/////', fill=False, linewidth=0, edgecolor='grey' ))
        for p in pats:
            ax.add_patch(p)

        ax.axhline(1, color='red', linestyle='-', linewidth=0.5)

        ax.errorbar(x+0.5*dx, yratio, yerr=[yeratio, yeratio], xerr=0.5*dx, fmt='.', ecolor='black', color='black', elinewidth=0.7, capsize=0)
    elif numerator == "mc":
        ax.errorbar(x+0.5*dx, y2ratio, yerr=[ye2ratio, ye2ratio], xerr=0.5*dx, fmt=',', ecolor="red", color="red", elinewidth=1.2, capsize=0)

        ax.errorbar(x+0.5*dx, yratio, yerr=[yeratio, yeratio], xerr=0.5*dx, fmt=',', ecolor=color, color=color, elinewidth=1.2, capsize=0)

    return yratio


#=====================================================================================================================
# Define a function to plot model parameters in pytorch
def print_model_parameters(model):
    count = 0
    for ele in model.state_dict():
        count += 1
        if count % 2 != 0:
            print ("The following are the parameters for the layer ", count // 2 + 1)
        if ele.find("bias") != -1:
            print("The size of bias: ", model.state_dict()[ele].size())
        else:
            print("The size of weights: ", model.state_dict()[ele].size())


#=====================================================================================================================
# Torch losses
class BCE_loss(nn.Module): # use with sigmoid
    def __init__(self):
        super(BCE_loss, self).__init__()

    def forward(self, y_true, y_pred, weight, device="cpu"):

        epsilon = 1e-7
        y_pred = (1-2*epsilon)*y_pred + epsilon


        total_bce_loss = torch.sum((-y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred))*weight)
        num_of_samples = torch.sum(weight)
        mean_bce_loss = total_bce_loss / num_of_samples

        return mean_bce_loss


class CCE_loss(nn.Module): # use with softmax
    def __init__(self, num_classes):
        super(CCE_loss, self).__init__()

        self.num_classes = num_classes

    def forward(self, y_true, y_pred, weight, device="cpu"):

        epsilon = 1e-7
        y_pred = (1-2*epsilon)*y_pred + epsilon

        if device == "cuda":
            y_true = torch.eye(self.num_classes, device="cuda")[y_true[:,0]]
        else:
            y_true = torch.eye(self.num_classes)[y_true[:,0]]

        loss_n = -torch.sum(y_true*torch.log(y_pred), dim=-1).view(-1,1)

        total_ce_loss = torch.sum(loss_n*weight)
        num_of_samples = torch.sum(weight)
        mean_ce_loss = total_ce_loss / num_of_samples

        return mean_ce_loss



#=====================================================================================================================
def batch_generator(data, batch_size):
    #Generate batches of data.

    #Given a list of numpy data, it iterates over the list and returns batches of the same size
    #This
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = numpy_random.choice(all_examples_indices, size=batch_size, replace=False)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr



#=====================================================================================================================
def train_model(input_path, N_signal, train_frac, load_size, parameters, variables, var_names, var_use, classes, reweight_info, n_iterations = 5000, mode = "torch", stat_values = None, eval_step_size = 0.2, feature_info = False, vec_variables=[], vec_var_names=[], vec_var_use=[], early_stopping=300, device="cpu", initial_model_path=None):

    n_classes = len(classes)

    model_type = parameters[0]
    batch_size = parameters[5]
    learning_rates = parameters[6]

    if not isinstance(learning_rates, list):
        learning_rates = [learning_rates]

    if mode == "torch":

        iteration_cum = 0
        iteration = []
        train_acc = []
        test_acc = []
        train_loss = []
        test_loss = []
        position = 0
        min_loss = 99999
        lr_last_it = []
        for ilr in range(len(learning_rates)):
            parameters[6] = learning_rates[ilr]

            if ilr == 0:
                torch.set_num_threads(6)

                # Criterion
                if parameters[4] == 'cce':
                    criterion = CCE_loss(num_classes=n_classes)
                elif parameters[4] == 'bce':
                    criterion = BCE_loss()

                # Model
                class_discriminator_model = build_model(model_type, parameters, n_classes, stat_values, variables, var_use, vec_variables, vec_var_use, device)
                if device == "cuda":
                    class_discriminator_model = nn.DataParallel(class_discriminator_model) # Wrap the model with DataParallel
                    class_discriminator_model = class_discriminator_model.to('cuda') # Move the model to the GPU
                #print(list(class_discriminator_model.parameters()))
                print(" ")
                print(class_discriminator_model.parameters)
                print(" ")

                #checkpoint_path='checkpoint_model.pt'
                checkpoint={'iteration':None, 'model_state_dict':None, 'optimizer_state_dict':None, 'loss': None}

                if initial_model_path is not None:
                    class_discriminator_model.load_state_dict(torch.load(initial_model_path, weights_only=True))

            early_stopping_count = 0
            load_it = 0
            period_count = 0
            waiting_period = 99999
            verbose = True
            for i in tqdm(range(n_iterations)):

                if ((load_it == 0) or (period_count == waiting_period)) and (iteration_cum == 0):
                    ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, colors, reweight_info = get_sample(input_path, parameters[1], classes, N_signal, train_frac, load_size, load_it, reweight_info, features=variables+["evtWeight"], vec_features=vec_variables, verbose=verbose)
                    if verbose:
                        verbose = False
                    load_it += 1
                    waiting_period = int(len(ds_full_train['mvaWeight'])/batch_size)
                    period_count = 0

                    train_data = process_data(model_type, ds_full_train, vec_full_train, variables, vec_variables, var_use, vec_var_use)
                    test_data = process_data(model_type, ds_full_test, vec_full_test, variables, vec_variables, var_use, vec_var_use)

                    del ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, colors

                    # Create batch samples
                    #trainloader = DataLoader(dataset = dataset, batch_size = 1) #alternative -> 3_2_Mini_Batch_Descent.py
                    # Return randomly a sample with number of elements equals to the batch size
                    train_batches = batch_generator(train_data, batch_size)

                    n_eval_train_steps = int(len(train_data[-1])/eval_step_size) + 1
                    n_eval_test_steps = int(len(test_data[-1])/eval_step_size) + 1
                    train_w_sum = train_data[-1].sum()
                    test_w_sum = test_data[-1].sum()


                period_count += 1
                batch_data = next(train_batches)
                class_discriminator_model.train()
                class_discriminator_model = update_model(model_type, class_discriminator_model, criterion, parameters, batch_data, stat_values, var_use, device)
                class_discriminator_model.eval()

                #------------------------------------------------------------------------------------
                if ((i + 1) % 1 == 0):

                    with torch.no_grad():
                        train_loss_i = 0
                        train_acc_i = 0
                        for i_eval in range(n_eval_train_steps):
                            i_eval_output = evaluate_model(model_type, train_data, class_discriminator_model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, device, mode="metric")
                            if i_eval_output is None:
                                continue
                            else:
                                i_eval_loss, i_eval_acc = i_eval_output
                            train_loss_i += i_eval_loss
                            train_acc_i += i_eval_acc
                        train_loss_i = train_loss_i/train_w_sum
                        train_acc_i = train_acc_i/train_w_sum


                        test_loss_i = 0
                        test_acc_i = 0
                        for i_eval in range(n_eval_test_steps):
                            i_eval_output = evaluate_model(model_type, test_data, class_discriminator_model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, device, mode="metric")
                            if i_eval_output is None:
                                continue
                            else:
                                i_eval_loss, i_eval_acc = i_eval_output
                            test_loss_i += i_eval_loss
                            test_acc_i += i_eval_acc
                        test_loss_i = test_loss_i/test_w_sum
                        test_acc_i = test_acc_i/test_w_sum


                        iteration.append(iteration_cum+i+1)
                        train_acc.append(train_acc_i)
                        test_acc.append(test_acc_i)
                        train_loss.append(train_loss_i)
                        test_loss.append(test_loss_i)

                        if( (test_loss_i < min_loss) ):
                            min_loss = test_loss_i
                            position = i+1
                            #checkpoint['iteration']=iteration
                            checkpoint['model_state_dict']=class_discriminator_model.state_dict()
                            #checkpoint['optimizer_state_dict']= optimizer.state_dict()
                            checkpoint['loss']=min_loss
                            early_stopping_count = 0
                        else:
                            early_stopping_count += 1

                        print("ilr =  %d, iterations %d, class loss =  %.10f, class accuracy =  %.3f"%(ilr, i+1, test_loss_i, test_acc_i ))

                        if early_stopping_count == early_stopping:
                            print("Early stopping activated!")
                            break

            #--------------------------------------------------------------------------------------
            if( position > 0 ):
                if( ilr == len(learning_rates)-1 ):
                    # Set weights of the best classification model
                    class_discriminator_model.load_state_dict(checkpoint['model_state_dict'])
                    min_loss = checkpoint['loss']
                else:
                    class_discriminator_model.load_state_dict(checkpoint['model_state_dict'])
                    min_loss = checkpoint['loss']
                    if early_stopping_count > 0:
                        iteration = iteration[:-early_stopping_count]
                        train_acc = train_acc[:-early_stopping_count]
                        test_acc = test_acc[:-early_stopping_count]
                        train_loss = train_loss[:-early_stopping_count]
                        test_loss = test_loss[:-early_stopping_count]
                    iteration_cum = iteration[-1]
                lr_last_it.append(iteration[-1])

        lr_info = pd.DataFrame(list(zip(learning_rates, lr_last_it)),columns=["lr_values", "lr_last_it"])

        if( position > 0 ):
            # Permutation feature importance
            # https://cms-ml.github.io/documentation/optimization/importance.html
            feature_score_info = []
            if feature_info:
                print("")
                print("------------------------------------------------------------------------")
                print("Computing feature importance")
                print("------------------------------------------------------------------------")
                with torch.no_grad():
                    feature_score_info = feature_score(model_type, test_data, class_discriminator_model, min_loss, eval_step_size, criterion, parameters, variables, vec_variables, var_use, vec_var_use, var_names, vec_var_names, stat_values, device)


        adv_source_acc = np.zeros_like(test_acc)
        adv_target_acc = np.zeros_like(test_acc)


    return class_discriminator_model, np.array(iteration), np.array(train_acc), np.array(test_acc), np.array(train_loss), np.array(test_loss), np.array(adv_source_acc), np.array(adv_target_acc), feature_score_info, lr_info



#=====================================================================================================================
def evaluate_models(period, library, tag, outpath_base):

    best_models_path = os.path.join(outpath_base, period, "ML", "best_models")
    if not os.path.exists(best_models_path):
        os.makedirs(best_models_path)

    list_signals = os.listdir(os.path.join(outpath_base, period, "ML", library, tag))

    print("#########################################################################################")
    print(library)
    print("#########################################################################################")

    ml_outpath = os.path.join(outpath_base, period, "ML", library, tag)
    os.system("rm -rf " + os.path.join(best_models_path, library, tag))
    print("outpath = ", ml_outpath)

    list_best_models = []
    for signal in list_signals:
        #print(signal)
        list_models = os.listdir(os.path.join(ml_outpath, signal, "models"))
        models_loss = []
        models_accuracy = []
        models_iterations = []
        models_name = []
        for model in list_models:
            #print(model)
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
        save_path = os.path.join(best_models_path, library, tag, signal)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        df_training.to_csv(os.path.join(save_path, "training_result.csv"))

        os.system("cp -rf " + os.path.join(ml_outpath, signal, 'features') + " " + save_path)

        models_path = os.path.join(save_path, 'models')
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if len(df_training) <= 1:
            for model in list_models:
                copyCommand = "cp -rf " + os.path.join(ml_outpath, signal, 'models', model) + " " + models_path
                os.system(copyCommand)
        else:
            for model in list_models:
                #if( model == df_training.loc[0]["Model"] or model == df_training.loc[1]["Model"] or model == df_training.loc[2]["Model"] ):
                if( model == df_training.loc[0]["Model"] ):
                    copyCommand = "cp -rf " + os.path.join(ml_outpath, signal, 'models', model) + " " + models_path
                    os.system(copyCommand)



    df_result = pd.DataFrame({"signal": list_signals, "best model": list_best_models})
    df_result = df_result.reset_index()
    df_result.to_csv(os.path.join(best_models_path, library, tag, "best_models.csv"))










#=====================================================================================================================
def build_model(model_type, parameters, n_classes, stat_values, variables, var_use, vec_variables, vec_var_use, device):

    if model_type == "NN":
        model = build_NN(parameters, variables, n_classes, stat_values, device)
    elif model_type == "PNN":
        model = build_PNN(parameters, variables, n_classes, stat_values, device)
    elif model_type == "PNET":
        model = build_PNET(vec_variables, vec_var_use, n_classes, parameters, stat_values, device)

    return model


#=====================================================================================================================
def model_parameters(model_type, param_dict):

    if model_type == "NN":
        model_parameters = model_parameters_NN(param_dict)
    elif model_type == "PNN":
        model_parameters = model_parameters_PNN(param_dict)
    elif model_type == "PNET":
        model_parameters = model_parameters_PNET(param_dict)

    return model_parameters


#=====================================================================================================================
def features_stat(model_type, train_data, test_data, vec_train_data, vec_test_data, variables, vec_variables, var_names, vec_var_names, var_use, vec_var_use, class_names, class_labels, class_colors, plots_outpath):

    if model_type == "NN":
        stat_values = features_stat_NN(train_data, test_data, variables, var_names, class_names, class_labels, class_colors, plots_outpath)
    elif model_type == "PNN":
        stat_values = features_stat_PNN(train_data, test_data, variables, var_names, var_use, class_names, class_labels, class_colors, plots_outpath)
    elif model_type == "PNET":
        stat_values = features_stat_PNET(train_data, test_data, vec_train_data, vec_test_data, vec_variables, vec_var_names, vec_var_use, class_names, class_labels, class_colors, plots_outpath)

    return stat_values


#=====================================================================================================================
def update_model(model_type, model, criterion, parameters, batch_data, stat_values, var_use, device):

    if model_type == "NN":
        model = update_NN(model, criterion, parameters, batch_data, device)
    elif model_type == "PNN":
        model = update_PNN(model, criterion, parameters, batch_data, stat_values, var_use, device)
    elif model_type == "PNET":
        model = update_PNET(model, criterion, parameters, batch_data, device)

    return model


#=====================================================================================================================
def process_data(model_type, scalar_var, vector_var, variables, vec_variables, var_use, vec_var_use):

    if model_type == "NN":
        input_data = process_data_NN(scalar_var, variables)
    elif model_type == "PNN":
        input_data = process_data_PNN(scalar_var, variables)
    elif model_type == "PNET":
        input_data = process_data_PNET(scalar_var, vector_var, vec_variables, vec_var_use)

    return input_data


#=====================================================================================================================
def evaluate_model(model_type, input_data, model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, device, mode="predict"):

    if model_type == "NN":
        i_eval_output = evaluate_NN(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode)
    elif model_type == "PNN":
        i_eval_output = evaluate_PNN(input_data, model, i_eval, eval_step_size, criterion, parameters, stat_values, var_use, device, mode)
    elif model_type == "PNET":
        i_eval_output = evaluate_PNET(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode)

    return i_eval_output


#=====================================================================================================================
def feature_score(model_type, input_data, model, min_loss, eval_step_size, criterion, parameters, variables, vec_variables, var_use, vec_var_use, var_names, vec_var_names, stat_values, device):

    if model_type == "NN":
        feature_score_info = feature_score_NN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, device)
    elif model_type == "PNN":
        feature_score_info = feature_score_PNN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, var_use, stat_values, device)
    elif model_type == "PNET":
        feature_score_info = feature_score_PNET(input_data, model, min_loss, eval_step_size, criterion, parameters, vec_variables, vec_var_use, vec_var_names, device)

    return feature_score_info


#=====================================================================================================================
def save_model(model_type, model, model_outpath, dim, device):

    if model_type == "NN":
        save_NN(model, model_outpath, dim, device)
    elif model_type == "PNN":
        save_PNN(model, model_outpath, dim, device)
    elif model_type == "PNET":
        save_PNET(model, model_outpath, dim, device)


