import sys
import pandas as pd
import os
import concurrent.futures as cf
from operator import itemgetter
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
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
from models.NN_setup import *
from models.PNET_setup import *

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
def get_sample(basedir, period, classes, n_signal, train_frac, load_size, load_it, reweight_info, features=[], vec_features=[]):

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
    for class_key in classes:

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

        #print("")
        print("Loading datasets of class", class_key)

        datasets_dir = os.path.join(basedir, period)
        datasets_abspath = [(f, os.path.join(datasets_dir, f)) for f in os.listdir(datasets_dir)]

        #=================================================================================
        datasets_length = []
        datasets_evtWsum = []
        n_datasets = 0
        for dataset, abspath in datasets_abspath:
            dataset_name = dataset.split(".")[0]

            if dataset.endswith(".h5") and dataset_name in input_list:
                with h5py.File(abspath) as f:
                    datasets_length.append(len(np.array(f["scalars/evtWeight"])))
                    datasets_evtWsum.append(np.array(f["scalars/evtWeight"]).sum())
                n_datasets += 1
        #print("datasets_length", datasets_length)

        #if events_slice is not None:
        if combination == "flat":
            #datasets_length = np.array(datasets_length)
            #total_length = datasets_length.sum()
            #datasets_frac = datasets_length/total_length
            datasets_frac = np.ones(n_datasets)*(1./n_datasets)
            #print("datasets_frac", datasets_frac)
        elif combination == "xsec":
            datasets_evtWsum = np.array(datasets_evtWsum)
            total_evtWsum = datasets_evtWsum.sum()
            datasets_frac = datasets_evtWsum/total_evtWsum
            #print("datasets_frac", datasets_frac)

        #=================================================================================
        class_load_size = int(load_size/len(classes))

        datasets_entries = datasets_frac*class_load_size
        datasets_entries = np.array([ int(i) for i in datasets_entries])
        datasets_entries = np.minimum(datasets_entries, datasets_length)
        #print("datasets_entries", datasets_entries)

        datasets_train_entries = train_frac*datasets_entries
        datasets_train_entries = np.array([ int(i) for i in datasets_train_entries])
        datasets_test_entries = datasets_entries - datasets_train_entries
        #print("datasets_train_entries", datasets_train_entries)
        #print("datasets_test_entries", datasets_test_entries)

        datasets_nSlices = np.array([int(datasets_length[i]/datasets_entries[i]) if datasets_entries[i] > 0 else 0 for i in range(len(datasets_length))])
        #print("datasets_nSlices", datasets_nSlices)

        datasets_slices = load_it%datasets_nSlices
        #print("datasets_slices", datasets_slices)

        datasets_train_limits = [[datasets_slices[i]*datasets_entries[i], datasets_train_entries[i]+datasets_slices[i]*datasets_entries[i]] for i in range(len(datasets_slices))]
        datasets_test_limits = [[datasets_train_entries[i]+datasets_slices[i]*datasets_entries[i], (datasets_slices[i]+1)*datasets_entries[i]] for i in range(len(datasets_slices))]

        #print("datasets_train_limits", datasets_train_limits)
        #print("datasets_test_limits", datasets_test_limits)

        #=================================================================================

        for it in range(2):
            datasets = {}
            datasets_vec = {}
            ids = 0
            #for dataset, abspath in tqdm(datasets_abspath):
            for dataset, abspath in datasets_abspath:
                dataset_name = dataset.split(".")[0]

                if dataset.endswith(".h5") and dataset_name in input_list:


                    variables_dict = {}
                    with h5py.File(abspath) as f:
                        if "scalars" in f.keys():
                            group = "scalars"
                            for variable in f[group].keys():
                                if len(features) == 0 or variable in features:
                                    if it == 0:
                                        variables_dict[variable] = np.array(f[group+"/"+variable])[datasets_train_limits[ids][0]:datasets_train_limits[ids][1]]
                                    elif it == 1:
                                        variables_dict[variable] = np.array(f[group+"/"+variable])[datasets_test_limits[ids][0]:datasets_test_limits[ids][1]]
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
                                            variables_dict[variable] = np.array(f[group+"/"+variable])[datasets_train_limits[ids][0]:datasets_train_limits[ids][1]]
                                        elif it == 1:
                                            variables_dict[variable] = np.array(f[group+"/"+variable])[datasets_test_limits[ids][0]:datasets_test_limits[ids][1]]
                                datasets_vec[dataset_name] = variables_dict
                            else:
                                print("Warning: Dataset " + dataset_name + " is empty!")

                    """
                    if mode == "metadata":
                        variables_dict = {}
                        with h5py.File(abspath) as f:
                            group = "metadata"
                            for variable in f[group].keys():
                                if len(features) == 0:
                                    variables_dict[variable] = np.array(f[group+"/"+variable])
                                elif len(features) > 0:
                                    if variable in features:
                                        variables_dict[variable] = np.array(f[group+"/"+variable])
                            datasets[dataset_name] = variables_dict
                    """

                    if len(datasets[dataset_name]["evtWeight"]) > 0:
                        if combination == "flat":
                            datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]/datasets[dataset_name]["evtWeight"].sum()
                        elif combination == "xsec":
                            ds_factor = datasets_evtWsum[ids]/datasets[dataset_name]["evtWeight"].sum()
                            datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]*ds_factor

                    ids += 1

            #==========================================================================
            check_list = [True if len(datasets[input_name]["evtWeight"]) > 0 else False for input_name in input_list]
            #print("check_list", check_list)

            if len(input_list) > 1:
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
                            fac = 1/bin_Wsum
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
                                fac = 1/bin_Wsum
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
                                    fac = 1/bin_Wsum
                                    if math.isnan(fac):
                                        fac = 1
                                    var_weights[class_key][it,j,i,k] = fac
                                data_var.loc[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])) & ((data_var[var3] >= split3[k]) & (data_var[var3] < split3[k+1])), 'mvaWeight'] = data_var[((data_var[var1] >= split1[j]) & (data_var[var1] < split1[j+1])) & ((data_var[var2] >= split2[i]) & (data_var[var2] < split2[i+1])) & ((data_var[var3] >= split3[k]) & (data_var[var3] < split3[k+1]))]['mvaWeight']*fac

                    dataset['mvaWeight'] = np.array(data_var['mvaWeight']/data_var['mvaWeight'].sum())
                    del data_var

            #dataset['mvaWeight'] = dataset['mvaWeight']/dataset['mvaWeight'].sum()

            #==========================================================================
            if it == 0:
                dataset_train = dataset.copy()
                dataset_vec_train = dataset_vec.copy()
            elif it == 1:
                dataset_test = dataset.copy()
                dataset_vec_test = dataset_vec.copy()
            del dataset, dataset_vec

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
                vec_full_train[variable] = np.concatenate((vec_full_train[variable], dataset_vec_train[variable]), axis=0)
                vec_full_test[variable] = np.concatenate((vec_full_test[variable], dataset_vec_test[variable]), axis=0)

    if len(reweight_info) > 0 and not has_weights:
        reweight_info.append(["var_weights", var_weights])
    del dataset_train, dataset_test, dataset_vec_train, dataset_vec_test

    return ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, class_colors, reweight_info


#=====================================================================================================================
def join_datasets(ds, new_name, input_list, check_list, mode="scalars", combination="xsec", delete_inputs=True):

    datasets_list = []
    for i in range(len(input_list)):
        if check_list[i]:
            if mode == "scalars" and combination == "flat":
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

    if delete_inputs:
        if good_list:
            for input_name in input_list:
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


    def roc_plot(self, label='Signal-bkg ROC', color='blue', linestyle="-"):
        plt.plot(self.rej_others, self.eff_signal, color=color, label=label, linestyle=linestyle)


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
"""
def build_DANN(parameters, n_var, n_classes):
    #Creates three different models, one used for source only training, two used for domain adaptation

    # Base network -> x4
    for i in range(len(parameters[1])):
        if i == 0:
            inputs = Input(shape=(n_var,))
            x4 = Dense(parameters[1][i], activation=parameters[2])(inputs)
        if i > 0:
            x4 = Dense(parameters[1][i], activation=parameters[2])(x4)

    if parameters[4] == 'mixed':
        activ_source = 'softmax'
        activ_domain = 'sigmoid'
    else:
        activ_source = parameters[4]
        activ_domain = parameters[4]

    # Source network
    class_discriminator = Dense(n_classes, activation=activ_source, name="class")(x4)

    # Domain network
    #domain_discriminator = Dense(2, activation=activ_domain, name="domain")(x4)
    domain_discriminator = Dense(2, activation=activ_domain, name="domain")(class_discriminator)

    # Full model
    comb_model = Model(inputs=inputs, outputs=[class_discriminator, domain_discriminator])
    comb_model.compile(optimizer=parameters[3], loss={"class": 'categorical_crossentropy', "domain": 'categorical_crossentropy'}, loss_weights={"class": 1, "domain": 2}, metrics=['accuracy'], )

    # Source model
    class_discriminator_model = Model(inputs=inputs, outputs=[class_discriminator])
    class_discriminator_model.compile(optimizer=parameters[3], loss={"class": 'categorical_crossentropy'}, metrics=['accuracy'], )

    # Domain model
    domain_discriminator_model = Model(inputs=inputs, outputs=[domain_discriminator])
    domain_discriminator_model.compile(optimizer=parameters[3], loss={"domain": 'categorical_crossentropy'}, metrics=['accuracy'])

    return comb_model, class_discriminator_model, domain_discriminator_model
"""







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

    def forward(self, y_true, y_pred, weight):

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

    def forward(self, y_true, y_pred, weight):

        epsilon = 1e-7
        y_pred = (1-2*epsilon)*y_pred + epsilon

        y_true = torch.eye(self.num_classes)[y_true[:,0]]

        loss_n = -torch.sum(y_true*torch.log(y_pred), dim=-1).view(-1,1)

        total_ce_loss = torch.sum(loss_n*weight)
        num_of_samples = torch.sum(weight)
        mean_ce_loss = total_ce_loss / num_of_samples

        return mean_ce_loss



#=====================================================================================================================
# Torch activation functions
"""
class last_sigmoid(nn.Module): # not necessary!
    def __init__(self):
        super(last_sigmoid, self).__init__()

        self.activation = nn.Sigmoid()

    def forward(self, x):

        epsilon = 1e-7

        return (1-2*epsilon)*self.activation(x) + epsilon
"""



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
def build_model(model_type, parameters, n_var, n_classes, stat_values, variables, var_use, vec_variables, vec_var_use, vec_stat_values, device):

    if model_type == "NN":
        model = build_NN(parameters, n_var, n_classes, stat_values, device)
    elif model_type == "PNET":
        model = build_PNET(vec_variables, vec_var_use, n_classes, parameters, vec_stat_values, device)

    return model


#=====================================================================================================================
def update_model(model_type, model, criterion, parameters, batch_data, device):

    if model_type == "NN":
        model = update_NN(model, criterion, parameters, batch_data, device)
    elif model_type == "PNET":
        model = update_PNET(model, criterion, parameters, batch_data, device)

    return model


#=====================================================================================================================
def process_data(model_type, scalar_var, vector_var, variables, vec_variables, var_use, vec_var_use):

    if model_type == "NN":
        input_data = process_data_NN(scalar_var, variables)
    elif model_type == "PNET":
        input_data = process_data_PNET(scalar_var, vector_var, vec_variables, vec_var_use)

    return input_data


#=====================================================================================================================
def evaluate_model(model_type, input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode="predict"):

    if model_type == "NN":
        i_eval_output = evaluate_NN(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode)
    elif model_type == "PNET":
        i_eval_output = evaluate_PNET(input_data, model, i_eval, eval_step_size, criterion, parameters, device, mode)

    return i_eval_output


#=====================================================================================================================
def feature_score(model_type, input_data, model, min_loss, eval_step_size, criterion, parameters, variables, vec_variables, var_use, vec_var_use, var_names, vec_var_names, device):

    if model_type == "NN":
        feature_score_info = feature_score_NN(input_data, model, min_loss, eval_step_size, criterion, parameters, variables, var_names, device)
    elif model_type == "PNET":
        feature_score_info = feature_score_PNET(input_data, model, min_loss, eval_step_size, criterion, parameters, vec_variables, vec_var_use, vec_var_names, device)

    return feature_score_info


#=====================================================================================================================
def save_model(model_type, model, model_outpath, dim, vec_dim, device):

    if model_type == "NN":
        save_NN(model, model_outpath, dim, device)
    elif model_type == "PNET":
        save_PNET(model, model_outpath, vec_dim, device)



#=====================================================================================================================
def train_model(outpath_base, N_signal, train_frac, load_size, parameters, variables, var_names, var_use, classes, reweight_info, n_iterations = 5000, signal_param = None, mode = "torch", stat_values = None, vec_stat_values = None, eval_step_size = 0.2, feature_info = False, vec_variables=[], vec_var_names=[], vec_var_use=[], vec_types=[], early_stopping=300, device="cpu"):


    n_var = len(variables)
    n_classes = len(classes)


    model_type = parameters[0]
    batch_size = parameters[5]
    learning_rate = parameters[6]

    """
    if model_type == "PNN":
        # Get max and min param values
        if len(signal_param) == 1:
            dp0 = 0.1*(signal_param[0][-1] - signal_param[0][0])
            p0_min = signal_param[0][0] - dp0
            p0_max = signal_param[0][-1] + dp0
            #print("parameters limit:", p0_min, p0_max)

        if len(signal_param) == 2:
            dp0 = 0.1*(signal_param[0][-1] - signal_param[0][0])
            p0_min = signal_param[0][0] - dp0
            p0_max = signal_param[0][-1] + dp0
            dp1 = 0.1*(signal_param[1][-1] - signal_param[1][0])
            p1_min = signal_param[1][0] - dp1
            p1_max = signal_param[1][-1] + dp1
            #print("parameters limit:", p0_min, p0_max, p1_min, p1_max)
    """

    if mode == "torch":
        torch.set_num_threads(6)

        # Criterion
        if parameters[4] == 'cce':
            criterion = CCE_loss(num_classes=n_classes)
        elif parameters[4] == 'bce':
            criterion = BCE_loss()

        #---------------------------------------------------------------------------------------
        # NN torch training
        #---------------------------------------------------------------------------------------
        if model_type == "NN" or model_type == "PNN":

            #------------------------------------------------------------------------------------
            # Model
            class_discriminator_model = build_model(model_type, parameters, n_var, n_classes, stat_values, variables, var_use, vec_variables, vec_var_use, vec_stat_values, device)
            #class_discriminator_model = build_NN(parameters, n_var, n_classes, stat_values, device)
            if device == "cuda":
                class_discriminator_model = nn.DataParallel(class_discriminator_model) # Wrap the model with DataParallel
                class_discriminator_model = class_discriminator_model.to('cuda') # Move the model to the GPU
            #print(list(class_discriminator_model.parameters()))
            #print_model_parameters(class_discriminator_model)
            print(class_discriminator_model.parameters)

            #checkpoint_path='checkpoint_model.pt'
            checkpoint={'iteration':None, 'model_state_dict':None, 'optimizer_state_dict':None, 'loss': None}

            iteration = []
            train_acc = []
            test_acc = []
            train_loss = []
            test_loss = []

            best_weights = []
            position = 0
            min_loss = 99999
            early_stopping_count = 0
            load_it = 0
            for i in tqdm(range(n_iterations)):

                #===============================================================================
                # Load Datasets

                if (load_it == 0) or (period_count == waiting_period):
                    ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, colors, reweight_info = get_sample(outpath_base, parameters[7], classes, N_signal, train_frac, load_size, load_it, reweight_info, features=variables+["evtWeight"], vec_features=vec_variables)
                    load_it += 1
                    waiting_period = int(len(ds_full_train['mvaWeight'])/batch_size)
                    period_count = 0

                    train_data = process_data(model_type, ds_full_train, vec_full_train, variables, vec_variables, var_use, vec_var_use)
                    test_data = process_data(model_type, ds_full_test, vec_full_test, variables, vec_variables, var_use, vec_var_use)

                    del ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, colors


                    #===============================================================================
                    # Create batch samples
                    train_batches = batch_generator(train_data, batch_size)

                    n_eval_train_steps = int(len(train_data[-1])/eval_step_size) + 1
                    n_eval_test_steps = int(len(test_data[-1])/eval_step_size) + 1
                    train_w_sum = train_data[-1].sum()
                    test_w_sum = test_data[-1].sum()


                #===============================================================================
                """
                if model_type == "PNN":
                    # Produce random values for signal parameters in background events in batch
                    train_bkg_len = len(train_x_b[:,-1][train_y_b != 0])

                    if len(signal_param) == 1:
                        train_x_b[:,-1][train_y_b != 0] = p0_min + (p0_max - p0_min)*numpy_random.rand(train_bkg_len)

                    if len(signal_param) == 2:
                        train_x_b[:,-2][train_y_b != 0] = p0_min + (p0_max - p0_min)*numpy_random.rand(train_bkg_len)
                        train_x_b[:,-1][train_y_b != 0] = p1_min + (p1_max - p1_min)*numpy_random.rand(train_bkg_len)
                """

                #------------------------------------------------------------------------------------
                #Option available, see 3_2_Mini_Batch_Descent.py
                #trainloader = DataLoader(dataset = dataset, batch_size = 1)
                # Return randomly a sample with number of elements equals to the batch size
                #train_x_b, train_y_b, train_w_b = next(train_batches)
                period_count += 1

                #print("class_discriminator_model.parameters", i, class_discriminator_model.parameters())

                # Train model to learn class
                batch_data = next(train_batches)
                class_discriminator_model = update_model(model_type, class_discriminator_model, criterion, parameters, batch_data, device)

                #------------------------------------------------------------------------------------

                #if False:
                if ((i + 1) % 100 == 0):

                    """
                    if model_type == "PNN":
                        # Produce random values for signal parameters in background events for evaluation
                        train_bkg_len = len(eval_train_x[:,-1][eval_train_y != 0])

                        if len(signal_param) == 1:
                            eval_train_x[:,-1][eval_train_y != 0] = p0_min + (p0_max - p0_min)*numpy_random.rand(train_bkg_len)

                        if len(signal_param) == 2:
                            eval_train_x[:,-2][eval_train_y != 0] = p0_min + (p0_max - p0_min)*numpy_random.rand(train_bkg_len)
                            eval_train_x[:,-1][eval_train_y != 0] = p1_min + (p1_max - p1_min)*numpy_random.rand(train_bkg_len)
                    """

                    train_loss_i = 0
                    train_acc_i = 0
                    for i_eval in range(n_eval_train_steps):
                        i_eval_output = evaluate_model(model_type, train_data, class_discriminator_model, i_eval, eval_step_size, criterion, parameters, device, mode="metric")
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
                        i_eval_output = evaluate_model(model_type, test_data, class_discriminator_model, i_eval, eval_step_size, criterion, parameters, device, mode="metric")
                        if i_eval_output is None:
                            continue
                        else:
                            i_eval_loss, i_eval_acc = i_eval_output
                        test_loss_i += i_eval_loss
                        test_acc_i += i_eval_acc
                    test_loss_i = test_loss_i/test_w_sum
                    test_acc_i = test_acc_i/test_w_sum


                    #------------------------------------------------------------------------------------
                    iteration.append(i+1)
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

                    print("Iterations %d, class loss =  %.10f, class accuracy =  %.3f"%(i+1, test_loss_i, test_acc_i ))

                    if early_stopping_count == early_stopping:
                        print("Early stopping activated!")
                        break
                #elif ((i + 1) % 10 == 0):
                #    print("Iterations", i+1)

            #--------------------------------------------------------------------------------------
            if( position > 0 ):

                # Set weights of the best classification model
                #torch.save(checkpoint, checkpoint_path)
                class_discriminator_model.load_state_dict(checkpoint['model_state_dict'])
                #Resume training model with Checkpoints
                #checkpoint = torch.load(checkpoint_path)
                #model_checkpoint = linear_regression(1,1)
                #model_checkpoint.state_dict()
                #optimizer = optim.SGD(model_checkpoint.parameters(), lr = 1)
                #optimizer.state_dict()
                #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                #optimizer.state_dict()
                min_loss = checkpoint['loss']

                # Permutation feature importance
                # https://cms-ml.github.io/documentation/optimization/importance.html
                feature_score_info = []
                if feature_info:
                    print("")
                    print("Computing Feature Importance...")
                    feature_score_info = feature_score(model_type, test_data, class_discriminator_model, min_loss, eval_step_size, criterion, parameters, variables, vec_variables, var_use, vec_var_use, var_names, vec_var_names, device)

            adv_source_acc = np.zeros_like(test_acc)
            adv_target_acc = np.zeros_like(test_acc)




        #---------------------------------------------------------------------------------------
        # ParticleNet training
        #---------------------------------------------------------------------------------------
        if model_type == "PNET":

            #------------------------------------------------------------------------------------
            # Model
            class_discriminator_model = build_model(model_type, parameters, n_var, n_classes, stat_values, variables, var_use, vec_variables, vec_var_use, vec_stat_values, device)
            #class_discriminator_model = build_PNET(vec_variables, vec_var_use, n_classes, parameters, vec_stat_values, device)
            if device == "cuda":
                class_discriminator_model = nn.DataParallel(class_discriminator_model) # Wrap the model with DataParallel
                class_discriminator_model = class_discriminator_model.to('cuda') # Move the model to the GPU
            #print(list(class_discriminator_model.parameters()))
            #print_model_parameters(class_discriminator_model)
            print(class_discriminator_model.parameters)

            #checkpoint_path='checkpoint_model.pt'
            checkpoint={'iteration':None, 'model_state_dict':None, 'optimizer_state_dict':None, 'loss': None}

            """
            # Optimizer
            # https://machinelearningknowledge.ai/pytorch-optimizers-complete-guide-for-beginner/
            if parameters[3] == "adam":
                optimizer = torch.optim.Adam(class_discriminator_model.parameters(), lr=learning_rate, eps=1e-07)
                # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
            elif parameters[3] == "sgd":
                optimizer = torch.optim.SGD(class_discriminator_model.parameters(), lr=learning_rate)
                # lr=?, momentum=0, dampening=0, weight_decay=0, nesterov=False
            elif parameters[3] == "ranger":
                optimizer = Ranger(class_discriminator_model.parameters(), lr=learning_rate)
            """




            iteration = []
            train_acc = []
            test_acc = []
            train_loss = []
            test_loss = []

            best_weights = []
            position = 0
            min_loss = 99999
            early_stopping_count = 0
            load_it = 0
            for i in tqdm(range(n_iterations)):
                #===============================================================================
                # Load Datasets

                if (load_it == 0) or (period_count == waiting_period):
                    ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, colors, reweight_info = get_sample(outpath_base, parameters[7], classes, N_signal, train_frac, load_size, load_it, reweight_info, features=variables+["evtWeight"], vec_features=vec_variables)
                    load_it += 1
                    waiting_period = int(len(ds_full_train['mvaWeight'])/batch_size)
                    period_count = 0


                    train_data = process_data(model_type, ds_full_train, vec_full_train, variables, vec_variables, var_use, vec_var_use)
                    test_data = process_data(model_type, ds_full_test, vec_full_test, variables, vec_variables, var_use, vec_var_use)

                    train_pf_points, train_pf_features, train_sv_points, train_sv_features, train_y, train_w = train_data
                    test_pf_points, test_pf_features, test_sv_points, test_sv_features, test_y, test_w = test_data

                    del ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, colors


                    """
                    #------------------------------------------------------------------------------
                    train_y = np.array(ds_full_train['class']).ravel()
                    train_w = np.array(ds_full_train['mvaWeight']).ravel()

                    train_pf_points_list = []
                    train_pf_features_list = []
                    train_sv_points_list = []
                    train_sv_features_list = []
                    for ivar in range(len(vec_variables)):
                        if vec_variables[ivar].split("_")[0] == 'jetPFcand':
                            if "P" in vec_var_use[ivar]:
                                train_pf_points_list.append(vec_full_train[vec_variables[ivar]])
                            if "F" in vec_var_use[ivar]:
                                train_pf_features_list.append(vec_full_train[vec_variables[ivar]])
                        elif vec_variables[ivar].split("_")[0] == 'jetSV':
                            if "P" in vec_var_use[ivar]:
                                train_sv_points_list.append(vec_full_train[vec_variables[ivar]])
                            if "F" in vec_var_use[ivar]:
                                train_sv_features_list.append(vec_full_train[vec_variables[ivar]])
                        del vec_full_train[vec_variables[ivar]]

                    train_pf_points = torch.FloatTensor(np.stack(train_pf_points_list, axis=1))
                    del train_pf_points_list
                    train_pf_features = torch.FloatTensor(np.stack(train_pf_features_list, axis=1))
                    del train_pf_features_list
                    train_sv_points = torch.FloatTensor(np.stack(train_sv_points_list, axis=1))
                    del train_sv_points_list
                    train_sv_features = torch.FloatTensor(np.stack(train_sv_features_list, axis=1))
                    del train_sv_features_list

                    #print("train_pf_points", train_pf_points.shape)
                    #print("train_pf_features", train_pf_features.shape)
                    #print("train_sv_points", train_sv_points.shape)
                    #print("train_sv_features", train_sv_features.shape)
                    #------------------------------------------------------------------------------
                    """

                    """
                    #------------------------------------------------------------------------------
                    test_y = np.array(ds_full_test['class']).ravel()
                    test_w = np.array(ds_full_test['mvaWeight']).ravel()

                    test_pf_points_list = []
                    test_pf_features_list = []
                    test_sv_points_list = []
                    test_sv_features_list = []
                    for ivar in range(len(vec_variables)):
                        if vec_variables[ivar].split("_")[0] == 'jetPFcand':
                            if "P" in vec_var_use[ivar]:
                                test_pf_points_list.append(vec_full_test[vec_variables[ivar]])
                            if "F" in vec_var_use[ivar]:
                                test_pf_features_list.append(vec_full_test[vec_variables[ivar]])
                        elif vec_variables[ivar].split("_")[0] == 'jetSV':
                            if "P" in vec_var_use[ivar]:
                                test_sv_points_list.append(vec_full_test[vec_variables[ivar]])
                            if "F" in vec_var_use[ivar]:
                                test_sv_features_list.append(vec_full_test[vec_variables[ivar]])
                        del vec_full_test[vec_variables[ivar]]

                    test_pf_points = torch.FloatTensor(np.stack(test_pf_points_list, axis=1))
                    del test_pf_points_list
                    test_pf_features = torch.FloatTensor(np.stack(test_pf_features_list, axis=1))
                    del test_pf_features_list
                    test_sv_points = torch.FloatTensor(np.stack(test_sv_points_list, axis=1))
                    del test_sv_points_list
                    test_sv_features = torch.FloatTensor(np.stack(test_sv_features_list, axis=1))
                    del test_sv_features_list
                    # The output is the list that serves as input for batch_generator
                    #------------------------------------------------------------------------------
                    """

                    #===============================================================================
                    # Create batch samples
                    train_batches = batch_generator(train_data, batch_size)

                    n_eval_train_steps = int(len(train_w)/eval_step_size) + 1
                    last_eval_train_step = len(train_w)%eval_step_size
                    train_w_sum = train_w.sum()
                    n_eval_test_steps = int(len(test_w)/eval_step_size) + 1
                    last_eval_test_step = len(test_w)%eval_step_size
                    test_w_sum = test_w.sum()

                #===============================================================================


                #Option available, see 3_2_Mini_Batch_Descent.py
                #trainloader = DataLoader(dataset = dataset, batch_size = 1)
                # Return randomly a sample with number of elements equals to the batch size
                #pf_points, pf_features, sv_points, sv_features, train_y_b, train_w_b = next(train_batches)
                period_count += 1

                #------------------------------------------------------------------------------------
                batch_data = next(train_batches)
                class_discriminator_model = update_model(model_type, class_discriminator_model, criterion, parameters, batch_data, device)


                """
                if device == "cuda":
                    w = torch.FloatTensor(train_w_b).view(-1,1).to("cuda")
                    y = torch.tensor(train_y_b).view(-1,1).to("cuda")
                    pf_points = pf_points.to("cuda")
                    pf_features = pf_features.to("cuda")
                    sv_points = sv_points.to("cuda")
                    sv_features = sv_features.to("cuda")
                else:
                    w = torch.FloatTensor(train_w_b).view(-1,1)
                    y = torch.tensor(train_y_b).view(-1,1)

                pf_points.requires_grad=True
                pf_features.requires_grad=True
                sv_points.requires_grad=True
                sv_features.requires_grad=True
                pf_mask = (pf_features.abs().sum(dim=1, keepdim=True) != 0)
                sv_mask = (sv_features.abs().sum(dim=1, keepdim=True) != 0)

                #print("pf_points", pf_points.shape)
                #print("pf_features", pf_features.shape)
                #print("sv_points", sv_points.shape)
                #print("sv_features", sv_features.shape)
                #print("pf_mask", pf_mask.shape)
                #print("sv_mask", pf_mask.shape)

                yhat = class_discriminator_model(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)

                loss = criterion(y, yhat, w)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                """
                # input and output is the class_discriminator_model
                #----------------------------------------------------------------------------------

                if ((i + 1) % 100 == 0):

                    train_loss_i = 0
                    train_acc_i = 0
                    for i_eval in range(n_eval_train_steps):
                        i_eval_output = evaluate_model(model_type, train_data, class_discriminator_model, i_eval, eval_step_size, criterion, parameters, device, mode="metric")
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
                        i_eval_output = evaluate_model(model_type, test_data, class_discriminator_model, i_eval, eval_step_size, criterion, parameters, device, mode="metric")
                        if i_eval_output is None:
                            continue
                        else:
                            i_eval_loss, i_eval_acc = i_eval_output
                        test_loss_i += i_eval_loss
                        test_acc_i += i_eval_acc
                    test_loss_i = test_loss_i/test_w_sum
                    test_acc_i = test_acc_i/test_w_sum


                    #------------------------------------------------------------------------------------
                    iteration.append(i+1)
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

                    print("Iterations %d, class loss =  %.10f, class accuracy =  %.3f"%(i+1, test_loss_i, test_acc_i ))

                    if early_stopping_count == early_stopping:
                        print("Early stopping activated!")
                        break
                #elif ((i + 1) % 10 == 0):
                #    print("Iterations", i+1)

            #--------------------------------------------------------------------------------------
            if( position > 0 ):

                # Set weights of the best classification model
                #torch.save(checkpoint, checkpoint_path)
                class_discriminator_model.load_state_dict(checkpoint['model_state_dict'])
                #Resume training model with Checkpoints
                #checkpoint = torch.load(checkpoint_path)
                #model_checkpoint = linear_regression(1,1)
                #model_checkpoint.state_dict()
                #optimizer = optim.SGD(model_checkpoint.parameters(), lr = 1)
                #optimizer.state_dict()
                #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                #optimizer.state_dict()
                min_loss = checkpoint['loss']

                # Permutation feature importance
                # https://cms-ml.github.io/documentation/optimization/importance.html
                feature_score_info = []
                if feature_info:
                    print("")
                    print("Computing Feature Importance...")
                    feature_score_info = feature_score(model_type, test_data, class_discriminator_model, min_loss, eval_step_size, criterion, parameters, variables, vec_variables, var_use, vec_var_use, var_names, vec_var_names, device)

            adv_source_acc = np.zeros_like(test_acc)
            adv_target_acc = np.zeros_like(test_acc)


            """
            ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, colors, reweight_info = get_sample(outpath_base, parameters[7], classes, N_signal, train_frac, load_size, 0, reweight_info, features=variables+["evtWeight"], vec_features=vec_variables)

            train_y = ds_full_train['class']
            train_w = ds_full_train['mvaWeight']
            print("Labels shape = " + str(train_y.shape))
            print("Weights shape = " + str(train_w.shape))

            test_y = ds_full_test['class']
            test_w = ds_full_test['mvaWeight']


            pf_points_list = []
            pf_features_list = []
            sv_points_list = []
            sv_features_list = []
            for i in range(len(vec_variables)):
                if vec_variables[i].split("_")[0] == 'jetPFcand':
                    if "P" in vec_var_use[i]:
                        pf_points_list.append(vec_full_train[vec_variables[i]])
                    if "F" in vec_var_use[i]:
                        pf_features_list.append(vec_full_train[vec_variables[i]])
                elif vec_variables[i].split("_")[0] == 'jetSV':
                    if "P" in vec_var_use[i]:
                        sv_points_list.append(vec_full_train[vec_variables[i]])
                    if "F" in vec_var_use[i]:
                        sv_features_list.append(vec_full_train[vec_variables[i]])
                del vec_full_train[vec_variables[i]]

            pf_points = torch.FloatTensor(np.stack(pf_points_list, axis=1))
            del pf_points_list
            pf_features = torch.FloatTensor(np.stack(pf_features_list, axis=1))
            del pf_features_list
            sv_points = torch.FloatTensor(np.stack(sv_points_list, axis=1))
            del sv_points_list
            sv_features = torch.FloatTensor(np.stack(sv_features_list, axis=1))
            del sv_features_list

            print("pf_points", pf_points.shape)
            print("pf_features", pf_features.shape)
            print("sv_points", sv_points.shape)
            print("sv_features", sv_features.shape)

            train_PNET_batches = batch_generator([pf_points, pf_features, sv_points, sv_features, train_y, train_w], batch_size)

            pf_points_b, pf_features_b, sv_points_b, sv_features_b, train_y_b, train_w_b = next(train_PNET_batches)

            print("pf_points_b", pf_points_b.shape)
            print("pf_features_b", pf_features_b.shape)
            print("sv_points_b", sv_points_b.shape)
            print("sv_features_b", sv_features_b.shape)

            pf_mask_b = (pf_features_b.abs().sum(dim=1, keepdim=True) != 0)
            sv_mask_b = (sv_features_b.abs().sum(dim=1, keepdim=True) != 0)

            #print("pf_mask_b", pf_mask_b[:20])
            #print("sv_mask_b", pf_mask_b[:20])


            yPNET = class_discriminator_model(pf_points_b, pf_features_b, pf_mask_b, sv_points_b, sv_features_b, sv_mask_b)
            print("yPNET",yPNET[:20])

            """

            """
            class_PNET_model = build_PNET(vec_variables, vec_var_use, n_classes, parameters, vec_stat_values)
            #print(class_PNET_model.parameters)

            pf_points = torch.FloatTensor([[[-0.2815, -0.2784, -0.2824, -0.2722, -0.2632],
                    [-0.0259, -0.0302, -0.0208, -0.0431, -0.0149]],

                    [[-0.1361,  0.1651, -0.1308,  0.1552, -0.1370],
                    [ 0.0003, -0.0160, -0.0025, -0.0171,  0.0101]],

                    [[ 0.0938,  0.0927,  0.0995,  0.0744,  0.1037],
                    [ 0.0830,  0.0836,  0.0865,  0.0911,  0.0729]]])

            pf_features = torch.FloatTensor([[[ 5.0000e+00,  4.8961e+00,  4.6584e+00,  4.3057e+00,  3.1942e+00],
                    [ 5.0000e+00,  4.4631e+00,  4.2437e+00,  3.9313e+00,  2.9281e+00],
                    [-2.8152e-01, -2.7841e-01, -2.8243e-01, -2.7218e-01, -2.6321e-01],
                    [-2.5885e-02, -3.0182e-02, -2.0807e-02, -4.3073e-02, -1.4947e-02]],

                    [[ 5.0000e+00,  5.0000e+00,  5.0000e+00,  4.8643e+00,  4.7858e+00],
                    [ 4.7046e+00,  4.5979e+00,  4.1577e+00,  4.0529e+00,  3.8770e+00],
                    [-1.3611e-01,  1.6511e-01, -1.3080e-01,  1.5522e-01, -1.3702e-01],
                    [ 3.3735e-04, -1.5972e-02, -2.4947e-03, -1.7144e-02,  1.0103e-02]],

                    [[ 5.0000e+00,  4.8721e+00,  3.8756e+00,  3.5312e+00,  3.2650e+00],
                    [ 5.0000e+00,  4.0393e+00,  3.1304e+00,  2.8285e+00,  2.5735e+00],
                    [ 9.3840e-02,  9.2742e-02,  9.9517e-02,  7.4430e-02,  1.0373e-01],
                    [ 8.3036e-02,  8.3622e-02,  8.6454e-02,  9.1142e-02,  7.2879e-02]]])

            pf_mask = torch.FloatTensor([[[1., 1., 1., 1., 1.]],

                    [[1., 1., 1., 1., 1.]],

                    [[1., 1., 1., 1., 1.]]])

            sv_points = torch.FloatTensor([[[ 0.0000,  0.0000,  0.0000,  0.0000],
                    [ 0.0000,  0.0000,  0.0000,  0.0000]],

                    [[-0.1348,  0.0000,  0.0000,  0.0000],
                    [ 0.0112,  0.0000,  0.0000,  0.0000]],

                    [[-0.0085,  0.0000,  0.0000,  0.0000],
                    [-0.5672,  0.0000,  0.0000,  0.0000]]])

            sv_features = torch.FloatTensor([[[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                    [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                    [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                    [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00]],

                    [[ 1.0774e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                    [ 2.1013e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                    [-1.3475e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                    [ 1.1189e-02,  0.0000e+00,  0.0000e+00,  0.0000e+00]],

                    [[-1.6868e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                    [-8.1219e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                    [-8.5306e-03,  0.0000e+00,  0.0000e+00,  0.0000e+00],
                    [-5.6715e-01,  0.0000e+00,  0.0000e+00,  0.0000e+00]]])

            sv_mask = torch.FloatTensor([[[0., 0., 0., 0.]],

                    [[1., 0., 0., 0.]],

                    [[1., 0., 0., 0.]]])


            pf_mask_2 = (pf_features.abs().sum(dim=1, keepdim=True) != 0)
            sv_mask_2 = (sv_features.abs().sum(dim=1, keepdim=True) != 0)
            print("pf_mask_2", pf_mask_2)
            print("sv_mask_2", sv_mask_2)

            yPNET = class_PNET_model(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)
            print("yPNET",yPNET)
            """


    #plot_model(model, "plots/combined_model.pdf", show_shapes=True)
    #plot_model(class_discriminator_model, "plots/class_discriminator_model.pdf", show_shapes=True)
    #plot_model(domain_discriminator_model, "plots/domain_discriminator_model.pdf", show_shapes=True)


    return class_discriminator_model, np.array(iteration), np.array(train_acc), np.array(test_acc), np.array(train_loss), np.array(test_loss), np.array(adv_source_acc), np.array(adv_target_acc), feature_score_info
