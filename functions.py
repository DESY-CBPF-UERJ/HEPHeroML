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
def get_sample(basedir, period, classes, n_signal, train_frac, load_size, load_it, features=[]):

    class_names = []
    class_labels = []
    class_colors = []
    control = True
    for class_key in classes:

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

        print("")
        print("Loading datasets...")

        datasets_dir = os.path.join(basedir, period)
        datasets_abspath = [(f, os.path.join(datasets_dir, f)) for f in os.listdir(datasets_dir)]

        #=================================================================================
        datasets_length = []
        datasets_evtWsum = []
        n_datasets = 0
        for dataset, abspath in tqdm(datasets_abspath):
            dataset_name = dataset.split(".")[0]

            if dataset.endswith(".h5") and dataset_name in input_list:
                with h5py.File(abspath) as f:
                    datasets_length.append(len(np.array(f["scalars/evtWeight"])))
                    datasets_evtWsum.append(np.array(f["scalars/evtWeight"]).sum())
                n_datasets += 1
        print("datasets_length", datasets_length)

        #if events_slice is not None:
        if combination == "flat":
            #datasets_length = np.array(datasets_length)
            #total_length = datasets_length.sum()
            #datasets_frac = datasets_length/total_length
            datasets_frac = np.ones(n_datasets)*(1./n_datasets)
            print("datasets_frac", datasets_frac)
        elif combination == "xsec":
            datasets_evtWsum = np.array(datasets_evtWsum)
            total_evtWsum = datasets_evtWsum.sum()
            datasets_frac = datasets_evtWsum/total_evtWsum
            print("datasets_frac", datasets_frac)

        #=================================================================================
        class_load_size = int(load_size/len(classes))

        datasets_entries = datasets_frac*class_load_size
        datasets_entries = np.array([ int(i) for i in datasets_entries])
        datasets_entries = np.minimum(datasets_entries, datasets_length)
        print("datasets_entries", datasets_entries)

        datasets_train_entries = train_frac*datasets_entries
        datasets_train_entries = np.array([ int(i) for i in datasets_train_entries])
        datasets_test_entries = datasets_entries - datasets_train_entries
        print("datasets_train_entries", datasets_train_entries)
        print("datasets_test_entries", datasets_test_entries)

        datasets_nSlices = np.array([int(datasets_length[i]/datasets_entries[i]) if datasets_entries[i] > 0 else 0 for i in range(len(datasets_length))])
        print("datasets_nSlices", datasets_nSlices)

        datasets_slices = load_it%datasets_nSlices
        print("datasets_slices", datasets_slices)

        datasets_train_limits = [[datasets_slices[i]*datasets_entries[i], datasets_train_entries[i]+datasets_slices[i]*datasets_entries[i]] for i in range(len(datasets_slices))]
        datasets_test_limits = [[datasets_train_entries[i]+datasets_slices[i]*datasets_entries[i], (datasets_slices[i]+1)*datasets_entries[i]] for i in range(len(datasets_slices))]

        print("datasets_train_limits", datasets_train_limits)
        print("datasets_test_limits", datasets_test_limits)

        #=================================================================================

        for it in range(2):
            datasets = {}
            ids = 0
            for dataset, abspath in tqdm(datasets_abspath):
                dataset_name = dataset.split(".")[0]

                if dataset.endswith(".h5") and dataset_name in input_list:

                    if mode == "normal" or mode == "scalars":
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
                                if mode == "normal":
                                    datasets[dataset_name] = pd.DataFrame(variables_dict)
                                if mode == "scalars":
                                    datasets[dataset_name] = variables_dict
                            else:
                                print("Warning: Dataset " + dataset_name + " is empty!")

                    if mode == "vectors":
                        variables_dict = {}
                        with h5py.File(abspath) as f:
                            if "vectors" in f.keys():
                                group = "vectors"
                                for variable in f[group].keys():
                                    if len(features) == 0 or variable in features:
                                        if it == 0:
                                            variables_dict[variable] = np.array(f[group+"/"+variable])[datasets_train_limits[ids][0]:datasets_train_limits[ids][1]]
                                        elif it == 1:
                                            variables_dict[variable] = np.array(f[group+"/"+variable])[datasets_test_limits[ids][0]:datasets_test_limits[ids][1]]
                                datasets[dataset_name] = variables_dict
                            else:
                                print("Warning: Dataset " + dataset_name + " is empty!")

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

                    if len(datasets[dataset_name]["evtWeight"]) > 0:
                        if combination == "flat":
                            datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]/datasets[dataset_name]["evtWeight"].sum()
                        elif combination == "xsec":
                            ds_factor = datasets_evtWsum[ids]/datasets[dataset_name]["evtWeight"].sum()
                            datasets[dataset_name]["evtWeight"] = datasets[dataset_name]["evtWeight"]*ds_factor

                    ids += 1

            #==========================================================================

            if len(input_list) > 1:
                join_datasets(datasets, class_name, input_list, mode=mode, combination=combination)
            #==========================================================================

            ikey = 0
            for key in classes:
                if key == class_key:
                    break
                ikey += 1


            dataset = datasets[class_name].sample(frac=1, random_state=seed)
            del datasets
            dataset = dataset.reset_index(drop=True)
            dataset["class"] = ikey
            dataset['mvaWeight'] = dataset['evtWeight']/dataset['evtWeight'].sum()


            #==========================================================================

            #jet_pt: [15, 20, 26, 35, 46, 61, 80, 106, 141, 186, 247, 326, 432, 571, 756, 1000]
            #jet_abseta: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.7]
            #jet_pt: [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 99999999.]
            #jet_abseta: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 99999999.]

            """
            split1 = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1200, 1400, 1700, 99999999.]
            var1 = 'jet_pt'
            for j in range(len(split1)-1):
                bin_Wsum = dataset[((dataset[var1] >= split1[j]) & (dataset[var1] < split1[j+1]))]['mvaWeight'].sum()
                fac = 1/bin_Wsum
                if math.isnan(fac):
                    fac = 1
                dataset.loc[((dataset[var1] >= split1[j]) & (dataset[var1] < split1[j+1])), 'mvaWeight'] = dataset[((dataset[var1] >= split1[j]) & (dataset[var1] < split1[j+1]))]['mvaWeight']*fac
            """

            """
            split2 = [-99999999., -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 99999999.]
            var2 = 'jet_eta'
            split1 = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 99999999.]
            var1 = 'jet_pt'
            for j in range(len(split1)-1):
                for i in range(len(split2)-1):
                    bin_Wsum = dataset[((dataset[var1] >= split1[j]) & (dataset[var1] < split1[j+1])) & ((dataset[var2] >= split2[i]) & (dataset[var2] < split2[i+1]))]['mvaWeight'].sum()
                    fac = 1/bin_Wsum
                    if math.isnan(fac):
                        fac = 1
                    dataset.loc[((dataset[var1] >= split1[j]) & (dataset[var1] < split1[j+1])) & ((dataset[var2] >= split2[i]) & (dataset[var2] < split2[i+1])), 'mvaWeight'] = dataset[((dataset[var1] >= split1[j]) & (dataset[var1] < split1[j+1])) & ((dataset[var2] >= split2[i]) & (dataset[var2] < split2[i+1]))]['mvaWeight']*fac
            """

            dataset['mvaWeight'] = dataset['mvaWeight']/dataset['mvaWeight'].sum()

            #==========================================================================
            if it == 0:
                dataset_train = dataset.copy()
            elif it == 1:
                dataset_test = dataset.copy()
            del dataset

        class_names.append(class_name)
        class_labels.append(class_label)
        class_colors.append(class_color)

        if control:
            ds_full_train = dataset_train.copy()
            ds_full_test = dataset_test.copy()
            control = False
        else:
            ds_full_train = pd.concat([ds_full_train, dataset_train])
            ds_full_test = pd.concat([ds_full_test, dataset_test])

    del dataset_train, dataset_test

    return ds_full_train, ds_full_test, class_names, class_labels, class_colors


#=====================================================================================================================
def join_datasets(ds, new_name, input_list, mode="normal", combination="xsec", delete_inputs=True):

    datasets_list = []
    for input_name in input_list:
        if len(ds[input_name]["evtWeight"]) > 0:
            if combination == "flat":
                ds[input_name].loc[:,"evtWeight"] = ds[input_name]["evtWeight"]/ds[input_name]["evtWeight"].sum()
            datasets_list.append(ds[input_name])

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
class torch_NN(nn.Module):
    # Constructor
    def __init__(self, parameters, n_var, n_classes, stat_values):
        super(torch_NN, self).__init__()

        self.mean = torch.tensor(stat_values["mean"], dtype=torch.float32)
        self.std = torch.tensor(stat_values["std"], dtype=torch.float32)

        if parameters[2] == 'relu':
            self.activation_hidden = nn.ReLU()
        elif parameters[2] == 'tanh':
            self.activation_hidden = nn.Tanh()
        elif parameters[2] == 'elu':
            self.activation_hidden = nn.ELU()
        else:
            print("Error: hidden activation function not supported!")

        if parameters[4] == 'cce':
            self.activation_last = nn.Softmax(dim=1)
            n_output = n_classes
        elif parameters[4] == 'bce' and n_classes == 2:
            self.activation_last = nn.Sigmoid()
            n_output = 1
        else:
            print("Error: last activation function or number of classes is not supported!")

        self.hidden = nn.ModuleList()
        for i in range(len(parameters[1])):
            if i == 0:
                self.hidden.append(nn.Linear(n_var, parameters[1][i]))
            if i > 0:
                self.hidden.append(nn.Linear(parameters[1][i-1], parameters[1][i]))
        self.hidden.append(nn.Linear(parameters[1][-1], n_output))

    # Prediction
    def forward(self, x):

        x = (x - self.mean) / self.std

        N_layers = len(self.hidden)
        for i, layer in enumerate(self.hidden):
            if i < N_layers-1:
                x = self.activation_hidden(layer(x))
            else:
                x = self.activation_last(layer(x))

        return x


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
def train_model(outpath_base, N_signal, train_frac, load_size, parameters, variables, classes, n_iterations = 5000, signal_param = None, mode = "keras", stat_values = None, eval_step_size = 0.2, feature_info = False):


    n_var = len(variables)
    n_classes = len(classes)


    model_type = parameters[0]
    batch_size = parameters[5]
    learning_rate = parameters[6]

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


    if mode == "torch":
        torch.set_num_threads(6)

        #---------------------------------------------------------------------------------------
        # NN torch training
        #---------------------------------------------------------------------------------------
        if model_type == "NN" or model_type == "PNN":

            #------------------------------------------------------------------------------------
            # Model
            class_discriminator_model = torch_NN(parameters, n_var, n_classes, stat_values)
            #print(list(class_discriminator_model.parameters()))
            #print_model_parameters(class_discriminator_model)
            print(class_discriminator_model.parameters)

            # Criterion
            if parameters[4] == 'cce':
                criterion = CCE_loss(num_classes=n_classes)
            elif parameters[4] == 'bce':
                criterion = BCE_loss()

            # Optimizer
            # https://machinelearningknowledge.ai/pytorch-optimizers-complete-guide-for-beginner/
            if parameters[3] == "adam":
                optimizer = torch.optim.Adam(class_discriminator_model.parameters(), lr=learning_rate, eps=1e-07)
                # lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False
            if parameters[3] == "sgd":
                optimizer = torch.optim.SGD(class_discriminator_model.parameters(), lr=learning_rate)
                # lr=?, momentum=0, dampening=0, weight_decay=0, nesterov=False

            print("")
            print(optimizer.state_dict())

            #------------------------------------------------------------------------------------


            #------------------------------------------------------------------------------------
            # Import the libraries and set the random seed
            #torch.manual_seed(1)

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
            for i in range(n_iterations):

                #===============================================================================
                # Load Datasets

                if (load_it == 0) or (period_count == waiting_period):
                    ds_full_train, ds_full_test, class_names, class_labels, colors = get_sample(outpath_base, parameters[7], classes, N_signal, train_frac, load_size, load_it, features=variables+["evtWeight"])
                    load_it += 1
                    waiting_period = int(len(ds_full_train)/batch_size)
                    period_count = 0

                    ds_full_train = ds_full_train.sample(frac=1, random_state=seed)
                    train_x = ds_full_train[variables]
                    train_x = train_x.values
                    train_y = np.array(ds_full_train['class']).ravel()
                    train_w = np.array(ds_full_train['mvaWeight']).ravel()                    # weight to signal x bkg comparison
                    print("Variables shape = " + str(train_x.shape))
                    print("Labels shape = " + str(train_y.shape))
                    print("Weights shape = " + str(train_w.shape))

                    ds_full_test = ds_full_test.sample(frac=1, random_state=seed)
                    test_x = ds_full_test[variables]
                    test_x = test_x.values
                    test_y = np.array(ds_full_test['class']).ravel()
                    test_w = np.array(ds_full_test['mvaWeight']).ravel()                      # weight to signal x bkg comparison

                    #df_source = pd.read_csv(os.path.join(inpath,"source.csv"))
                    df_source = ds_full_train.copy()
                    df_source = df_source.sample(frac=1, random_state=seed)
                    source_x = df_source[variables]
                    source_x = source_x.values
                    source_w = np.array(df_source['mvaWeight']).ravel()                  # weight to source x target comparison

                    #df_target = pd.read_csv(os.path.join(inpath,"target.csv"))
                    df_target = ds_full_test.copy()
                    df_target = df_target.sample(frac=1, random_state=seed)
                    target_x = df_target[variables]
                    target_x = target_x.values
                    target_w = np.array(df_target['mvaWeight']).ravel()                  # weight to source x target comparison

                    #del ds_full_train, ds_full_test, class_names, class_labels, colors

                    #===============================================================================
                    # Create batch samples
                    train_batches = batch_generator([train_x, train_y, train_w], batch_size)

                    #eval_train_batches = batch_generator([train_x, train_y, train_w], int(len(train_x)*eval_step_size))
                    #eval_test_batches = batch_generator([test_x, test_y, test_w], int(len(train_x)*eval_step_size))
                    #eval_step_size = 1000
                    n_eval_train_steps = int(len(train_x)/eval_step_size) + 1
                    last_eval_train_step = len(train_x)%eval_step_size
                    train_w_sum = train_w.sum()
                    n_eval_test_steps = int(len(test_x)/eval_step_size) + 1
                    last_eval_test_step = len(test_x)%eval_step_size
                    test_w_sum = test_w.sum()


                #===============================================================================

                #------------------------------------------------------------------------------------
                #Option available, see 3_2_Mini_Batch_Descent.py
                #trainloader = DataLoader(dataset = dataset, batch_size = 1)
                # Return randomly a sample with number of elements equals to the batch size
                train_x_b, train_y_b, train_w_b = next(train_batches)
                period_count += 1

                if model_type == "PNN":
                    # Produce random values for signal parameters in background events in batch
                    train_bkg_len = len(train_x_b[:,-1][train_y_b != 0])

                    if len(signal_param) == 1:
                        train_x_b[:,-1][train_y_b != 0] = p0_min + (p0_max - p0_min)*numpy_random.rand(train_bkg_len)

                    if len(signal_param) == 2:
                        train_x_b[:,-2][train_y_b != 0] = p0_min + (p0_max - p0_min)*numpy_random.rand(train_bkg_len)
                        train_x_b[:,-1][train_y_b != 0] = p1_min + (p1_max - p1_min)*numpy_random.rand(train_bkg_len)

                #------------------------------------------------------------------------------------
                # Train model to learn class

                w = torch.FloatTensor(train_w_b).view(-1,1)
                x = torch.FloatTensor(train_x_b)
                x.requires_grad=True
                y = torch.tensor(train_y_b).view(-1,1)

                yhat = class_discriminator_model(x)
                #print("yhat type ", yhat.dtype)

                loss = criterion(y, yhat, w)
                #print("Loss = ", loss.item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                #if False:
                if ((i + 1) % 10 == 0):

                    #print("Evaluating!!!")
                    #eval_train_x, eval_train_y, eval_train_w = next(eval_train_batches)
                    #eval_test_x, eval_test_y, eval_test_w = next(eval_test_batches)

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
                        if i_eval < n_eval_train_steps-1:
                            eval_train_x = train_x[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
                            eval_train_y = train_y[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
                            eval_train_w = train_w[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
                        elif last_eval_train_step > 0:
                            eval_train_x = train_x[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_train_step]
                            eval_train_y = train_y[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_train_step]
                            eval_train_w = train_w[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_train_step]
                        else:
                            continue

                        eval_train_yhat = class_discriminator_model(torch.FloatTensor(eval_train_x))
                        eval_train_w_sum = eval_train_w.sum()
                        train_loss_i += eval_train_w_sum*criterion(torch.tensor(eval_train_y).view(-1,1), eval_train_yhat, torch.FloatTensor(eval_train_w).view(-1,1)).item()
                        if parameters[4] == 'cce':
                            train_acc_i += eval_train_w_sum*np.average(eval_train_y == eval_train_yhat.max(1)[1].numpy(), weights=eval_train_w)
                        elif parameters[4] == 'bce':
                            train_acc_i += eval_train_w_sum*np.average(eval_train_y == (eval_train_yhat[:, 0] > 0.5).numpy(), weights=eval_train_w)
                        del eval_train_yhat
                    train_loss_i = train_loss_i/train_w_sum
                    train_acc_i = train_acc_i/train_w_sum


                    test_loss_i = 0
                    test_acc_i = 0
                    for i_eval in range(n_eval_test_steps):
                        if i_eval < n_eval_test_steps-1:
                            eval_test_x = test_x[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
                            eval_test_y = test_y[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
                            eval_test_w = test_w[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
                        elif last_eval_train_step > 0:
                            eval_test_x = test_x[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_test_step]
                            eval_test_y = test_y[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_test_step]
                            eval_test_w = test_w[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_test_step]
                        else:
                            continue

                        eval_test_yhat = class_discriminator_model(torch.FloatTensor(eval_test_x))
                        eval_test_w_sum = eval_test_w.sum()
                        test_loss_i += eval_test_w_sum*criterion(torch.tensor(eval_test_y).view(-1,1), eval_test_yhat, torch.FloatTensor(eval_test_w).view(-1,1)).item()
                        if parameters[4] == 'cce':
                            test_acc_i += eval_test_w_sum*np.average(eval_test_y == eval_test_yhat.max(1)[1].numpy(), weights=eval_test_w)
                        elif parameters[4] == 'bce':
                            test_acc_i += eval_test_w_sum*np.average(eval_test_y == (eval_test_yhat[:, 0] > 0.5).numpy(), weights=eval_test_w)
                        del eval_test_yhat
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

                    if early_stopping_count == 30:
                        print("Early stopping activated!")
                        break

            if( position > 0 ):
                #------------------------------------------------------------------------------------
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
                features_score = []
                features_score_unc = []
                if feature_info:
                    print("")
                    print("Computing Feature Importance...")
                    for ivar in tqdm(range(n_var)):
                        losses = []
                        for irep in range(30):

                            test_x_shuffled = test_x.copy()
                            numpy_random.shuffle(test_x_shuffled[:,ivar])

                            test_loss_i = 0
                            for i_eval in range(n_eval_test_steps):
                                if i_eval < n_eval_test_steps-1:
                                    eval_test_x = test_x_shuffled[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
                                    eval_test_y = test_y[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
                                    eval_test_w = test_w[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
                                else:
                                    eval_test_x = test_x_shuffled[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_test_step]
                                    eval_test_y = test_y[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_test_step]
                                    eval_test_w = test_w[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_test_step]

                                eval_test_yhat = class_discriminator_model(torch.FloatTensor(eval_test_x))
                                eval_test_w_sum = eval_test_w.sum()
                                test_loss_i += eval_test_w_sum*criterion(torch.tensor(eval_test_y).view(-1,1), eval_test_yhat, torch.FloatTensor(eval_test_w).view(-1,1)).item()
                                del eval_test_yhat
                            test_loss_i = test_loss_i/test_w_sum

                            losses.append(test_loss_i)
                        losses = np.array(losses)
                        mean_loss = np.mean(losses)
                        std_loss = np.std(losses)

                        features_score.append(np.around((mean_loss - min_loss)/np.abs(min_loss), decimals=3))
                        features_score_unc.append(np.around(std_loss/np.abs(min_loss), decimals=3))

            adv_source_acc = np.zeros_like(test_acc)
            adv_target_acc = np.zeros_like(test_acc)


    #plot_model(model, "plots/combined_model.pdf", show_shapes=True)
    #plot_model(class_discriminator_model, "plots/class_discriminator_model.pdf", show_shapes=True)
    #plot_model(domain_discriminator_model, "plots/domain_discriminator_model.pdf", show_shapes=True)


    return class_discriminator_model, np.array(iteration), np.array(train_acc), np.array(test_acc), np.array(train_loss), np.array(test_loss), np.array(adv_source_acc), np.array(adv_target_acc), np.array(features_score), np.array(features_score_unc)

