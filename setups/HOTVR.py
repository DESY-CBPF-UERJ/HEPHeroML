#--------------------------------------------------------------------------------------------------
# General Setup
#--------------------------------------------------------------------------------------------------
analysis = "HOTVR_ML_R3"
selection = "Skimming"
periods = ['0_22']
tag = "Test"


#--------------------------------------------------------------------------------------------------
# Models setup
#--------------------------------------------------------------------------------------------------
library = "torch"  # torch, keras
NN_type = [ "NN" ]  # [ "NN", "DANN", "PNN" ]
num_layers = [2, 3]
num_nodes = [20, 30, 40, 50, 60]#, 50] #, 500, 1000, 2000, 5000 ]
activation_func = [ "elu" ] # [ "elu", "relu", "tanh" ]
optimizer = [ "adam" ] # [ "adam", "sgd" ]
loss_func = [ "bce" ] # [ "bce", "cce" ]
learning_rate = [ 0.01 ]#, 0.1, 0.001 ]


#--------------------------------------------------------------------------------------------------
# Training setup
#--------------------------------------------------------------------------------------------------
batch_size = [ 1000 ]
load_size = 50000
train_frac = 0.5
eval_step_size = 1000
num_max_iterations = 11000
early_stopping = 20


#--------------------------------------------------------------------------------------------------
# Inputs setup
#--------------------------------------------------------------------------------------------------
input_mode = "normal" #"parameterized"
feature_info = False

input_variables = [
    ["jet_pt",           "Jet_pt"],
    ["jet_eta",          "Jet_eta"],
    ["jet_mass",         "Jet_mass"],
    #["LepLep_deltaR",           r"$\Delta R^\mathrm{ll}$"],
    #["LepLep_deltaM",           r"$\Delta M^\mathrm{ll}$"],
    #["MET_pt",                  r"$E_\mathrm{T}^\mathrm{miss}$"],
    #["MET_LepLep_Mt",           r"$M_\mathrm{T}^\mathrm{ll,MET}$"],
    #["MET_LepLep_deltaPhi",     r"$\Delta \phi^\mathrm{ll,MET}$"],
    #["MT2LL",                   r"$M_\mathrm{T2}^\mathrm{ll}$"],
    #["Nbjets",                  r"$N_\mathrm{b\,jets}$"],
    #["Njets_forward",           r"Njets_forward"],
    #["Dijet_deltaEta",          r"$\Delta\eta^\mathrm{jj}$"],
    ##["Dijet_pt",                r"$p_\mathrm{T}^\mathrm{jj}$"],
    ##["Dijet_M",                 r"$M^\mathrm{jj}$"],
    #["HT30",                    r"HT30"],
    #["MHT30",                   r"MHT30"],
    #["OmegaMin30",              r"OmegaMin30"],
    #["FMax30",                  r"FMax30"],
    ]

reweight_variables = [
    ["jet_pt",      [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 99999999.]],
    #["jet_mass",    [-99999999., -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 99999999.]],
    ]

#jet_pt: [15, 20, 26, 35, 46, 61, 80, 106, 141, 186, 247, 326, 432, 571, 756, 1000]
#jet_abseta: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.7]

input_parameters = [
    ["m_H", r"$m_H$"],
    ["m_a", r"$m_a$"],
    ]


#--------------------------------------------------------------------------------------------------
# Classes setup
#--------------------------------------------------------------------------------------------------
classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal_sample": [[
    "Zto2Q_PTQQ-100to200",
    "Zto2Q_PTQQ-200to400",
    "Zto2Q_PTQQ-400to600",
    "Zto2Q_PTQQ-600"
    ], "normal", "flat", "Signal", "green"],
"Background": [[
    "QCD_PT-120to170",
    "QCD_PT-470to600",
    "QCD_PT-1000to1400",
    "QCD_PT-2400to3200"
    ], "normal", "flat", "Background", "red"],
}


# Signal class names must start with "Signal"

# If a class has only one process, the class name must be equal to the process name

# Parameterized signal must have a class name starting with "Signal_parameterized"

# More than one signal class is allowed

# If a class name starts with "Signal_samples", the models will be trained to each signal point separately. In addition, combination and label are ignored.

# If two or more class names start with "Signal_samples", the signal points from these classes are paired together during the loop

# The code support a maximum of 2 reweight variables
