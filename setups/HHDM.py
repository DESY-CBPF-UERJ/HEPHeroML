#--------------------------------------------------------------------------------------------------
# General Setup
#--------------------------------------------------------------------------------------------------
analysis = "HHDM"
selection = "ML"
periods = ['APV_16', '16', '17', '18']
tag = "single"


#--------------------------------------------------------------------------------------------------
# Models setup
#--------------------------------------------------------------------------------------------------
library = "torch"  # torch, keras
NN_type = [ "NN" ]  # [ "NN", "DANN", "PNN" ]
num_layers = [2, 3]
num_nodes = [20, 30, 40, 50]#, 50] #, 500, 1000, 2000, 5000 ]
activation_func = [ "elu" ] # [ "elu", "relu", "tanh" ]
optimizer = [ "adam" ] # [ "adam", "sgd" ]
loss_func = [ "bce" ] # [ "bce", "cce" ]
learning_rate = [ 0.01 ]#, 0.1, 0.001 ]


#--------------------------------------------------------------------------------------------------
# Training setup
#--------------------------------------------------------------------------------------------------
batch_size = [ 1000 ]
load_size = 500000
train_frac = 0.5
eval_step_size = 1000
num_max_iterations = 2000


#--------------------------------------------------------------------------------------------------
# Inputs setup
#--------------------------------------------------------------------------------------------------
input_mode = "normal" #"parameterized"
feature_info = False

input_variables = [
    ["LeadingLep_pt",           r"$\mathrm{leading}\,p_\mathrm{T}^\mathrm{l}$"],
    ["TrailingLep_pt",          r"$\mathrm{trailing}\,p_\mathrm{T}^\mathrm{l}$"],
    ["LepLep_pt",               r"$p_\mathrm{T}^\mathrm{ll}$"],
    ["LepLep_deltaR",           r"$\Delta R^\mathrm{ll}$"],
    ["LepLep_deltaM",           r"$\Delta M^\mathrm{ll}$"],
    ["MET_pt",                  r"$E_\mathrm{T}^\mathrm{miss}$"],
    ["MET_LepLep_Mt",           r"$M_\mathrm{T}^\mathrm{ll,MET}$"],
    ["MET_LepLep_deltaPhi",     r"$\Delta \phi^\mathrm{ll,MET}$"],
    ["MT2LL",                   r"$M_\mathrm{T2}^\mathrm{ll}$"],
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

input_parameters = [
    ["m_H", r"$m_H$"],
    ["m_a", r"$m_a$"],
    ]


#--------------------------------------------------------------------------------------------------
# Classes setup
#--------------------------------------------------------------------------------------------------
classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal_samples": [[
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
    #"Signal_1400_100",
    #"Signal_1400_400",
    #"Signal_1400_600",
    #"Signal_1400_1000",
    #"Signal_1400_1200",
    #"Signal_2000_100",
    #"Signal_2000_400",
    #"Signal_2000_600",
    #"Signal_2000_1000",
    #"Signal_2000_1200",
    #"Signal_2000_1800",
    ], "normal", "flat", "Signal", "green"],
"Background": [[
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
    "ST_s-channel",
    "ST_t-channel_top",
    "ST_t-channel_antitop",
    "WZTo3LNu",
    "ZZTo4L",
    "ZZTo2L2Nu",
    "WZZ",
    "WWZ",
    "ZZZ",
    "WWW",
    "TTWZ",
    "TTZZ",
    "TTWW",
    "TTWJetsToLNu",
    "TTWJetsToQQ",
    "TTZToQQ",
    "TTZToLL",
    "TTZToNuNu",
    "tZq_ll",
    "WWTo2L2Nu",
    "WZTo2Q2L",
    "ZZTo2Q2L",
    ], "normal", "xsec", "Background", "red"],
}


# Signal class names must start with "Signal"

# If a class has only one process, the class name must be equal to the process name

# Parameterized signal must have a class name starting with "Signal_parameterized"

# More than one signal class is allowed

# If a class name starts with "Signal_samples", the models will be trained to each signal point separately. In addition, combination and label are ignored.

# If two or more class names start with "Signal_samples", the signal points from these classes are paired together during the loop
