#--------------------------------------------------------------------------------------------------
# General Setup
#--------------------------------------------------------------------------------------------------
analysis = "OPENDATA"
selection = "MLOD"
periods = ['12']
tag = "Class"


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
learning_rate = [ 0.01, 0.1 ]#, 0.001 ]


#--------------------------------------------------------------------------------------------------
# Training setup
#--------------------------------------------------------------------------------------------------
batch_size = [ 1000, 9000 ]
load_size = 500000
train_frac = 0.5
eval_step_size = 1000
num_max_iterations = 2000
early_stopping = 20


#--------------------------------------------------------------------------------------------------
# Inputs setup
#--------------------------------------------------------------------------------------------------
input_mode = "normal" #"parameterized"
feature_info = False

scalar_variables = [
    ["MuonL_pt",                r"$p_\mathrm{T}(\mu)$"],
    ["MET_pt",                  r"$p_\mathrm{T}^\mathrm{miss}$"],
    #["HT30",                    r"HT30"],
    #["MHT30",                   r"MHT30"],
    #["LeadingJet_pt",           r"LeadingJet_pt"],
    #["TauH_pt",                 r"TauH_pt"],
    ["TauH_MuonL_M",            r"$M(\mu,\tau_h)$"],
    #["TauH_MuonL_pt",           r"TauH_MuonL_pt"],
    ["MuonL_MET_pt",            r"$p_\mathrm{T}(\mu,\vec{p}_\mathrm{T}^\mathrm{miss})$"],
    ["MuonL_MET_dphi",          r"$\Delta \phi(\mu,\vec{p}_\mathrm{T}^\mathrm{miss})$"],
    ["MuonL_MET_Mt",            r"$M_\mathrm{T}(\mu,\vec{p}_\mathrm{T}^\mathrm{miss})$"],
    #["TauH_TauL_pt",            r"TauH_TauL_pt"],
    ["TauH_TauL_Mt",            r"$M_\mathrm{T}((\mu,\vec{p}_\mathrm{T}^\mathrm{miss}),\tau_h)$"],
    #["TauH_MuonL_dr",           r"TauH_MuonL_dr"],
    #["LeadingJet_MuonL_dr",     r"LeadingJet_MuonL_dr"],
    #["LeadingJet_TauL_dphi",    r"LeadingJet_TauL_dphi"],
    #["LeadingJet_TauH_dr",      r"LeadingJet_TauH_dr"],
    ["LeadingJet_TauHMuonL_dr", r"$\Delta R((\mu,\tau_h),\mathrm{Jet}_L)$"],
    ]

vector_variables = []

reweight_variables = []

input_parameters = [
    ["m_H", r"$m_H$"],
    ["m_a", r"$m_a$"],
    ]


#--------------------------------------------------------------------------------------------------
# Classes setup
#--------------------------------------------------------------------------------------------------

"""
classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal_VBF": [["Signal_VBF"], "normal", "xsec", r"$qq \rightarrow qqH$", "green"],
"Signal_GluGlu": [["Signal_GluGlu"], "normal", "xsec", r"$gg \rightarrow H$", "skyblue"],
"Bkg": [["DYJetsToLL", "TTbar", "W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu"], "normal", "xsec", "Background", "red"],
}
"""

classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal": [["Signal_VBF", "Signal_GluGlu"], "normal", "xsec", "Signal", "green"],
"Bkg": [["DYJetsToLL", "TTbar", "W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu"], "normal", "xsec", "Background", "red"],
}


"""
classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal_samples": [["Signal_VBF", "Signal_GluGlu"], "normal", "ignored", "ignored", "green"],
"Bkg": [["DYJetsToLL", "TTbar", "W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu"], "normal", "xsec", "Bkg", "red"],
}
"""

"""
classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal_samples": [["Signal_VBF", "Signal_GluGlu"], "normal", "ignored", "ignored", "green"],
"DYJetsToLL": [["DYJetsToLL"], "normal", "xsec", "DYJetsToLL", "red"],
"TTbar": [["TTbar"], "normal", "xsec", "TTbar", "skyblue"],
"WJetsToLNu": [["W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu"], "normal", "xsec", "WJetsToLNu", "darkgoldenrod"],
}
"""

"""
classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal_samples_1": [signal_list_1, "normal", "ignored", "ignored", "green"],
"Signal_samples_2": [signal_list_2, "normal", "ignored", "ignored", "limegreen"],
"DYJetsToLL": [["DYJetsToLL"], "normal", "xsec", "Bkg", "red"],
"TTbar": [["TTbar"], "normal", "xsec", "Bkg", "skyblue"],
"WJetsToLNu": [["W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu"], "normal", "xsec", "Bkg", "darkgoldenrod"],
}


classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal_parameterized_1": [signal_list_1, "normal", "flat", "ignored", "green"],
"Signal_parameterized_2": [signal_list_2, "normal", "flat", "ignored", "limegreen"],
"DYJetsToLL": [["DYJetsToLL"], "normal", "xsec", "Bkg", "red"],
"TTbar": [["TTbar"], "normal", "xsec", "Bkg", "skyblue"],
"WJetsToLNu": [["W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu"], "normal", "xsec", "Bkg", "darkgoldenrod"],
}
"""






# Signal classes must start with "Signal"

# If a class has only one process, the class name must be equal to the process name

# Parameterized signal must have a class name starting with "Signal_parameterized"

# More than one signal class is allowed

# If a class name starts with "Signal_samples", the models will be trained to each signal point separately. In addition, combination and label are ignored.

# If two or more class names start with "Signal_samples", the signal points from these classes are paired together during the loop

# The code support a maximum of 2 reweight variables
