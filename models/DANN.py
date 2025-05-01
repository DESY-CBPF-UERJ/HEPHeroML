import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.ticker import AutoMinorLocator
import torch
import torch.nn as nn
import sys
import os
from tqdm import tqdm
from statsmodels.stats.weightstats import DescrStatsW
numpy_random = np.random.RandomState(16)
sys.path.append("..")
from custom_opts.ranger import Ranger
import tools







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










