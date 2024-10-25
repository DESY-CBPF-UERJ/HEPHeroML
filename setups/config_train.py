

#-------------------------------------------------------------------------------
# [DO NOT TOUCH THIS PART] 
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--job", type=int, default=0)
parser.add_argument("--check", dest='check_flag', action='store_true')
parser.set_defaults(check_flag=False)
parser.add_argument("--clean", dest='clean_flag', action='store_true')
parser.set_defaults(clean_flag=False)

args = parser.parse_args()


#outpath = "/home/gilson/cernbox/HEP/ANALYSIS"
#outpath_base = os.path.join(outpath, analysis, selection, "datasets")
outpath_base = "/home/gilson/datasets"

#===============================================================================
# CHECK ARGUMENT
#===============================================================================
#inpath = "files"

has_signal_list = False
N_signal_points = 1
Signal_class = None
for key in classes:
    if key.startswith("Signal"):
        Signal_class = key
        if key.startswith("Signal_samples"):
            has_signal_list = True
            N_signal_points = len(classes[key][0])
            break

modelName = []
model = []
for i_signal in range(N_signal_points):
    for i_NN_type in NN_type:
        for i_num_layers in num_layers:
            for i_num_nodes in num_nodes:
                for i_activation_func in activation_func:
                    for i_optimizer in optimizer:
                        for i_loss_func in loss_func:
                            for i_batch_size in batch_size:
                                for i_lr in learning_rate:
                                    for i_period in periods:
                                        modelName.append(i_NN_type+"_"+str(i_num_layers)+"_"+str(i_num_nodes)+"_"+i_activation_func+"_"+i_optimizer+"_"+i_loss_func+"_"+str(i_batch_size)+"_"+str(i_lr).replace(".", "p")+"_"+str(i_period))
                                        if N_signal_points == 1:
                                            model.append([i_NN_type] + [[i_num_nodes for i in range(i_num_layers)]] + [i_activation_func] + [i_optimizer] + [i_loss_func] + [i_batch_size] + [i_lr] + [i_period] + [Signal_class])
                                        else:
                                            model.append([i_NN_type] + [[i_num_nodes for i in range(i_num_layers)]] + [i_activation_func] + [i_optimizer] + [i_loss_func] + [i_batch_size] + [i_lr] + [i_period] + [classes[Signal_class][0][i_signal]])

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


N_signal = int(N/(len(model)/N_signal_points))
if N_signal_points == 1:
    signal_tag = Signal_class
else:
    signal_tag = classes[Signal_class][0][N_signal]

#===============================================================================
# Output setup
#===============================================================================
if args.clean_flag:
    os.system("rm -rf " + os.path.join(outpath_base, model[N][7], "ML", library, tag, signal_tag))
    sys.exit()

ml_outpath = os.path.join(outpath_base, model[N][7], "ML")
if not os.path.exists(ml_outpath):
    os.makedirs(ml_outpath)

signal_outpath = os.path.join(ml_outpath, library, tag, signal_tag)
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


#===============================================================================
import torch


variables = [scalar_variables[i][0] for i in range(len(scalar_variables))]
var_names = [scalar_variables[i][1] for i in range(len(scalar_variables))]

vec_variables = [vector_variables[i][0] for i in range(len(vector_variables))]
vec_var_names = [vector_variables[i][1] for i in range(len(vector_variables))]

signal_parameters = [input_parameters[i][0] for i in range(len(input_parameters))]
signal_parameters_names = [input_parameters[i][1] for i in range(len(input_parameters))]


if input_mode == "parameterized":
    if len(signal_parameters) > 2:
        sys.exit("Code does not support more than 2 signal parameters!")
        # It can be extended for more than 2 variables
    else:
        variables = variables + signal_parameters
        var_names = var_names + signal_parameters_names




#===============================================================================
# Preprocessing input data (modify and stay)
#===============================================================================
print("")
print("Preprocessing input data...")

seed = 16


ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, colors = get_sample(outpath_base, model[N][7], classes, N_signal, train_frac, load_size, 0, features=variables+["evtWeight"], vec_features=vec_variables, reweight_info=reweight_variables)

ds_full_train = pd.DataFrame.from_dict(ds_full_train)
ds_full_test = pd.DataFrame.from_dict(ds_full_test)


n_classes = len(classes)

signal_param = []



#=================================================================================

mean = []
std = []
for i in range(len(variables)):
    weighted_stats = DescrStatsW(ds_full_train[variables[i]], weights=ds_full_train["mvaWeight"], ddof=0)
    mean.append(weighted_stats.mean)
    std.append(weighted_stats.std)
print("mean: " + str(mean))
print("std: " + str(std))


stat_values={"mean": mean, "std": std}


#==================================================================================





#===============================================================================
# Plot training and test distributions (modify and stay - only for batch size?)
#===============================================================================
for ivar in range(len(variables)):

    fig1 = plt.figure(figsize=(10,7))
    gs1 = gs.GridSpec(1, 1)
    #==================================================
    ax1 = plt.subplot(gs1[0])
    #==================================================
    var = variables[ivar]
    if library == "keras":
        bins = np.linspace(-2.5,2.5,51)
    elif library == "torch":
        bins = np.linspace(mean[ivar]-2.5*std[ivar],mean[ivar]+2.5*std[ivar],51)
    for ikey in range(len(class_names)):
        step_plot( ax1, var, ds_full_train[ds_full_train["class"] == ikey], label=class_labels[ikey]+" (train)", color=colors[ikey], weight="mvaWeight", bins=bins, error=True )
        step_plot( ax1, var, ds_full_test[ds_full_test["class"] == ikey], label=class_labels[ikey]+" (test)", color=colors[ikey], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )
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



del ds_full_train, ds_full_test, class_names, class_labels, colors

if args.check_flag:
    sys.exit()



#===============================================================================
# RUN TRAINING
#===============================================================================
print("")
print("Training...")

start = time.time()


class_model, iteration, train_acc, test_acc, train_loss, test_loss, adv_source_acc, adv_target_acc, features_score, features_score_unc = train_model(
    outpath_base,
    N_signal,
    train_frac,
    load_size,
    model[N],
    variables,
    classes,
    n_iterations = num_max_iterations,
    signal_param = signal_param,
    mode = library,
    stat_values = stat_values,
    eval_step_size = eval_step_size,
    feature_info = feature_info,
    reweight_variables=reweight_variables,
    early_stopping=early_stopping,
    )



if library == "keras":
    class_model.save(os.path.join(model_outpath, "model.h5"))
elif library == "torch":
    #torch.save(class_model, os.path.join(model_outpath, "model.pt"))
    model_scripted = torch.jit.script(class_model) # Export to TorchScript
    model_scripted.save(os.path.join(model_outpath, "model_scripted.pt"))


if feature_info:
    #===============================================================================
    # SAVE FEATURE IMPORTANCE INFORMATION
    #===============================================================================
    df_feature = pd.DataFrame(list(zip(variables, features_score, features_score_unc, var_names)),columns=["Feature", "Score", "Score_unc", "Feature_name"])
    df_feature = df_feature.sort_values("Score", ascending=False)
    df_feature.to_csv(os.path.join(model_outpath, 'features.csv'), index=False)

    fig1 = plt.figure(figsize=(9,5))
    grid = [1, 1]
    gs1 = gs.GridSpec(grid[0], grid[1])
    #-----------------------------------------------------------------------------------------------------------------
    # Accuracy
    #-----------------------------------------------------------------------------------------------------------------
    ax1 = plt.subplot(gs1[0])
    y_pos = np.arange(len(variables))
    ax1.barh(y_pos, df_feature["Score"], xerr=df_feature["Score_unc"], align='center', color="lightsteelblue")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_feature["Feature_name"], size=18)
    ax1.invert_yaxis()  # labels read top-to-bottom
    ax1.set_xlabel("Feature score", size=14, horizontalalignment='right', x=1.0)

    ax1.tick_params(which='major', length=8)
    ax1.tick_params(which='minor', length=4)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.grid(which='major', axis='x', linewidth=0.2, linestyle='-', color='0.75')
    ax1.spines['bottom'].set_linewidth(1)
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.margins(x=0)
    ax1.legend(numpoints=1, ncol=1, prop={'size': 10.5}, frameon=False, loc='lower right')


    plt.subplots_adjust(left=0.15, bottom=0.115, right=0.990, top=0.95, wspace=0.18, hspace=0.165)
    plt.savefig(os.path.join(model_outpath, "features.png"))


#===============================================================================
# SAVE TRAINING INFORMATION
#===============================================================================
df_training = pd.DataFrame(list(zip(iteration, train_acc, test_acc, train_loss, test_loss, adv_source_acc, adv_target_acc)),columns=["iteration", "train_acc", "test_acc", "train_loss", "test_loss", "adv_source_acc", "adv_target_acc"])

df_training.to_csv(os.path.join(model_outpath, 'training.csv'), index=False)

#iteration = df_training['iteration']
#train_acc = df_training['train_acc']
#test_acc = df_training['test_acc']
#train_loss = df_training['train_loss']
#test_loss = df_training['test_loss']
#adv_source_acc = df_training['adv_source_acc']
#adv_target_acc = df_training['adv_target_acc']
#adv_sum_acc = np.array(df_training['adv_source_acc']) + np.array(df_training['adv_target_acc'])

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
#plt.yscale('log')
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


#===============================================================================
# CHECK OVERTRAINING
#===============================================================================
ds_full_train, ds_full_test, vec_full_train, vec_full_test, class_names, class_labels, colors = get_sample(outpath_base, model[N][7], classes, N_signal, train_frac, load_size, 0, features=variables+["evtWeight"], vec_features=vec_variables, reweight_info=reweight_variables)

ds_full_train = pd.DataFrame.from_dict(ds_full_train)
ds_full_test = pd.DataFrame.from_dict(ds_full_test)

for i in range(n_classes):
    pred_name = 'score_C'+str(i)
    ds_full_train[pred_name] = 0.
    ds_full_test[pred_name] = 0.

for ikey in range(len(class_names)):
    train_x = ds_full_train[ds_full_train["class"] == ikey][variables]
    train_x = train_x.values
    test_x = ds_full_test[ds_full_test["class"] == ikey][variables]
    test_x = test_x.values


    n_eval_train_steps = int(len(train_x)/eval_step_size) + 1
    last_eval_train_step = len(train_x)%eval_step_size
    n_eval_test_steps = int(len(test_x)/eval_step_size) + 1
    last_eval_test_step = len(test_x)%eval_step_size

    train_class_pred = []
    for i_eval in range(n_eval_train_steps):
        if i_eval < n_eval_train_steps-1:
            eval_train_x = train_x[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        else:
            eval_train_x = train_x[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_train_step]

        if library == "keras":
            i_train_class_pred = class_model.predict(eval_train_x)
        elif library == "torch":
            i_train_class_pred = model_scripted(torch.FloatTensor(eval_train_x)).detach().numpy()

        train_class_pred = train_class_pred + i_train_class_pred.tolist()
    train_class_pred = np.array(train_class_pred)

    test_class_pred = []
    for i_eval in range(n_eval_test_steps):
        if i_eval < n_eval_test_steps-1:
            eval_test_x = test_x[i_eval*eval_step_size:(i_eval+1)*eval_step_size]
        else:
            eval_test_x = test_x[i_eval*eval_step_size:(i_eval*eval_step_size)+last_eval_test_step]

        if library == "keras":
            i_test_class_pred = class_model.predict(eval_test_x)
        elif library == "torch":
            i_test_class_pred = model_scripted(torch.FloatTensor(eval_test_x)).detach().numpy()

        test_class_pred = test_class_pred + i_test_class_pred.tolist()
    test_class_pred = np.array(test_class_pred)


    if model[N][4] == "cce":
        n_outputs = n_classes
        for i in range(n_outputs):
            pred_name = 'score_C'+str(i)
            ds_full_test.loc[ds_full_test["class"] == ikey, pred_name] = test_class_pred[:,i]
            ds_full_train.loc[ds_full_train["class"] == ikey, pred_name] = train_class_pred[:,i]
    if model[N][4] == "bce":
        n_outputs = 1
        pred_name = 'score_C0'
        ds_full_test.loc[ds_full_test["class"] == ikey, pred_name] = 1 - test_class_pred[:,0]
        ds_full_train.loc[ds_full_train["class"] == ikey, pred_name] = 1 - train_class_pred[:,0]

for i in range(n_outputs):
    fig1 = plt.figure(figsize=(20,7))
    gs1 = gs.GridSpec(1,1)
    #==================================================
    ax1 = plt.subplot(gs1[0])
    #==================================================
    var = 'score_C'+str(i)
    bins = np.linspace(0,1,51)
    for ikey in range(len(class_names)):
        step_plot( ax1, var, ds_full_train[ds_full_train["class"] == ikey], label=class_labels[ikey]+" (train)", color=colors[ikey], weight="mvaWeight", bins=bins, error=True )
        step_plot( ax1, var, ds_full_test[ds_full_test["class"] == ikey], label=class_labels[ikey]+" (test)", color=colors[ikey], weight="mvaWeight", bins=bins, error=True, linestyle='dotted' )
    ax1.set_xlabel(class_names[i] + " score", size=14, horizontalalignment='right', x=1.0)
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

del ds_full_train, ds_full_test, class_names, class_labels, colors
"""
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
"""


#===============================================================================
end = time.time()
hours = int((end - start)/3600)
minutes = int(((end - start)%3600)/60)
seconds = int(((end - start)%3600)%60)

print("")
print("-----------------------------------------------------------------------------------")
print("Total process duration: " + str(hours) + " hours " + str(minutes) + " minutes " + str(seconds) + " seconds")
print("-----------------------------------------------------------------------------------")
print("")


