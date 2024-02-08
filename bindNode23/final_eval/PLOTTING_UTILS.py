import numpy as np

import importlib
from config import FileManager, FileSetter
import data_preparation
importlib.reload(data_preparation)
from data_preparation import ProteinInformation, ProteinResults
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
cmap = sns.color_palette('colorblind', n_colors=5)
cmap = ["#56B4E9","#F0E442",  "#009E73","#E69F00", "#CC79A7"]
from scipy.stats import bootstrap


import random
def sample_with_replacement_k_times(bindlist,total,  n=100):
    resultarr = np.zeros((n))
    for bootstrap in range(n):
        sample = random.choices(bindlist, k=total)
        unique, counts = np.unique(np.array(sample), return_counts=True)
        
        try:
            cov = counts[1]/total
        except IndexError:
            cov = counts[0]/total
        #print(cov)
        resultarr[bootstrap]=cov
    return resultarr

def get_bootstrap_error(share, ligand):
    
    totals = {"metal": 579, "nucleic":174, "small":826, "overall":1009}
    
    total = totals[ligand]
    protnum = int(total*share)
    bindings = [1]*int(protnum)
    nonbindings = [0]*int(total-protnum)
    samplelist = bindings + nonbindings
    print(f"Entering with total:{total}, bindings {protnum}")
    data = sample_with_replacement_k_times(samplelist, total)
    data = (data,)
    bootstrap_ci = bootstrap(data, np.mean, confidence_level=0.95,
                         random_state=1, method='percentile')
    print(bootstrap_ci.confidence_interval)
    CI = (bootstrap_ci.confidence_interval.high-bootstrap_ci.confidence_interval.low)
    print("calculated CI with:",CI)
    return CI

    

def plot_multiple_results(
    ax, metrics, model_results, ligand, x_pos, cmap=cmap
):
    
    xpos = np.arange(len(metrics))
    
    for xpos, model in enumerate(model_results):
        results = [model[ligand][metric] for metric in metrics]
        errors = [model[ligand][f"{metric}_CI"] for metric in metrics]
        labels = metrics
                
        width = 0.8 / len(model_results)
        ax[0].bar(
            x_pos +(len(model_results)*width/2)- xpos*(width),
            results,
            width=width,
            ecolor="black",
            capsize=12,
            yerr=errors,
            color=cmap[xpos]
        )
        ax[0].set_ylabel("Mean Performance", size=20)

        ax[0].set_xticks(x_pos)
        ax[0].set_xticklabels(labels, size=20)
        ax[0].yaxis.grid(True)
    
        covonebind = [model[ligand]["cov_percentage"]]
        bootstrap_error = get_bootstrap_error(covonebind[0],ligand)
        ax[1].bar(
            0 +(len(model_results)*width/2) - xpos*(width),
            covonebind,
            width=width,
            ecolor="black",
            yerr=bootstrap_error,
            capsize=10,
            color=cmap[xpos]
        )
        ax[1].set_ylabel("CovOneBind", size=20)
        ax[1].set_xticks([0])
        ax[1].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

        # ax[1].set_xticks([0])
        #ax[1].tick_params(axis='both', which='minor', labelsize=15)
        #ax[1].set_xticklabels(["model"], size=15)
    return ax


# section for final plotting of DevSet1009
gcndata = json.load(open("/home/frank/bind/bindPredict/predictions/T5_GCNConv_128feat.json"))
oridata = json.load(open("/home/frank/bind/backup/final_eval/bindEmbed21DL_128_1/bindEmbed21DL_T5_embeds.json"))
sagedata = json.load(open("/home/frank/bind/backup/final_eval/BindNode23SAGEConv_128_1/results_2023-03-28-13-38.json"))
gatdata = json.load(open("/home/frank/bind/backup/final_eval/BindNode23SAGEConvGATMLP_320_4/results_2023-03-28-13-01.json"))
mlpdata = json.load(open("/home/frank/bind/backup/final_eval/BindNode23SAGEConvMLP_320_1/results_2023-03-28-13-26.json"))


plt.cla()
plt.clf()

fig = plt.figure(constrained_layout=True)
#fig.suptitle('Model performances on DevSet1009', size=25)
fig.set_size_inches(10, 15)
plt.rcParams.update({'font.size': 15})

# create 3x1 subfigs https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
subtitles = ["nucleic", "small", "metal", "overall"]
subfigs = fig.subfigures(nrows=4, ncols=1)
for row, subfig in enumerate(subfigs):
    subfig.suptitle(subtitles[row], size=20 ) #, fontweight='bold')

    # create 1x3 subplots per subfig
    ax = subfig.subplots(nrows=1, ncols=2, width_ratios=[2, 1])
    xpos = np.arange(2)
    ax = plot_multiple_results(ax, ["MCC", "F1"], [oridata, gcndata, sagedata, mlpdata, gatdata], subtitles[row], xpos, cmap = cmap)


#fig.tight_layout()
fig.legend(["bindEmbed21DL", "GCNBaseline", "SAGEConv", "SAGEConvMLP", "SAGEConvGATMLP"], bbox_to_anchor=(0.65, 0), shadow=False, fontsize=15)
# plt.tight_layout
plt.savefig("compare_models.png", bbox_inches='tight',dpi=100)



#########################################################################################
#   ########    ########    ########    ########                                        #
#       #       #           #               #                                           #
#       #       ######      ########        #                                           #
#       #       #                  #        #                                           #
#       #       ######      ########        #                                           #
#########################################################################################
# section for final plotting of TestSet1009
oridata = json.load(open("/home/frank/bind/bindPredict/predictions/TESTSET_BINDEMBED21.json"))
mlpdata = json.load(open("/home/frank/bind/bindPredict/predictions/TESTSET_SAGECONVMLP.json"))
cmap = ["#56B4E9","#E69F00","#F0E442",  "#009E73","#E69F00", "#CC79A7"]

plt.cla()
plt.clf()

fig = plt.figure(constrained_layout=True)
#fig.suptitle('Model performances on DevSet1009', size=25)
fig.set_size_inches(10, 15)
plt.rcParams.update({'font.size': 15})

# create 3x1 subfigs https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
subtitles = ["nucleic", "small", "metal", "overall"]
subfigs = fig.subfigures(nrows=4, ncols=1)
for row, subfig in enumerate(subfigs):
    subfig.suptitle(subtitles[row], size=20 ) #, fontweight='bold')

    # create 1x3 subplots per subfig
    ax = subfig.subplots(nrows=1, ncols=2, width_ratios=[2, 1])
    xpos = np.arange(2)
    ax = plot_multiple_results(ax, ["MCC", "F1"], [oridata, mlpdata], subtitles[row], xpos, cmap = cmap)


#fig.tight_layout()
fig.legend(["bindEmbed21DL", "SAGEConvMLP"], bbox_to_anchor=(0.65, 0), shadow=False, fontsize=15)
# plt.tight_layout
plt.savefig("PLOT_TEST_Results.png", bbox_inches='tight',dpi=100)





# CASE RESAMPLING BOOTSTRAPPING
def get_bootstrap_error(share, ligand):
    
    totals = {"metal": 579, "nucleic":174, "small":826, "overall":1009}
    
    total = totals[ligand]
    protnum = int(total*share)
    bindings = [1]*int(protnum)
    nonbindings = [0]*int(total-protnum)
    samplelist = bindings + nonbindings
    # print(f"Entering with total:{total}, bindings {protnum}")
    data = sample_with_replacement_k_times(samplelist, total)
    data = (data,)
    bootstrap_ci = bootstrap(data, np.mean, confidence_level=0.95,
                         random_state=1, method='percentile')
    #print(bootstrap_ci.confidence_interval)
    CI = (bootstrap_ci.confidence_interval.high-bootstrap_ci.confidence_interval.low)
    # print("calculated CI with:",CI)
    return CI

def sample_with_replacement_k_times(bindlist,total,  n=100):
    maxobserved = 0
    minobserved = 1000
    resultarr = np.zeros((n))
    for bootstrap in range(n):
        sample = random.choices(bindlist, k=total)
        unique, counts = np.unique(np.array(sample), return_counts=True)
        
        try:
            cov = counts[1]/total
            if counts[1]<minobserved:
                #print(f"Observing new minimum: {minobserved}")
                minobserved=counts[1]
            if counts[1]>maxobserved:
                #print(f"Observing new MAX: {maxobserved}")
                maxobserved=counts[1]
        except IndexError:
            cov = counts[0]/total
        #print(cov)
        resultarr[bootstrap]=cov
    #print("MEAN:",np.mean(resultarr))
    return resultarr


def give_all_CIs_from_file(file_in, ligand):

    subtitles = ["nucleic", "small", "metal", "overall"]
    covonebind = file_in[ligand]["cov_percentage"]
    print(ligand)
    print(round(get_bootstrap_error(share=covonebind, ligand=ligand)*100/2, 2))
    print(" ")


for file in [oridata, gcndata, sagedata, mlpdata, gatdata]:
    if 'architecture' in file.keys():
        print(file["architecture"])
    for ligand in ["nucleic", "small", "metal", "overall"]:
        ciinpercent = give_all_CIs_from_file(file, ligand=ligand)

# testing 
for file in [oridata, mlpdata]:
    if 'architecture' in file.keys():
        print(file["architecture"])
    for ligand in ["nucleic", "small", "metal", "overall"]:
        ciinpercent = give_all_CIs_from_file(file, ligand=ligand)
# section to create the tables in overleaf
results = [oridata, mlpdata]


for result in results:
    scores = [round(result[ligand]['F1'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    cis = [round(result[ligand]['F1_CI'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    mccs =  [round(result[ligand]['MCC'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    mcis = [round(result[ligand]['MCC_CI'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    for k, score in enumerate(scores):
        print(f"& {int(score*100)} \pm {int(cis[k]*100)} \%", end="")
    print("\\")
    for k, score in enumerate(mccs):
        print(f"& {int(score*100)} \pm {int(mcis[k]*100)} \%", end="")
    print("\\")

for result in results:
    scores = [round(result[ligand]['cov_percentage'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    
    for k, score in enumerate(scores):
        print(f"& {int(score*100)} \%", end="")

    print("\\")


# section for final tables of baselines

mlpdata = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/BindNode23SAGEConvMLP_320_1/results_2023-03-28-13-26.json"))
zeror = json.load(open("/home/frank/bind/bindPredict/predictions/ZEROR_BASELINE_results.json"))
randomr = json.load(open("/home/frank/bind/bindPredict/predictions/RANDOMRATE_results.json"))
rand_embd = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/BindNode23SAGEConvGATMLP_RANDOM_EMBD_320_1/results_2023-04-03-13-10.json"))
rand_struc = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/BindNode23SAGEConvGATMLP_RANDOM_STRUC_320_1/results_2023-04-03-13-00.json"))

results = [mlpdata, zeror, randomr, rand_embd, rand_struc]


for result in results:
    scores = [round(result[ligand]['F1'], 2) for ligand in ['overall']]
    cis = [round(result[ligand]['F1_CI'], 2) for ligand in ['overall']]
    mccs =  [round(result[ligand]['MCC'], 2) for ligand in ['overall']]
    mcis = [round(result[ligand]['MCC_CI'], 2) for ligand in ['overall']]
    for k, score in enumerate(scores):
        print(f"& {int(score*100)} \pm {int(cis[k]*100)} \%", end="")
    print("\\")
    for k, score in enumerate(mccs):
        print(f"& {int(score*100)} \pm {int(mcis[k]*100)} \%", end="")
    print("\\")



"""Observations from case studies on proteins:
metal has no min-neighbor binding residues (might be pockets)
"P03882" has a whole side-branch for nucleic that does not get detected by bindembed21, but this method does
found it bc: has min number of neighbors
my method predicts MORE residues to be binding than the anno, but there is reason to doubt that. see image

same as above for P0A6X7, see images. bindEmbed misses most of the branch

for metal, found with max neighbors, prediction does also pretty well

https://www.ncbi.nlm.nih.gov/Structure/icn3d/full.html?mmdbafid=Q8TL28&bu=1
select -> advanced 
:227 or :228 or :231 or :237 or :253 or :257 or :290
"""

# section for plotting F1 vs distance
from matplotlib import pyplot as plt
import numpy as np
cmap = sns.color_palette('colorblind', n_colors=2)
baselinedata = json.load(open("/home/frank/bind/bindPredict/predictions/GCNCONV_DIST1_1024_noBN.json"))
five = json.load(open("/home/frank/bind/bindPredict/predictions/GCN_dist5_results_2023-05-05-20-37.json"))
nine = json.load(open("/home/frank/bind/bindPredict/predictions/GCN_dist9.json"))
ft = json.load(open("/home/frank/bind/bindPredict/predictions/GCN_dist14.json"))
results = [baselinedata, five, nine, ft]
#some example data
x = np.array([1,5,9,14])
f1 = np.array([result["overall"]["F1"] for result in results])
f1_ci = np.array([result["overall"]["F1_CI"] for result in results])
mcc = np.array([result["overall"]["MCC"] for result in results])
mcc_ci = np.array([result["overall"]["MCC_CI"] for result in results])
#some confidence interval
fig, ax = plt.subplots()
ax.plot(x,f1, color=cmap[0])
ax.plot(x,mcc, color=cmap[1])
fig.legend(["F1", "MCC"], bbox_to_anchor=(1.2, 0.6))
ax.fill_between(x, (f1-f1_ci), (f1+f1_ci), color=cmap[0], alpha=.1)
ax.fill_between(x, (mcc-mcc_ci), (mcc+mcc_ci), color=cmap[1], alpha=.1)
# fig.suptitle('GCNConv performance drops with bigger neighborhood', size=20)
ax.set_xticklabels(x)
ax.set_xticks(x)
ax.set_xlabel("Distance cutoff [Å]", fontsize=20)
ax.set_ylabel("Mean Performance", fontsize=20)

plt.savefig("PLOTDIST.png", bbox_inches='tight',dpi=100)



# section to create the tables in overleaf
sagedata = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/BindNode23SAGEConv_128_1/results_2023-03-28-13-38.json"))
gatdata = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/BindNode23SAGEConvGATMLP_320_4/results_2023-03-28-13-01.json"))
mlpdata = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/BindNode23SAGEConvMLP_320_1/results_2023-03-28-13-26.json"))
baselinedata = json.load(open("/home/frank/bind/bindPredict/predictions/GCNCONV_DIST1_1024_noBN.json"))
oridata = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/bindEmbed21DL_128_1/results_2023-03-28-14-05.json"))

results = [oridata ,baselinedata, sagedata, mlpdata, gatdata]


for result in results:
    scores = [round(result[ligand]['F1'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    cis = [round(result[ligand]['F1_CI'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    mccs =  [round(result[ligand]['MCC'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    mcis = [round(result[ligand]['MCC_CI'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    for k, score in enumerate(scores):
        print(f"& {int(score*100)} \pm {int(cis[k]*100)} \%", end="")
    for k, score in enumerate(mccs):
        print(f"& {int(score*100)} \pm {int(mcis[k]*100)} \%", end="")
    print("\\")

for result in results:
    scores = [round(result[ligand]['cov_percentage'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    
    for k, score in enumerate(scores):
        print(f"& {int(score*100)} \%", end="")

    print("\\")


# section to create the tables for baselines in overleaf
sagedata = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/BindNode23SAGEConv_128_1/results_2023-03-28-13-38.json"))
gatdata = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/BindNode23SAGEConvGATMLP_320_4/results_2023-03-28-13-01.json"))
mlpdata = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/BindNode23SAGEConvMLP_320_1/results_2023-03-28-13-26.json"))
baselinedata = json.load(open("/home/frank/bind/bindPredict/predictions/GCNCONV_DIST1_1024_noBN.json"))
oridata = json.load(open("/home/frank/bind/bindPredict/bindNode23/final_eval/bindEmbed21DL_128_1/results_2023-03-28-14-05.json"))

results = [oridata ,baselinedata, sagedata, mlpdata, gatdata]


for result in results:
    scores = [round(result[ligand]['F1'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    cis = [round(result[ligand]['F1_CI'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    mccs =  [round(result[ligand]['MCC'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    mcis = [round(result[ligand]['MCC_CI'], 2) for ligand in ['metal', 'nucleic', 'small', 'overall']]
    for k, score in enumerate(scores):
        print(f"& {int(score*100)} \pm {int(cis[k]*100)} \%", end="")
    for k, score in enumerate(mccs):
        print(f"& {int(score*100)} \pm {int(mcis[k]*100)} \%", end="")
    print("\\")

for result in results:
    print(" & ".join([str(round(result[ligand]['MCC'], 3)) for ligand in ['metal', 'nucleic', 'small', 'overall']]))
    print(" & ".join([str(round(result[ligand]['MCC_CI'], 3)) for ligand in ['metal', 'nucleic', 'small', 'overall']]))

# this was to create hyperparam tablbe
for result in results:
    try: 
        architecture = result['architecture']
    except KeyError:
        architecture = "N/A"
    try: 
        cutoff = result['cutoff']
    except KeyError:
        cutoff = "N/A"
    print(f"{architecture} & {result['lr']} & {result['features']} & {result['dropout']} & {result['dropout_fcn']} & {cutoff} & {result['heads']}")
gatdata.keys()

def load_result(path):
    """Loads a single protein prediction file from disk"""
    with open(path, "r") as csvin:
        data = csvin.readlines()
    metal = []
    nuclear = []
    small = []
    metal_probs = []
    nuclear_probs = []
    small_probs = []
    for line in data[1:]:
        (
            pos,
            metal_prob,
            metal_class,
            nuclear_prob,
            nuclear_class,
            small_prob,
            small_class,
            any_class,
        ) = line.rstrip("\n").split("\t")
        metal.append(metal_class)
        nuclear.append(nuclear_class)
        small.append(small_class)
        metal_probs.append(metal_prob)
        nuclear_probs.append(nuclear_prob)
        small_probs.append(small_prob)

    metal_arr = ((np.array(metal)) == "b").astype(int)
    nuclear_arr = ((np.array(nuclear)) == "b").astype(int)
    small_arr = ((np.array(small)) == "b").astype(int)
    metalprobarr = np.array(metal_probs)
    nuclearprobarr = np.array(nuclear_probs)
    smallprobarr = np.array(small_probs)

    predictions = np.zeros((metal_arr.shape[0], 3))
    predictions[:, 0] = metal_arr
    predictions[:, 1] = nuclear_arr
    predictions[:, 2] = small_arr
    probs = np.zeros((metal_arr.shape[0], 3))
    probs[:, 0] = metalprobarr
    probs[:, 1] = nuclearprobarr
    probs[:, 2] = smallprobarr

    prot = ProteinResults(path.name.split(".")[0], bind_cutoff=0.5)
    binding_metal = True if np.sum(metal_arr) > 0 else False
    binding_nuclear = True if np.sum(nuclear_arr) > 0 else False
    binding_small = True if np.sum(small_arr) > 0 else False
    prot.predictions = predictions
    prot.probs = probs
    return prot, binding_metal, binding_nuclear, binding_small


# this section grabs all the predictions for all models
def assemble_model_results(final_eval_dir):
    """Loads all the protein predictions from all models in a directory."""
    models = [folder for folder in os.listdir(final_eval_dir) if os.path.isdir(os.path.join(final_eval_dir, folder))]
    resultdict = {"all_prots": {}}
    for model in models:
        modelresults = {
            "metal_prots": [],
            "nucleic_prots": [],
            "small_prots": [],
            "overall_prots": [],
            "all_prots": {},
        }
        globres = Path(
            f"/home/frank/bind/bindPredict/bindNode23/final_eval/{model}/predictions_devset"
        ).rglob("*.bindPredict_out")

        metal = 0
        nuclear = 0
        small = 0
        overall = 0

        for prot in globres:
            protn, binding_metal, binding_nuclear, binding_small = load_result(prot)
            if binding_metal:
                metal += 1
                modelresults["metal_prots"].append(protn.name)
            if binding_nuclear:
                nuclear += 1
                modelresults["nucleic_prots"].append(protn.name)
            if binding_small:
                small += 1
                modelresults["small_prots"].append(protn.name)
            if binding_metal or binding_nuclear or binding_small:
                overall += 1
                modelresults["overall_prots"].append(protn.name)

            modelresults["metal_sum"] = metal
            modelresults["nucleic_sum"] = nuclear
            modelresults["small_sum"] = small

            if not prot.name in modelresults["all_prots"].keys():
                modelresults["all_prots"][protn.name] = protn
            else:
                continue

            resultdict[model] = modelresults

    return models, resultdict


def plot_versus_baseline(
    fig, ax, labels, x_pos, model, k, ligand, old_metrics, old_ci, new_metrics, new_ci
):

    width = 0.2
    ax[k].bar(
        x_pos - width / 2,
        old_metrics,
        width=width,
        ecolor="black",
        capsize=10,
        yerr=old_ci,
    )
    ax[k].bar(
        x_pos + width / 2,
        new_metrics,
        width=width,
        ecolor="black",
        capsize=10,
        yerr=new_ci,
    )
    ax[k].set_ylabel("Value")
    ax[k].set_xticks(x_pos)
    ax[k].set_xticklabels(labels)
    ax[k].set_title(ligand)
    ax[k].yaxis.grid(True)


def calculate_metrics_on_overlap_sets(models, resultdict, outputdir=None):
    """Constructs the overlapping set and calculates the metrics"""
    for model in models[1:]:
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots(4)
        fig.set_size_inches(13, 15)
        plotmodels = ["bindEmbed21DL", model]
        plotlabels = ["MCC", "F1"]
        x_pos = np.arange(len(plotlabels))
        allidgen = Path(
            f"/home/frank/bind/bindPredict/bindNode23/final_eval/bindEmbed21DL_128_1/predictions_devset"
        ).rglob("*.bindPredict_out")
        ids = [path.name.split(".")[0] for path in allidgen]
        proteins, metals, nucleics, smalls = ProteinInformation.get_labels_for_overlap(
            ids=ids, sequences=sequences
        )
        complete_truth = {
            "metal": [id for id in ids if id in metals.keys()],
            "nucleic": [id for id in ids if id in nucleics.keys()],
            "small": [id for id in ids if id in smalls.keys()],
            "overall": [id for id in ids if id in proteins.keys()],
        }

        for k, ligand in enumerate(["metal", "nucleic", "small", "overall"]):
            print(f"Comparing {model} vs. bindEmbed21DL_128_1 for ligand {ligand}.")
            identifier = f"{ligand}_prots"
            print(
                f"{model} preds: {len(resultdict[model][identifier])} vs. bE21DL preds: {len(resultdict['bindEmbed21DL_128_1'][identifier])}."
            )

            baseline_set = {}
            newmethod_set = {}

            for protid in resultdict[model][f"{ligand}_prots"]:
                if protid in resultdict["bindEmbed21DL_128_1"][f"{ligand}_prots"]:
                    newmethod_set[protid] = resultdict[model]["all_prots"][protid]
                    baseline_set[protid] = resultdict["bindEmbed21DL_128_1"][
                        "all_prots"
                    ][protid]
            for protid in complete_truth[ligand]:
                if protid not in newmethod_set.keys():
                    
                    newmethod_set[protid] = resultdict[model]["all_prots"][protid]
                    baseline_set[protid] = resultdict["bindEmbed21DL_128_1"][
                        "all_prots"
                    ][protid]

            print("bindEmbed21DL")
            # assess performance for bE21DL
            labels = ProteinInformation.get_labels(baseline_set.keys(), sequences)
            model_performances = (
                PerformanceAssessment.combine_protein_performance_for_ligand(
                    baseline_set, cutoff, labels, ligands={ligand: []}
                )
            )
            (
                mcc_old,
                mcc_ci_old,
                f1_old,
                f1_ci_old,
            ) = PerformanceAssessment.print_performance_results_for_ligand(
                model_performances, ligands={ligand: []}, outputdir=outputdir, model="bE21"
            )

            print(model)
            # assess performance for my model
            labels = ProteinInformation.get_labels(newmethod_set.keys(), sequences)
            model_performances = (
                PerformanceAssessment.combine_protein_performance_for_ligand(
                    newmethod_set, cutoff, labels, ligands={ligand: []}
                )
            )
            (
                mcc_new,
                mcc_ci_new,
                f1_new,
                f1_ci_new,
            ) = PerformanceAssessment.print_performance_results_for_ligand(
                model_performances, ligands={ligand: []}, outputdir=outputdir, model=model
            )
            plot_versus_baseline(
                fig=fig,
                ax=ax,
                labels=plotlabels,
                x_pos=x_pos,
                model=model,
                k=k,
                ligand=ligand,
                old_metrics=[mcc_old, f1_old],
                old_ci=[mcc_ci_old, f1_ci_old],
                new_metrics=[mcc_new, f1_new],
                new_ci=[mcc_ci_new, f1_ci_new],
            )
            print("********************************************************")
            print("********************************************************")
            print("********************************************************")

        plt.legend(plotmodels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.tight_layout()
        plt.savefig(
            f"/home/frank/bind/bindPredict/bindNode23/plots/F1_MCC_versus_{ligand}_{model}.png",
            dpi=100,
        )

# first bit to retrieve ground truth for all proteins in devset
def get_ground_truth_for_all_predicted_proteins():
    sequences = FileManager.read_fasta(FileSetter.fasta_file())
    allidgen = Path(
        f"/home/frank/bind/bindPredict/bindNode23/final_eval/bindEmbed21DL_128_1/predictions_devset"
    ).rglob("*.bindPredict_out")
    ids = [path.name.split(".")[0] for path in allidgen]
    proteins, metals, nucleics, smalls = ProteinInformation.get_labels_for_overlap(
                ids=ids, sequences=sequences
            )
    return proteins, metals, nucleics, smalls


def get_binding_residues_with_minmax_neighbors():
    """get the min/ max neighborhood proteins and match them against binding residues from annotation"""
    minmaxdict = json.load(open("/home/frank/bind/bindPredict/json_output_minmax.json")) 
    proteins, metals, nucleics, smalls = get_ground_truth_for_all_predicted_proteins()
    further_inspect={"min":{"metal":[], "nucleic":[], "small":[]}, "max":{"metal":[], "nucleic":[], "small":[]}}

    for el in ["min", "max"]:

        for tupel in minmaxdict[el]:
            if tupel[0] in metals and tupel[0] in proteins.keys():
                #print(f"{tupel[0]} is in metal dict!")
                if tupel[1] in metals[tupel[0]]:
                    print(f"This tuple is a metal binding residue: {tupel[0]} with residue {tupel[1]}, given its {tupel[2]} neighbors")
                    further_inspect[el]["metal"].append((tupel))
            if tupel[0] in smalls and tupel[0] in proteins.keys():
                #print(f"{tupel[0]} is in metal dict!")
                if tupel[1] in smalls[tupel[0]]:
                    print(f"This tuple is a small binding residue: {tupel[0]} with residue {tupel[1]}, given its {tupel[2]} neighbors")
                    further_inspect[el]["small"].append((tupel))
            if tupel[0] in nucleics and tupel[0] in proteins.keys():
                #print(f"{tupel[0]} is in metal dict!")
                if tupel[1] in nucleics[tupel[0]]:
                    print(f"This tuple is a nucleics binding residue: {tupel[0]} with residue {tupel[1]}, given its {tupel[2]} neighbors")
                    further_inspect[el]["nucleic"].append((tupel))
    return further_inspect


def calculate_metrics_on_minmax_sets(models, resultdict, minmaxset):
    """Compares the predictions on proteins with min/max number of neighbors in graphs and calculates the metrics"""
    for model in models[1:]:
        plt.cla()
        plt.clf()
        fig, ax = plt.subplots(4)
        fig.set_size_inches(13, 15)
        plotmodels = ["bindEmbed21DL", model]
        plotlabels = ["MCC", "F1"]
        x_pos = np.arange(len(plotlabels))
        allidgen = Path(
            f"/home/frank/bind/bindPredict/bindNode23/final_eval/bindEmbed21DL_128_1/predictions_devset"
        ).rglob("*.bindPredict_out")
        ids = [path.name.split(".")[0] for path in allidgen]
        proteins, metals, nucleics, smalls = ProteinInformation.get_labels_for_overlap(
            ids=ids, sequences=sequences
        )

        for el in ["min", "max"]:

            for k, ligand in enumerate(["metal", "nucleic", "small"]):
                print(f"Comparing {model} vs. bindEmbed21DL_128_1 for ligand {ligand}.")
                identifier = f"{ligand}_prots"
                print(
                    f"{model} preds: {len(resultdict[model][identifier])} vs. bE21DL preds: {len(resultdict['bindEmbed21DL_128_1'][identifier])}."
                )

                baseline_set = {}
                newmethod_set = {}
                headers_to_inspect = ["P03882","P10970"]
                for protid in headers_to_inspect:
                    newmethod_set[protid] = resultdict[model]["all_prots"][protid]
                    baseline_set[protid] = resultdict["bindEmbed21DL_128_1"][
                        "all_prots"
                    ][protid]
                

                print("bindEmbed21DL")

                # assess performance for bE21DL
                labels = ProteinInformation.get_labels(baseline_set.keys(), sequences)
                model_performances = (
                    PerformanceAssessment.combine_protein_performance_for_ligand(
                        baseline_set, cutoff, labels, ligands={ligand: []}
                    )
                )
                (
                    mcc_old,
                    mcc_ci_old,
                    f1_old,
                    f1_ci_old,
                ) = PerformanceAssessment.print_performance_results_for_ligand(
                    model_performances, ligands={ligand: []}
                )

                print(model)
                # assess performance for bE21DL
                labels = ProteinInformation.get_labels(newmethod_set.keys(), sequences)
                model_performances = (
                    PerformanceAssessment.combine_protein_performance_for_ligand(
                        newmethod_set, cutoff, labels, ligands={ligand: []}
                    )
                )
                (
                    mcc_new,
                    mcc_ci_new,
                    f1_new,
                    f1_ci_new,
                ) = PerformanceAssessment.print_performance_results_for_ligand(
                    model_performances, ligands={ligand: []}
                )
                plot_versus_baseline(
                    fig=fig,
                    ax=ax,
                    labels=plotlabels,
                    x_pos=x_pos,
                    model=model,
                    k=k,
                    ligand=ligand,
                    old_metrics=[mcc_old, f1_old],
                    old_ci=[mcc_ci_old, f1_ci_old],
                    new_metrics=[mcc_new, f1_new],
                    new_ci=[mcc_ci_new, f1_ci_new],
                )
                print("********************************************************")
                print("********************************************************")
                print("********************************************************")

        plt.legend(plotmodels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.tight_layout()
        plt.savefig(
            f"/home/frank/bind/bindPredict/bindNode23/plots/MINMAXSET_F1_MCC_versus_{ligand}_{model}.png",
            dpi=100,
        )

# this section serves to calculate the metrics on overlapping set of predictions
plt.cla()
plt.clf()
cutoff = 0.5
ri = False  # Should RI or raw probabilities be written?
sequences = FileManager.read_fasta(FileSetter.fasta_file())
eval_dir = "/home/frank/bind/bindPredict/bindNode23/final_eval"
models, results = assemble_model_results(
    eval_dir
)
models
models = ["", 'BindNode23SAGEConvGATMLP_320_4', 'BindNode23SAGEConv_128_1']
calculate_metrics_on_overlap_sets(models, resultdict=results, outputdir=os.path.join(eval_dir, models[1]))




# in this section, we plot the results of min/max neighbors protein sets
plt.cla()
plt.clf()
cutoff = 0.5
ri = False  # Should RI or raw probabilities be written?
models, results = assemble_model_results(
    "/home/frank/bind/bindPredict/bindNode23/final_eval"
)
models = ['BindNode23SAGEConv_128_1', 'BindNode23SAGEConvMLP_320_1', 'BindNode23SAGEConvGATMLP_320_4']
calculate_metrics_on_minmax_sets(models, resultdict=results, minmaxset=get_binding_residues_with_minmax_neighbors())
# summary: nothing really sticks out, not min, not max, no ligand class

# case study 
from pprint import pprint
pprint(get_binding_residues_with_minmax_neighbors())

baseline_set = {}
newmethod_set = {}
models, resultdict = assemble_model_results(
    "/home/frank/bind/bindPredict/bindNode23/final_eval"
)
sequences = FileManager.read_fasta(FileSetter.fasta_file())
# min
#   small: 'P12306'
#   nucleic: 'P03882'
#   metal: None
# max
#   small: ''Q8TL28'
#   nucleic: None
#   metal: 'Q8TL28'
headers_to_inspect = ["P03882","P10970", "P0A6X7", 'Q5SID6', "Q8TL28", 'P12306', "P13271", "Q84AQ1"]
for protid in headers_to_inspect:
    newmethod_set[protid] = resultdict["BindNode23SAGEConvGATMLP_320_4"]["all_prots"][protid]
    newmethod_set[protid] = resultdict["BindNode23SAGEConvMLP_320_1"]["all_prots"][protid]
    baseline_set[protid] = resultdict["bindEmbed21DL_128_1"][
        "all_prots"
    ][protid]

protein_to_inspect = "P03882"
# this line tells us which residues are predicted binding for nucleic so we can overlay it on 3D
(newmethod_set[protein_to_inspect].probs[:,1]>0.5).nonzero()
(baseline_set[protein_to_inspect].probs[:,1]>0.5).nonzero() # bindEmbed21 does not catch it!!
# this line tells us GT for the same
labels = ProteinInformation.get_labels([protein_to_inspect], sequences)
labels[protein_to_inspect][:,1].nonzero()

protein_to_inspect = "P0A6X7"
# this line tells us which residues are predicted binding for nucleic so we can overlay it on 3D
labels = ProteinInformation.get_labels([protein_to_inspect], sequences)
(newmethod_set[protein_to_inspect].probs[:,1]>0.5).nonzero()[0]
(baseline_set[protein_to_inspect].probs[:,1]>0.5).nonzero() # also bindEmbed21 misses the far end of the branch
# this line tells us GT for the same
labels[protein_to_inspect][:,1].nonzero()

protein_to_inspect = "Q8TL28"
# this line tells us which residues are predicted binding for nucleic so we can overlay it on 3D
labels = ProteinInformation.get_labels([protein_to_inspect], sequences)
(newmethod_set[protein_to_inspect].probs[:,0]>0.5).nonzero()
(baseline_set[protein_to_inspect].probs[:,0]>0.5).nonzero() #
# this line tells us GT for the same
labels[protein_to_inspect][:,0].nonzero()

protein_to_inspect = "P12306"
# this line tells us which residues are predicted binding for nucleic so we can overlay it on 3D
labels = ProteinInformation.get_labels([protein_to_inspect], sequences)
(newmethod_set[protein_to_inspect].probs[:,0]>0.5).nonzero()
(baseline_set[protein_to_inspect].probs[:,0]>0.5).nonzero() #
# this line tells us GT for the same
labels[protein_to_inspect][:,0].nonzero()

def protein_annotation(protein_to_inspect = "P12306", type="small"):
    if type=="metal":
        index = 0
    if type=="nucleic":
        index = 1
    if type=="small":
        index = 2
    # this line tells us which residues are predicted binding for nucleic so we can overlay it on 3D
    new= (newmethod_set[protein_to_inspect].probs[:,index]>0.5).nonzero()
    old= (baseline_set[protein_to_inspect].probs[:,index]>0.5).nonzero() # bindEmbed21 does not catch it!!
    # this line tells us GT for the same
    labels = ProteinInformation.get_labels([protein_to_inspect], sequences)
    true=labels[protein_to_inspect][:,index].nonzero()
    return new, old, true

def turn_into_3d_notation(array):
    string = "+".join(map(str, array[0].tolist()))
    return string

def get_strings(protein_to_inspect, type):
    new, old, true = protein_annotation(protein_to_inspect, type)
    print("New:")
    print(turn_into_3d_notation(new))
    print("Old:")
    print(turn_into_3d_notation(old))
    print("True:")
    print(turn_into_3d_notation(true))

get_strings(protein_to_inspect="Q5SID6", type="metal")

# min
#   small: 'P12306'
#   nucleic: 'P03882'
#   metal: None
# max
#   small: ''Q8TL28'
#   nucleic: None
#   metal: 'Q8TL28'

# another section to find out, for min/max neighbors, which proteins are predicted way better or way worse
minmaxresidues = get_binding_residues_with_minmax_neighbors()
headers_to_analyze = []
for key, value in minmaxresidues.items():
    for nestkey, nestvalue in value.items():
        headers_to_analyze.extend([tupleval[0] for tupleval in nestvalue])
headers_to_analyze
cutoff = 0.5
models, results = assemble_model_results(
    "/home/frank/bind/bindPredict/bindNode23/final_eval"
)
models = ['BindNode23SAGEConvGATMLP_320_4']

labels, metals, nucleics, smalls = get_ground_truth_for_all_predicted_proteins()
maxdiff = {key: 0 for key in ['metal', 'small', 'nucleic']}
for header in headers_to_analyze:
    prot_ours = results['BindNode23SAGEConvGATMLP_320_4']["all_prots"][header]
    prot_ours.set_labels(labels[header])
    prot_theirs = results['bindEmbed21DL_128_1']["all_prots"][header]
    prot_theirs.set_labels(labels[header])
    # calculate performance for protein
    performance_ours = prot_ours.calc_performance_measurements(cutoff)
    performance_theirs = prot_theirs.calc_performance_measurements(cutoff)
    for k in maxdiff.keys():
        perf_diff = performance_ours[k]['mcc']-performance_theirs[k]['mcc']
        if perf_diff > maxdiff[k]:
            print(f"New max diff in single performance: {perf_diff} for ligand {k}")
            maxdiff[k]=perf_diff

# replay this section, but the max diffs are not great. It should rather be a single residue, where the confidence is a lot higher
# get all the binding residues, get the prediction probs from both, and for the ones with the highest confident correct prediction, compare


# this section analyzes probabilites to find residues where confidence increased the most (for true pred)
cutoff = 0.5
models, results = assemble_model_results(
    "/home/frank/bind/bindPredict/bindNode23/final_eval"
)
models
models = ['BindNode23SAGEConvGATMLP_320_4']

labels, metals, nucleics, smalls = get_ground_truth_for_all_predicted_proteins()
truths = {'metal':metals, 'small':smalls, 'nucleic':nucleics}
indices = {'metal':0, 'small':2, 'nucleic':1}
len(labels)
metals.keys()
maxdiff = {key: 0 for key in ['metal', 'small', 'nucleic']}
for header in labels.keys():
    prot_ours = results['BindNode23SAGEConvGATMLP_320_4']["all_prots"][header]
    prot_ours.set_labels(labels[header])
    prot_theirs = results['bindEmbed21DL_128_1']["all_prots"][header]
    prot_theirs.set_labels(labels[header])
    # calculate performance for protein
    for key, val in truths.items():
        if header in val.keys():
            for residue in val[header]:
                prob_diff=prot_ours.probs[residue-1, indices[key]]-prot_theirs.probs[residue-1, indices[key]]
                if prob_diff>maxdiff[key]:
                    print(f"Found a new max prob diff for {key}-prediction in {header}: {prob_diff} in residue {residue}")
                    maxdiff[key]=prob_diff
print(maxdiff)
# for metal class P04608, res 30 (1-based!), 0.779
# for small class, Q1HRLZ, res 100, 0,729
# for nucleic class, Q9X7H4: 0.859 in residue 20