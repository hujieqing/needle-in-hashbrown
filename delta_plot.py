import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

communitiesLinkResultData = {
	"Base Model": ["GCN", "GCN", "GCN", "GCN", "GraphSage", "GraphSage", "GraphSage", "GraphSage", "GAT", "GAT", "GAT", "GAT", "GIN", "GIN", "GIN", "GIN", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L"],
	"Position Encoding": ["None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined"],
	"AUC": [0.997, 0.986, 0.986, 0.986, 0.986, 0.979, 0.985, 0.986, 0.981, 0.919, 0.983, 0.987, 0.980, 0.988, 0.987, 0.987, 0.978, 0.978, 0.982, 1.0, 0.965, 0.973, 0.985, 0.982],
	"Kendall's Tau": [0.182, 0.215, 0.267, 0.301, 0.218, 0.268, 0.248, 0.253, 0.203, 0.263, 0.300, 0.303, 0.241, 0.272, 0.290, 0.432, 0.334, 0.340, 0.338, 1.0, 0.554, 0.600, 0.566, 0.588]
}

emailLinkResultData = {
	"Base Model": ["GCN", "GCN", "GCN", "GCN", "GraphSage", "GraphSage", "GraphSage", "GraphSage", "GAT", "GAT", "GAT", "GAT", "GIN", "GIN", "GIN", "GIN", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L"],
	"Position Encoding": ["None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined"],
	"AUC": [0.709, 0.767, 0.721, 0.782, 0.571, 0.735, 0.557, 0.769, 0.538, 0.758, 0.550, 0.769, 0.724, 0.785, 0.791, 0.808, 0.751, 0.769, 0.835, 1.0, 0.792, 0.770, 1.0, 0.775],
	"Kendall's Tau": [0.239, 0.364, 0.217, 0.364, 0.147, 0.361, 0.156, 0.363, 0.086, 0.336, 0.096, 0.353, 0.402, 0.443, 0.429, 0.452, 0.503, 0.529, 0.516, 1.0, 0.547, 0.549, 1.0, 0.550]
}

communitiesLinkPairResultData = {
	"Base Model": ["GCN", "GCN", "GCN", "GCN", "GraphSage", "GraphSage", "GraphSage", "GraphSage", "GAT", "GAT", "GAT", "GAT", "GIN", "GIN", "GIN", "GIN", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L"],
	"Position Encoding": ["None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined"],
	"AUC": [0.988, 0.992, 0.992, 0.992, 0.993, 0.982, 0.994, 0.993, 0.989, 0.975, 0.989, 0.993, 0.991, 0.982, 0.992, 0.984, 0.986, 0.988, 0.991, 1.0, 0.981, 0.981, 0.990, 0.994],
	"Kendall's Tau": [0.146, 0.217, 0.277, 0.335, 0.212, 0.257, 0.270, 0.280, 0.205, 0.334, 0.318, 0.324, 0.213, 0.250, 0.301, 0.365, 0.346, 0.358, 0.357, 1.0, 0.518, 0.576, 0.535, 0.607]
}

emailLinkPairResultData = {
	"Base Model": ["GCN", "GCN", "GCN", "GCN", "GraphSage", "GraphSage", "GraphSage", "GraphSage", "GAT", "GAT", "GAT", "GAT", "GIN", "GIN", "GIN", "GIN", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L"],
	"Position Encoding": ["None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined"],
	"AUC": [0.518, 0.681, 0.575, 0.708, 0.538, 0.693, 0.533, 0.744, 0.507, 0.725, 0.528, 0.747, 0.723, 0.741, 0.726, 0.774, 0.751, 0.734, 0.772, 1.0, 0.735, 0.784, 1.0, 0.753],
	"Kendall's Tau": [0.238, 0.432, 0.267, 0.437, 0.139, 0.428, 0.206, 0.433, 0.083, 0.407, 0.112, 0.435, 0.479, 0.525, 0.482, 0.521, 0.576, 0.615, 0.609, 1.0, 0.603, 0.638, 1.0, 0.614]
}

resultDataAll = {
	"Dataset": 24 * ["Communities"] + 24 * ["Email"] + 24 * ["Communities"] + 24 * ["Email"],
	"Task": 24 * 2 * ["Link Prediction"] + 24 * 2 * ["Pairwise Node Classification"],
	"Base Model": 4 * ["GCN", "GCN", "GCN", "GCN", "GraphSage", "GraphSage", "GraphSage", "GraphSage", "GAT", "GAT", "GAT", "GAT", "GIN", "GIN", "GIN", "GIN", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-F-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L", "P-GNN-E-2L"],
	"Position Encoding": 4 * ["None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined", "None", "Hash", "MSE", "Combined"],
	"AUC": communitiesLinkResultData["AUC"] + emailLinkResultData["AUC"] + communitiesLinkPairResultData["AUC"] + emailLinkPairResultData["AUC"],
	"Kendall's Tau": communitiesLinkResultData["Kendall's Tau"] + emailLinkResultData["Kendall's Tau"] + communitiesLinkPairResultData["Kendall's Tau"] + emailLinkPairResultData["Kendall's Tau"]
}

# print(len(emailLinkPairResultData["Base Model"]), len(emailLinkPairResultData["Position Encoding"]), len(emailLinkPairResultData["AUC"]), len(emailLinkPairResultData["Kendall's Tau"]))

# communitiesLinkDF = pd.DataFrame(communitiesLinkResultData)

# print(communitiesLinkDF)
sns.set(style="ticks")

# sns_plot = sns.scatterplot(x="Kendall's Tau", y="AUC", hue="Position Encoding", style="Base Model", data=communitiesLinkDF, legend="brief")
# figure = sns_plot.get_figure()    
# figure.savefig('figs/KTvsAUC.png')

DF = pd.DataFrame(resultDataAll)
dfEmail = DF.loc[(DF["Task"] == "Link Prediction") & ((DF["Position Encoding"] == "None") | (DF["Position Encoding"] == "Combined"))]
# g = sns.FacetGrid(DF, row="Dataset", col="Task", hue="Base Model", style="Position Encoding", margin_titles=True, height=2.5)
# g.map(plt.scatter, "Kendall's Tau", "AUC")
# g.set_axis_labels("Kendall's Tau", "AUC")

# # sns_plot = sns.relplot(x="Kendall's Tau", y="AUC", hue="Base Model", style="Position Encoding", col="Task", row="Dataset", data=DF, s=150)
# # fig, ax = plt.subplots()
# g = sns.relplot(x="Kendall's Tau", y="AUC", hue="Base Model", style="Position Encoding", col="Dataset", col_wrap=2, data=dfEmail, s=150)
# # g.map(plt.plot, "Kendall's Tau", "AUC", data=dfEmail.loc[DF["Base Model"].isin(["GCN"])])
# # sns.relplot(x="Kendall's Tau", y="AUC", col="Dataset", col_wrap=2, data=dfEmail.loc[DF["Base Model"].isin(["GCN"])], kind="line", ax=ax)
# # figure = sns_plot.get_figure()    
# # figure.savefig('figs/KTvsAUC.png')
# g.savefig('figs/KTvsAUCLP.png')

fig, ax = plt.subplots(1, 2)
g1 = sns.scatterplot(x="Kendall's Tau", y="AUC", hue="Base Model", style="Position Encoding", data=dfEmail.loc[DF["Dataset"] == "Communities"], s=70, ax=ax[0], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "GCN")], ax=ax[0], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "GraphSage")], ax=ax[0], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "GAT")], ax=ax[0], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "GIN")], ax=ax[0], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "P-GNN-F-2L")], ax=ax[0], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "P-GNN-E-2L")], ax=ax[0], legend=False)

g2 = sns.scatterplot(x="Kendall's Tau", y="AUC", hue="Base Model", style="Position Encoding", data=dfEmail.loc[DF["Dataset"] == "Email"], s=70, ax=ax[1])
# h,l = ax[1].get_legend_handles_labels()
lgd = ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "GCN")], ax=ax[1], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "GraphSage")], ax=ax[1], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "GAT")], ax=ax[1], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "GIN")], ax=ax[1], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "P-GNN-F-2L")], ax=ax[1], legend=False)
sns.lineplot(x="Kendall's Tau", y="AUC", data=dfEmail.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "P-GNN-E-2L")], ax=ax[1], legend=False)

ax[1].set_ylabel('')
ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[0].set_title('Communities')
ax[1].set_title('Email')

# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Kendall's Tau")

fig.savefig('figs/KTvsAUCLP.png', bbox_extra_artists=(lgd,), bbox_inches='tight')