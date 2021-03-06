import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

communitiesLinkResultData = {
	"Base Model": ["GCN"] * 4 + ["SAGE"] * 4 + ["GAT"] * 4 + ["GIN"] * 4 + ["P-GNN-F-2L"] * 4 + ["P-GNN-E-2L"] * 4,
	"Variant": ["None", "Hash", "MSE", "Both"] * 6,
	"AUC": [0.977, 0.986, 0.986, 0.986, 0.986, 0.979, 0.985, 0.9861, 0.981, 0.919, 0.983, 0.987, 0.980, 0.988, 0.987, 0.987, 0.978, 0.978, 0.982, 0.980, 0.965, 0.973, 0.985, 0.982],
	"Kendall's Tau": [0.182, 0.215, 0.267, 0.301, 0.218, 0.268, 0.248, 0.253, 0.203, 0.263, 0.300, 0.303, 0.241, 0.272, 0.290, 0.432, 0.334, 0.340, 0.338, 0.341, 0.554, 0.600, 0.566, 0.588]
}

emailLinkResultData = {
	"Base Model": ["GCN"] * 4 + ["SAGE"] * 4 + ["GAT"] * 4 + ["GIN"] * 4 + ["P-GNN-F-2L"] * 4 + ["P-GNN-E-2L"] * 4,
	"Variant": ["None", "Hash", "MSE", "Both"] * 6,
	"AUC": [0.709, 0.767, 0.721, 0.782, 0.571, 0.735, 0.557, 0.769, 0.538, 0.758, 0.550, 0.769, 0.724, 0.785, 0.791, 0.808, 0.751, 0.769, 0.835, 0.823, 0.792, 0.770, 0.765, 0.775],
	"Kendall's Tau": [0.239, 0.364, 0.217, 0.364, 0.147, 0.361, 0.156, 0.363, 0.086, 0.336, 0.096, 0.353, 0.402, 0.443, 0.429, 0.452, 0.503, 0.529, 0.516, 0.532, 0.547, 0.549, 0.539, 0.550]
}

ppiLinkResultData = {
	"Base Model": ["GCN"] * 4 + ["SAGE"] * 4 + ["GAT"] * 4 + ["GIN"] * 4 + ["P-GNN-F-2L"] * 4 + ["P-GNN-E-2L"] * 4,
	"Variant": ["None", "Hash", "MSE", "Both"] * 6,
	"AUC": [0.798, 0.810, 0.760, 0.821, 0.809, 0.818, 0.804, 0.812, 0.798, 0.818, 0.789, 0.819, 0.755, 0.788, 0.751, 0.789, 0.812, 0.823, 0.815, 0.819, 0.772, 0.775, 0.792, 0.803],
	"Kendall's Tau": [0.413, 0.441, 0.422, 0.444, 0.426, 0.464, 0.421, 0.461, 0.416, 0.462, 0.413, 0.465, 0.442, 0.450, 0.432, 0.460, 0.375, 0.388, 0.372, 0.391, 0.435, 0.426, 0.446, 0.454]
}

communitiesLinkPairResultData = {
	"Base Model": ["GCN"] * 4 + ["SAGE"] * 4 + ["GAT"] * 4 + ["GIN"] * 4 + ["P-GNN-F-2L"] * 4 + ["P-GNN-E-2L"] * 4,
	"Variant": ["None", "Hash", "MSE", "Both"] * 6,
	"AUC": [0.988, 0.992, 0.992, 0.992, 0.993, 0.982, 0.994, 0.9931, 0.989, 0.975, 0.989, 0.993, 0.991, 0.982, 0.992, 0.984, 0.986, 0.988, 0.991, 0.987, 0.981, 0.981, 0.990, 0.994],
	"Kendall's Tau": [0.146, 0.217, 0.277, 0.335, 0.212, 0.257, 0.270, 0.280, 0.205, 0.334, 0.318, 0.324, 0.213, 0.250, 0.301, 0.365, 0.346, 0.358, 0.357, 0.359, 0.518, 0.576, 0.535, 0.607]
}

emailLinkPairResultData = {
	"Base Model": ["GCN"] * 4 + ["SAGE"] * 4 + ["GAT"] * 4 + ["GIN"] * 4 + ["P-GNN-F-2L"] * 4 + ["P-GNN-E-2L"] * 4,
	"Variant": ["None", "Hash", "MSE", "Both"] * 6,
	"AUC": [0.518, 0.681, 0.575, 0.708, 0.538, 0.693, 0.533, 0.744, 0.507, 0.725, 0.528, 0.747, 0.723, 0.741, 0.726, 0.774, 0.751, 0.734, 0.772, 0.769, 0.735, 0.784, 0.772, 0.753],
	"Kendall's Tau": [0.238, 0.432, 0.267, 0.437, 0.139, 0.428, 0.206, 0.433, 0.083, 0.407, 0.112, 0.435, 0.479, 0.525, 0.482, 0.521, 0.576, 0.615, 0.609, 0.616, 0.603, 0.638, 0.612, 0.614]
}

resultDataAll = {
	"Dataset": 24 * ["Communities"] + 24 * ["Email"] + 24 * ["PPI"] + 24 * ["Communities"] + 24 * ["Email"],
	"Task": 24 * 3 * ["Link Prediction"] + 24 * 2 * ["Pairwise Node Classification"],
	"Base Model": 5 * communitiesLinkResultData["Base Model"],
	"Variant": 5 * communitiesLinkResultData["Variant"],
	"AUC": communitiesLinkResultData["AUC"] + emailLinkResultData["AUC"] + ppiLinkResultData["AUC"] + communitiesLinkPairResultData["AUC"] + emailLinkPairResultData["AUC"],
	"Kendall's Tau": communitiesLinkResultData["Kendall's Tau"] + emailLinkResultData["Kendall's Tau"] + ppiLinkResultData["Kendall's Tau"] + communitiesLinkPairResultData["Kendall's Tau"] + emailLinkPairResultData["Kendall's Tau"]
}

base_models = ["GCN", "SAGE", "GAT", "GIN", "P-GNN-F-2L", "P-GNN-E-2L"]

# print(len(emailLinkPairResultData["Base Model"]), len(emailLinkPairResultData["Variant"]), len(emailLinkPairResultData["AUC"]), len(emailLinkPairResultData["Kendall's Tau"]))

# communitiesLinkDF = pd.DataFrame(communitiesLinkResultData)

# print(communitiesLinkDF)
sns.set(style="ticks")

# sns_plot = sns.scatterplot(x="Kendall's Tau", y="AUC", hue="Variant", style="Base Model", data=communitiesLinkDF, legend="brief")
# figure = sns_plot.get_figure()    
# figure.savefig('figs/KTvsAUC.png')

DF = pd.DataFrame(resultDataAll)
dfLP = DF.loc[(DF["Task"] == "Link Prediction") & ((DF["Variant"] == "None") | (DF["Variant"] == "Both"))]
dfNC = DF.loc[(DF["Task"] == "Pairwise Node Classification") & ((DF["Variant"] == "None") | (DF["Variant"] == "Both"))]
# g = sns.FacetGrid(DF, row="Dataset", col="Task", hue="Base Model", style="Variant", margin_titles=True, height=2.5)
# g.map(plt.scatter, "Kendall's Tau", "AUC")
# g.set_axis_labels("Kendall's Tau", "AUC")

# # sns_plot = sns.relplot(x="Kendall's Tau", y="AUC", hue="Base Model", style="Variant", col="Task", row="Dataset", data=DF, s=150)
# # fig, ax = plt.subplots()
# g = sns.relplot(x="Kendall's Tau", y="AUC", hue="Base Model", style="Variant", col="Dataset", col_wrap=2, data=dfEmail, s=150)
# # g.map(plt.plot, "Kendall's Tau", "AUC", data=dfEmail.loc[DF["Base Model"].isin(["GCN"])])
# # sns.relplot(x="Kendall's Tau", y="AUC", col="Dataset", col_wrap=2, data=dfEmail.loc[DF["Base Model"].isin(["GCN"])], kind="line", ax=ax)
# # figure = sns_plot.get_figure()    
# # figure.savefig('figs/KTvsAUC.png')
# g.savefig('figs/KTvsAUCLP.png')

for l, df in {"LP": dfLP, "NC": dfNC}.items():
	if l == "NC":
		fig, ax = plt.subplots(1, 2)
		ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

		# plot for communities
		g1 = sns.scatterplot(y="Kendall's Tau", x="AUC", hue="Base Model", style="Variant", data=df.loc[DF["Dataset"] == "Communities"], s=70, ax=ax[0], legend=False)
		for base_model in base_models:
			sns.lineplot(y="Kendall's Tau", x="AUC", data=df.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == base_model)], ax=ax[0], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "SAGE")], ax=ax[0], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "GAT")], ax=ax[0], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "GIN")], ax=ax[0], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "P-GNN-F-2L")], ax=ax[0], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == "P-GNN-E-2L")], ax=ax[0], legend=False)

		
		# plot for email
		g2 = sns.scatterplot(y="Kendall's Tau", x="AUC", hue="Base Model", style="Variant", data=df.loc[DF["Dataset"] == "Email"], s=70, ax=ax[1])
		# h,l = ax[1].get_legend_handles_labels()
		lgd = ax[1].legend(bbox_to_anchor=(0.98, 0.5), loc=6, borderaxespad=0., frameon=False)
		for base_model in base_models:
			sns.lineplot(y="Kendall's Tau", x="AUC", data=df.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == base_model)], ax=ax[1], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "SAGE")], ax=ax[1], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "GAT")], ax=ax[1], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "GIN")], ax=ax[1], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "P-GNN-F-2L")], ax=ax[1], legend=False)
		# sns.lineplot(x="Kendall's Tau", y="AUC", data=dfLP.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == "P-GNN-E-2L")], ax=ax[1], legend=False)
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
		plt.xlabel("AUC")
		fig.savefig("figs/AUCvsKT" + l +".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight', format="pdf")

	elif l == "LP":
		fig, ax = plt.subplots(1, 3)
		ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		ax[2].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		# plot for communities
		g1 = sns.scatterplot(y="Kendall's Tau", x="AUC", hue="Base Model", style="Variant", data=df.loc[DF["Dataset"] == "Communities"], s=70, ax=ax[0], legend=False)
		for base_model in base_models:
			sns.lineplot(y="Kendall's Tau", x="AUC", data=df.loc[(DF["Dataset"] == "Communities") &  (DF["Base Model"] == base_model)], ax=ax[0], legend=False)
		# plot for email
		g2 = sns.scatterplot(y="Kendall's Tau", x="AUC", hue="Base Model", style="Variant", data=df.loc[DF["Dataset"] == "Email"], s=70, ax=ax[1], legend=False)
		# h,l = ax[1].get_legend_handles_labels()
		# lgd = ax[1].legend(bbox_to_anchor=(0.98, 0.5), loc=6, borderaxespad=0., frameon=False)
		for base_model in base_models:
			sns.lineplot(y="Kendall's Tau", x="AUC", data=df.loc[(DF["Dataset"] == "Email") &  (DF["Base Model"] == base_model)], ax=ax[1], legend=False)
		# plot for ppi
		g3 = sns.scatterplot(y="Kendall's Tau", x="AUC", hue="Base Model", style="Variant", data=df.loc[DF["Dataset"] == "PPI"], s=70, ax=ax[2], legend=False)
		# h,l = ax[1].get_legend_handles_labels()
		# lgd = ax[2].legend(bbox_to_anchor=(0.98, 0.5), loc=6, borderaxespad=0., frameon=False)
		for base_model in base_models:
			sns.lineplot(y="Kendall's Tau", x="AUC", data=df.loc[(DF["Dataset"] == "PPI") &  (DF["Base Model"] == base_model)], ax=ax[2], legend=False)
		ax[2].set_ylabel('')
		ax[2].set_xlabel('')
		ax[2].set_title('PPI')
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
		plt.xlabel("AUC")		

		# fig.tight_layout()
		plt.subplots_adjust(wspace = 0.4)

		fig.savefig("figs/AUCvsKT" + l +".pdf", bbox_inches='tight', format="pdf")