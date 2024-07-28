import pickle
import numpy as np
import pandas as pd
import thisnotthat as tnt
import panel as pn

pn.extension()
pn.extension('tabulator')

with open('bib_specter_embedded.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('word_umap.pkl', 'rb') as f:
    word_map = pickle.load(f)

with open('word_pacmap2.pkl', 'rb') as f:
    word_map2 = pickle.load(f)

with open('word_pacmap3.pkl', 'rb') as f:
    word_map3 = pickle.load(f)

df = pd.read_csv("Mijn Bibliotheek.csv")
df = df.dropna(subset=["Title", "Abstract Note"])


df.rename(columns={
    "Item Type": "ItemType",
    "Title": "Title",
    "Author": "Author",
    "Abstract Note": "AbstractNote",
    "Url": "Url",
    "Date": "Date",
    "Access Date": "AccessDate",
    "Manual Tags": "ManualTags"
}, inplace=True)


def assign_period(date):
    if pd.isnull(date):
        return 'No Access Date'
    elif date < pd.Timestamp(2022, 9, 1):
        return 'Before Sep 2022'
    elif date < pd.Timestamp(2023, 1, 1):
        return 'Sep-Dec 2022'
    elif date < pd.Timestamp(2023, 3, 1):
        return 'Jan-Feb 2023'
    elif date < pd.Timestamp(2023, 6, 1):
        return 'Mar-May 2023'
    elif date < pd.Timestamp(2023, 9, 1):
        return 'Jun-Aug 2023'
    elif date < pd.Timestamp(2024, 1, 1):
        return 'Sep-Dec 2023'
    else:
        return 'After Dec 2023'


df['AccessDate'] = pd.to_datetime(df["AccessDate"])
# Assign period based on the date
df['Period'] = df['AccessDate'].apply(assign_period)

COLOR_KEY = {
    'No Access Date': '#1f77b4',  # Dark blue
    # 'Before Sep 2022': '#2ca02c',  # Green
    'Sep-Dec 2022': '#fcbf49',  # Yellow
    'Jan-Feb 2023': '#fc8f44',  # Orange
    'Mar-May 2023': '#d62728',  # Red
    'Jun-Aug 2023': '#9467bd',  # Purple
    'Sep-Dec 2023': '#8c564b',  # Brown
    'After Dec 2023': '#e377c2',  # Pink
}


basic_plot1 = tnt.BokehPlotPane(
    word_map,
    labels=df['Period'],
    label_color_mapping=COLOR_KEY,
    hover_text=df["Title"],
    show_legend=True,
    legend_location="top_right",
    sizing_mode='stretch_both')

basic_plot2 = tnt.BokehPlotPane(
    word_map2,
    labels=df['Period'],
    label_color_mapping=COLOR_KEY,
    hover_text=df["Title"],
    show_legend=True,
    legend_location="top_right",
    sizing_mode='stretch_both')

basic_plot3 = tnt.BokehPlotPane(
    word_map3,
    labels=df['Period'],
    label_color_mapping=COLOR_KEY,
    hover_text=df["Title"],
    show_legend=True,
    legend_location="top_right",
    sizing_mode='stretch_both')

dfsmall = df[["ItemType", "Title", "Author", "AbstractNote",
              "Url", "Date", "AccessDate", "ManualTags"]]

data_view = tnt.SimpleDataPane(
    dfsmall,
    sizing_mode="stretch_both", max_rows=400, max_cols=50)

basic_plot2.link(
    basic_plot1,
    selected="selected",
    bidirectional=True
)

basic_plot3.link(
    basic_plot1,
    selected="selected",
    bidirectional=True
)

data_view.link(
    basic_plot1,
    selected="selected",
    bidirectional=True
)


column1 = pn.Column(basic_plot2, name="pacmap nn 5")
column2 = pn.Column(basic_plot1, name="umap")
column3 = pn.Column(basic_plot3, name="pacmap nn 7")
column4 = pn.Column(data_view, name="data")

search = tnt.SearchWidget(df)
search.link_to_plot(basic_plot1)

app = pn.Tabs(
    column1,
    column2,
    column3,
    column4,
    search)

simplesearch = tnt.SimpleSearchWidget(basic_plot1, raw_dataframe=df)
app = pn.Column(simplesearch, app)

# word_maps = [word_map, word_map2, word_map3]
# basic_plots = [basic_plot1, basic_plot2, basic_plot3]
# for w_map, plot in zip(word_maps, basic_plots):

#     label_layers = tnt.MetadataLabelLayers(
#         np.array(embeddings),
#         np.array(w_map),
#         df["ManualTags"].str.get_dummies(";"),
#         hdbscan_min_cluster_size=2,
#         hdbscan_min_samples=2,
#         contamination=1e-6,
#         min_clusters_in_layer=5,
#         vector_metric="cosine",
#         cluster_distance_threshold=0.0,
#         random_state=0,
#         items_per_label=2
#     )

#     plot.add_cluster_labels(
#         label_layers, text_size_scale=100, text_layer_scale_factor=3.0)

# label_layers = tnt.MetadataLabelLayers(
#     np.array(embeddings),
#     np.array(word_map2),
#     df["ManualTags"].str.get_dummies(";"),
#     hdbscan_min_cluster_size=2,
#     hdbscan_min_samples=2,
#     contamination=1e-6,
#     min_clusters_in_layer=5,
#     vector_metric="cosine",
#     cluster_distance_threshold=0.0,
#     random_state=0,
#     items_per_label=2
# )

# basic_plot2.add_cluster_labels(
#     label_layers, text_size_scale=100, text_layer_scale_factor=2.0)

pn.serve(app)
# app.servable()
