import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pacmap
import thisnotthat as tnt
import panel as pn

pn.extension()
pn.extension('tabulator')

print("Loading data...")
zot_df = pd.read_csv('zot_clean.csv')

# Convert the date columns to datetime objects
date_columns = ["Date", "Date Added", "Date Modified"]
for col in date_columns:
    zot_df[col] = pd.to_datetime(zot_df[col], errors='coerce')

zot_df["Publication Year"] = zot_df["Publication Year"].astype("Int64")
zot_df["Hearts"] = zot_df["Hearts"].astype("Int64")

zot_df["Manual Tags"] = zot_df["Manual Tags"].fillna("").str.split(
    ";").apply(lambda tags: [tag.strip() for tag in tags])
zot_df["Common Tags"] = zot_df["Common Tags"].fillna("").str.split(
    ";").apply(lambda tags: [tag.strip() for tag in tags])
zot_df["Common Tags"] = zot_df["Common Tags"].apply(
    lambda tags: [tag for tag in tags if tag != ""])

# dropping rows without title
zot_df = zot_df.dropna(subset=['Title'])

# reordering the columns
first_order = ["Title", "Author", "Date", "Abstract Note", "Date Added"]
new_order = first_order + \
    [col for col in zot_df.columns.tolist() if col not in first_order]
zot_df = zot_df.reindex(columns=new_order)

# this is for lower case search
zot_df_lowercase = zot_df.copy()
zot_df_lowercase["Title Lower"] = zot_df_lowercase["Title"].str.lower()
zot_df_lowercase["Author Lower"] = zot_df_lowercase["Author"].str.lower()
zot_df_lowercase["Abstract Note Lower"] = zot_df_lowercase["Abstract Note"].str.lower()

# loading computed embeddings
embeddings_df = pd.read_csv('zot_embeddings.csv')

print("calculating pacmap...")
pac5 = pacmap.PaCMAP(
    n_components=2,
    n_neighbors=5,
    MN_ratio=0.5,
    FP_ratio=2.0,
    distance="angular",
    random_state=3)

pac7 = pacmap.PaCMAP(
    n_components=2,
    n_neighbors=7,
    MN_ratio=0.5,
    FP_ratio=2.0,
    distance="angular",
    random_state=3)

# 10 sec for 720
zot_pac5 = pac5.fit_transform(np.array(embeddings_df))
zot_pac7 = pac7.fit_transform(np.array(embeddings_df))

print("building panel app...")
zot_pacs = [zot_pac5, zot_pac7]
plots = []
for zot_pac in zot_pacs:
    plots.append(
        tnt.BokehPlotPane(
            zot_pac,
            hover_text=zot_df["Title"],
            marker_size=(zot_df["Hearts"].fillna(0)+2)/50,
            show_legend=True,
            legend_location="top_right",
            sizing_mode='stretch_both',
            min_point_size=0.001,
            max_point_size=0.05,
        )
    )


data_view = tnt.SimpleDataPane(
    zot_df,
    sizing_mode="stretch_both", max_rows=400, max_cols=50)

plots[1].link(
    plots[0],
    selected="selected",
    bidirectional=True
)

data_view.link(
    plots[0],
    selected="selected",
    bidirectional=True
)


tag_legend = tnt.TagWidget(zot_df["Common Tags"] + zot_df["Hearts"].fillna(
    0).apply(lambda x: [(str(x) if x != 0 else "?") + " likeability"]))
tag_legend.link_to_plot(plots[0])

app = pn.Tabs(
    pn.Row(plots[0], tag_legend, name="pac7"),
    pn.Row(plots[1], tag_legend, name="pac5"),
    pn.Column(data_view, name="data"))

simplesearch = tnt.SimpleSearchWidget(plots[0], raw_dataframe=zot_df_lowercase)

app = pn.Column(simplesearch, app)

pn.serve(app)
# app.servable()
