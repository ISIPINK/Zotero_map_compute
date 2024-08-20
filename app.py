import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import thisnotthat as tnt
import panel as pn
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
print("importing libraries...done")

pn.extension()
pn.extension('tabulator')

print("Loading data...")
zot_df = pd.read_csv('./data/zot_clean.csv')

# Convert the date columns to datetime objects
date_columns = ["Date", "Date Added", "Date Modified"]
for col in date_columns:
    zot_df[col] = pd.to_datetime(zot_df[col], errors='coerce', format="mixed")

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

# loading pacmap data
zot_pac5 = np.array(pd.read_csv('./data/zot_pac5.csv', index_col=0))
zot_pac7 = np.array(pd.read_csv('./data/zot_pac7.csv', index_col=0))

# loading clusters
clusters = np.array(pd.read_csv("./data/clusters.csv", index_col=0))
clusters = [str(label[0]) for label in clusters]

print("building panel app...")
# tabs for pacmap viz
plots = []
for zot_pac in [zot_pac5, zot_pac7]:
    plots.append(
        tnt.BokehPlotPane(
            zot_pac,
            hover_text=zot_df["Date"].dt.year.astype(
                str) + " " + zot_df["Title"],
            marker_size=(zot_df["Hearts"].fillna(0) + 2) / 50,
            labels=clusters,
            show_legend=False,
            legend_location="top_right",
            sizing_mode='stretch_both',
            min_point_size=0.001,
            max_point_size=0.05,
        )
    )

# tabs using dates
scaler = MinMaxScaler()
for date in [zot_df["Date Added"], zot_df["Date"]]:
    date_num = date.apply(lambda x: x.timestamp()).values.reshape(-1, 1)
    date_scaled = 20 * scaler.fit_transform(date_num)
    plots.append(
        tnt.BokehPlotPane(
            np.column_stack((date_scaled, zot_pac5[:, 0])),
            hover_text=zot_df["Date"].dt.year.astype(
                str) + " " + zot_df["Title"],
            marker_size=(zot_df["Hearts"].fillna(0) + 2) / 50,
            labels=clusters,
            show_legend=False,
            legend_location="top_right",
            sizing_mode='stretch_both',
            min_point_size=0.001,
            max_point_size=0.05,
        )
    )
# data view tab
data_view = tnt.SimpleDataPane(
    zot_df,
    sizing_mode="stretch_both", max_rows=9999, max_cols=50)

data_view_simple = tnt.SimpleDataPane(
    zot_df[["Title", "Author", "Date", "Abstract Note"]],
    sizing_mode="stretch_both", max_rows=9999, max_cols=50)

# linking the tabs
for i in range(1, len(plots)):
    plots[i].link(
        plots[0],
        selected="selected",
        bidirectional=True
    )

data_view.link(
    plots[0],
    selected="selected",
    bidirectional=True
)

data_view_simple.link(
    plots[0],
    selected="selected",
    bidirectional=True
)
# tag selector
tag_legend = tnt.TagWidget(zot_df["Common Tags"] + zot_df["Hearts"].fillna(
    0).apply(lambda x: [(str(x) if x != 0 else "?") + " likeability"]))
tag_legend.link_to_plot(plots[0])

tabs = [pn.Row(plot, tag_legend, name=name)
        for plot, name in zip(plots, ["pac5", "pac7", "Date Added", "Date"])]


# tags stats tab, just a plot
# Flatten the list of tags
all_tags = [tag for sublist in zot_df["Manual Tags"]
            for tag in sublist if tag != ""]

# Count the occurrences of each tag
tag_counts = Counter(all_tags)

total_tags = sum(tag_counts.values())

# Filter out tags that are less than 0.5% of the total
filtered_tag_counts = {tag: count for tag,
                       count in tag_counts.items() if count / total_tags >= 0.005}

# Sort the tags by frequency
sorted_tag_counts = dict(
    sorted(filtered_tag_counts.items(), key=lambda item: item[1], reverse=True))

# Create a figure with 2 subplots
plt.figure(figsize=(18, 9))

# First subplot: Most Common Tags
plt.subplot(1, 2, 1)
plt.barh(list(sorted_tag_counts.keys())[
         ::-1], list(sorted_tag_counts.values())[::-1], color='skyblue')
plt.xlabel('Counts')
plt.ylabel('Tags')
plt.title('Most Common Tags')
plt.xscale('log')
plt.tight_layout()

# Flatten the list of tags and their corresponding hearts
all_tags = [(tag, hearts) for tags, hearts in zip(
    zot_df["Manual Tags"], zot_df["Hearts"].fillna(1)) for tag in tags if tag != ""]

# Sum the hearts for each tag
tag_hearts = Counter()
for tag, hearts in all_tags:
    tag_hearts[tag] += hearts

# Divide the total hearts by the frequency of each tag
weighted_tag_counts = {
    tag: tag_hearts[tag] / count for tag, count in tag_counts.items() if count >= 9}

# Sort the tags by frequency
sorted_tag_counts = dict(
    sorted(weighted_tag_counts.items(), key=lambda item: item[1], reverse=True))

# Second subplot: Most Likeable Tags (Avg Hearts/Item)
plt.subplot(1, 2, 2)
plt.barh(list(sorted_tag_counts.keys())[
         ::-1], list(sorted_tag_counts.values())[::-1], color='skyblue')
plt.xlabel('Avg Hearts/Item')
plt.ylabel('Tags')
plt.title('Most Likeable Tags (Avg Hearts/Item)')
plt.tight_layout()

# Convert the plot to a Panel object
tag_plot = pn.pane.Matplotlib(plt.gcf(), sizing_mode='stretch_both')

app = pn.Tabs(
    *tabs,
    pn.Column(data_view_simple, name="data simple"),
    pn.Column(tag_plot, name="tags stats"),
    pn.Column(data_view, name="data all"))

# add simple search
simplesearch = tnt.SimpleSearchWidget(plots[0], raw_dataframe=zot_df_lowercase)
app = pn.Column(simplesearch, app)

pn.serve(app)
# app.servable()
