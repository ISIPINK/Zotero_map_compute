# runs 1.5 min on my desktop loading numba I guess
import pandas as pd
import thisnotthat as tnt
import panel as pn
import pickle


def main():
    pn.extension()
    pn.extension('tabulator')

    with open('word_umap.pkl', 'rb') as f:
        word_map = pickle.load(f)

    with open('word_pacmap2.pkl', 'rb') as f:
        word_map2 = pickle.load(f)

    with open('word_pacmap3.pkl', 'rb') as f:
        word_map3 = pickle.load(f)

    df = pd.read_csv("Mijn Bibliotheek.csv")
    df = df.dropna(subset=["Title", "Abstract Note"])
    access_dates_month = pd.to_datetime(df["Access Date"]).fillna(
        pd.Timestamp.min).dt.strftime('%B')

    COLOR_KEY = {
        'January': '#1f77b4',     # Dark blue
        'February': '#268bd2',    # Slightly lighter blue
        'March': '#2ca02c',       # Green
        'April': '#36a641',       # Slightly brighter green
        'May': '#fcbf49',         # Yellow
        'June': '#fca946',        # Slightly brighter orange
        'July': '#fc8f44',        # Orange
        'August': '#e47745',      # Slightly darker brown
        'September': '#d16446',   # Brown
        'October': '#bb5147',     # Slightly darker red
        'December': '#a73f48'     # Purple
    }

    basic_plot1 = tnt.BokehPlotPane(
        word_map,
        labels=access_dates_month,
        label_color_mapping=COLOR_KEY,
        hover_text=df["Title"],
        show_legend=True,
        legend_location="top_right",
        sizing_mode='stretch_both')

    basic_plot2 = tnt.BokehPlotPane(
        word_map2,
        labels=access_dates_month,
        label_color_mapping=COLOR_KEY,
        hover_text=df["Title"],
        show_legend=True,
        legend_location="top_right",
        sizing_mode='stretch_both')

    basic_plot3 = tnt.BokehPlotPane(
        word_map3,
        labels=access_dates_month,
        label_color_mapping=COLOR_KEY,
        hover_text=df["Title"],
        show_legend=True,
        legend_location="top_right",
        sizing_mode='stretch_both')

    data_view = tnt.SimpleDataPane(
        df[["Item Type", "Title", "Author", "Abstract Note",
            "Url", "Date", "Access Date", "Manual Tags"]],
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

    column1 = pn.Column(basic_plot1, name="umap")
    column2 = pn.Column(basic_plot2, name="pacmap nn 5")
    column3 = pn.Column(basic_plot3, name="pacmap nn 7")
    column4 = pn.Column(data_view, name="data")

    app = pn.Tabs(
        column1,
        column2,
        column3,
        column4)

    pn.serve(app)


if __name__ == "__main__":
    main()
