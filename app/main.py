import functools
import logging
import logging.config
import os
import time

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
from matplotlib import table

from app.utils import *

# logging.config.fileConfig(fname=r"logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

dpg.create_context()
dpg.create_viewport(title="xrf_splitter", width=1920, height=1080)


@timeit
def sort_callback(sender: int, sort_specs: None | list[list[int]]):
    if sort_specs is None:
        return

    sort_col_id = sort_specs[0][0]
    sort_col_label: str = dpg.get_item_configuration(sort_col_id)["label"]
    table_data = dpg.get_item_user_data("results_table")

    if table_data is None or not table_data:
        return

    df: pd.DataFrame = table_data["df"]

    arr = df[sort_col_label].tolist()
    sorted_arr_ids = [
        b[0]
        for b in sorted(enumerate(arr), key=pd_sorter, reverse=sort_specs[0][1] < 0)
    ]

    df = df.iloc[sorted_arr_ids, :]

    table_data["df"] = df
    dpg.set_item_user_data("results_table", table_data)
    show_selected_rows(df)


def on_key_lq():
    if dpg.is_key_down(dpg.mvKey_Q):
        dpg.stop_dearpygui()


with dpg.handler_registry():
    dpg.add_key_press_handler(dpg.mvKey_Control, callback=on_key_lq)


pd_sorter = functools.cmp_to_key(sorter)


def enable_table_controls():
    dpg.enable_item("err col checkbox")
    dpg.show_item("err col checkbox")
    dpg.enable_item("junk col checkbox")
    dpg.show_item("junk col checkbox")
    dpg.enable_item("highlight")
    dpg.show_item("highlight")
    dpg.set_value("err col checkbox", False)
    dpg.set_value("junk col checkbox", False)
    toggle_table_columns(RESULTS_JUNK, False)
    toggle_table_columns(RESULT_KEYS_ERR, False)


def toggle_table_columns(keys: list[str], state: bool):
    table_data = dpg.get_item_user_data("results_table")
    if table_data is not None:
        original_df: pd.DataFrame = table_data["original_df"]
        current_df: pd.DataFrame = table_data["df"]
        if state:
            columns_to_show = current_df.columns.tolist() + keys
            original_columns = original_df.columns.tolist()
            ordered_intersection = sorted(
                set(original_columns) & set(columns_to_show), key=original_columns.index
            )
            df = original_df[ordered_intersection]
        else:
            df = current_df.drop(keys, axis=1)

        table_data["df"] = df
        dpg.set_item_user_data("results_table", table_data)
        show_selected_rows(df)
        unhighlight_table()


def select_rows(df: pd.DataFrame):
    row_range: tuple[int, int] = (dpg.get_value("from"), dpg.get_value("to"))

    if row_range[1] == -1:
        df = df.loc[pd.to_numeric(df["File #"]) >= row_range[0]]
    else:
        df = df.loc[
            pd.to_numeric(df["File #"]).isin(range(row_range[0], row_range[1] + 1))
        ]

    return df


def show_selected_rows(df):
    df = select_rows(df)

    with dpg.mutex():
        populate_table(df)


def row_select_callback(s, a):
    spectrum_label = dpg.get_item_configuration(s)["label"]
    data = dpg.get_item_user_data("primary")
    if data is None or not data:
        return

    table_data = dpg.get_item_user_data("results_table")
    if table_data is None or not table_data:
        return

    if table_data.get("selections") is None:
        table_data["selections"] = set()

    folder: str = data.get("pdz_folder")
    if folder is None or not folder:
        return

    try:
        file = [
            filename for filename in os.listdir(folder) if spectrum_label in filename
        ][0]
    except IndexError:
        file = None

    if not a and file:
        dpg.delete_item(file)
        table_data["selections"].remove(spectrum_label)

    if a and file:
        add_plot(f"{folder}/{file}")
        table_data["selections"].add(spectrum_label)


@timeit
def populate_table(df: pd.DataFrame):
    try:
        dpg.delete_item("results_table", children_only=True)
    except:
        pass

    arr = df.to_numpy()
    cols = df.columns

    table_data = dpg.get_item_user_data("results_table")
    if table_data is None or not table_data:
        table_data = {"selections": set()}

    selections: set[str] = table_data["selections"]

    for i in range(arr.shape[1]):
        dpg.add_table_column(
            label=cols[i],
            tag=f"{cols[i]}",
            parent="results_table",
            prefer_sort_ascending=False,
            prefer_sort_descending=True,
        )

    for i in range(arr.shape[0]):
        with dpg.table_row(parent="results_table"):
            for j in range(arr.shape[1]):
                if j == 0:
                    dpg.add_selectable(
                        label=f"{arr[i,j]}",
                        span_columns=True,
                        tag=f"{arr[i,j]}",
                        callback=row_select_callback,
                    )
                else:
                    dpg.add_selectable(
                        label=f"{arr[i,j]}",
                        span_columns=True,
                        callback=row_select_callback,
                    )

            dpg.configure_item(arr[i, 0], default_value=(arr[i, 0] in selections))

    if dpg.get_value("highlight"):
        highlight_table()
    else:
        unhighlight_table()


def create_table(path: Path):
    keys = RESULT_KEYS

    try:
        dpg.delete_item("results_table")
    except:
        pass

    csv_data = construct_data(path)
    selected_data = select_data(
        csv_data,
        [0],
        keys,
    )
    df = pd.read_csv(data_to_csv(selected_data, keys), dtype=str).fillna("")

    dpg.add_table(
        user_data={"original_df": df, "df": df, "path": path, "selections": set()},
        label="Results",
        tag="results_table",
        parent="data",
        header_row=True,
        hideable=False,
        resizable=True,
        clipper=True,
        freeze_columns=1,
        freeze_rows=1,
        scrollX=True,
        scrollY=True,
        sortable=False,
        callback=lambda s, a: sort_callback(s, a),
        delay_search=True,
        policy=dpg.mvTable_SizingFixedFit,
        borders_outerH=True,
        borders_innerV=True,
        borders_innerH=True,
        borders_outerV=True,
        reorderable=True,
        precise_widths=True,
        height=-1,
        width=-1,
    )

    with dpg.mutex():
        show_selected_rows(df)

    enable_table_controls()

    dpg.configure_item("results_table", sortable=True)


def csv_file_dialog_callback(_, app_data):
    create_table(Path(list(app_data["selections"].values())[0]))


def toggle_highlight_table():
    if dpg.get_value("highlight"):
        highlight_table()
    else:
        unhighlight_table()


def unhighlight_table():
    table_data = dpg.get_item_user_data("results_table")
    if table_data is None or not table_data:
        return

    df = select_rows(table_data["df"])

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            dpg.unhighlight_table_cell("results_table", i, j)


def highlight_table():
    table_data = dpg.get_item_user_data("results_table")
    if table_data is None or not table_data:
        return

    df = select_rows(table_data["df"])

    col_ids = [df.columns.get_loc(c) for c in RESULT_ELEMENTS if c in df]
    for row in range(df.shape[0]):
        arr = df.iloc[row, col_ids].replace(["< LOD", ""], 0).to_numpy().astype(float)
        t = np.nan_to_num(arr / np.max(arr), nan=0)
        for i, e in enumerate(t):
            sample = dpg.sample_colormap(dpg.mvPlotColormap_Jet, e)
            color = np.array(sample) * np.array(
                [255, 255, 255, np.clip([e * 100 + 25], 0, 100)[0]]
            )
            try:
                dpg.highlight_table_cell(
                    "results_table", row, col_ids[i], color.tolist()
                )
            except:
                pass


def add_plot(path: str):
    pdz = PDZFile(path)
    spectra_sum = [
        pdz.spectra[1].counts[i] + pdz.spectra[2].counts[i]
        for i, _ in enumerate(pdz.spectra[1].counts)
    ]

    dpg.add_line_series(
        pdz.spectra[2].energies,
        spectra_sum,
        tag=pdz.name,
        label=f"[1+2] {pdz.name}",
        parent="y_axis",
    )


def pdz_file_dialog_callback(_, app_data):
    data: dict | None = dpg.get_item_user_data("primary")
    if data is None:
        data = {}
    pdz_folder = app_data.get("file_path_name")
    dpg.set_item_user_data("primary", {**data, **{"pdz_folder": pdz_folder}})
    print(dpg.get_item_user_data("primary"))
    # for v in app_data["selections"].values():
    #     add_plot(v)


def file_dialog_cancel_callback(_s, _a):
    ...


with dpg.file_dialog(
    directory_selector=True,
    show=False,
    callback=pdz_file_dialog_callback,
    tag="pdz_dialog",
    file_count=1,
    cancel_callback=file_dialog_cancel_callback,
    width=700,
    height=400,
):
    dpg.add_file_extension(".pdz")
    dpg.add_file_extension("*")


with dpg.file_dialog(
    directory_selector=False,
    show=False,
    modal=True,
    default_filename="Results",
    callback=csv_file_dialog_callback,
    tag="csv_dialog",
    cancel_callback=file_dialog_cancel_callback,
    width=700,
    height=400,
):
    dpg.add_file_extension(".csv")
    dpg.add_file_extension("*")

with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(
            dpg.mvStyleVar_FrameBorderSize, 1, category=dpg.mvThemeCat_Core
        )

    with dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_style(
            dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
        )

    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(
            dpg.mvThemeCol_Header,
            (0, 119, 200, 60),
            category=dpg.mvThemeCat_Core,
        )

with dpg.window(label="xrfsplitter", tag="primary", autosize=True, user_data={}):
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save As")

            with dpg.menu(label="Settings"):
                dpg.add_menu_item(label="Setting 1", check=True)
                dpg.add_menu_item(label="Setting 2")

        dpg.add_menu_item(label="Help")
        with dpg.menu(label="Tools"):
            dpg.add_menu_item(
                label="Show About", callback=lambda: dpg.show_tool(dpg.mvTool_About)
            )
            dpg.add_menu_item(
                label="Show Metrics", callback=lambda: dpg.show_tool(dpg.mvTool_Metrics)
            )
            dpg.add_menu_item(
                label="Show Documentation",
                callback=lambda: dpg.show_tool(dpg.mvTool_Doc),
            )
            dpg.add_menu_item(
                label="Show Debug", callback=lambda: dpg.show_tool(dpg.mvTool_Debug)
            )
            dpg.add_menu_item(
                label="Show Style Editor",
                callback=lambda: dpg.show_tool(dpg.mvTool_Style),
            )
            dpg.add_menu_item(
                label="Show Font Manager",
                callback=lambda: dpg.show_tool(dpg.mvTool_Font),
            )
            dpg.add_menu_item(
                label="Show Item Registry",
                callback=lambda: dpg.show_tool(dpg.mvTool_ItemRegistry),
            )

    with dpg.group(horizontal=True):
        dpg.add_button(
            label="Select spectra folder",
            callback=lambda: dpg.show_item("pdz_dialog"),
        )

        dpg.add_button(
            label="Select results table",
            callback=lambda: dpg.show_item("csv_dialog"),
        )

    dpg.add_checkbox(
        label="Show Err columns",
        default_value=True,
        tag="err col checkbox",
        callback=lambda _s, a: toggle_table_columns(RESULT_KEYS_ERR, a),
        enabled=False,
        show=False,
    )

    dpg.add_checkbox(
        label="Show junk columns",
        default_value=True,
        tag="junk col checkbox",
        callback=lambda _s, a: toggle_table_columns(RESULTS_JUNK, a),
        enabled=False,
        show=False,
    )

    dpg.add_checkbox(
        label="Highlight table",
        tag="highlight",
        default_value=False,
        callback=toggle_highlight_table,
        show=False,
        enabled=(
            not dpg.get_value("err col checkbox")
            and not dpg.get_value("junk col checkbox")
        ),
    )

    dpg.add_text("File # range:")
    with dpg.group(horizontal=True):
        dpg.add_text("From")
        dpg.add_input_int(
            tag="from",
            width=100,
            max_value=10000,
            min_value=1,
            min_clamped=True,
            max_clamped=True,
            default_value=1850,
            on_enter=True,
            callback=lambda _s, _a: show_selected_rows(
                dpg.get_item_user_data("results_table").get("df")  # type:ignore
            ),
        )
        dpg.add_text("to")
        dpg.add_input_int(
            tag="to",
            width=100,
            max_value=10000,
            min_value=-1,
            default_value=1880,
            min_clamped=True,
            max_clamped=True,
            on_enter=True,
            callback=lambda s, _a: show_selected_rows(
                dpg.get_item_user_data("results_table").get("df")  # type:ignore
            ),
        )
        dpg.add_text("?", tag="range_tooltip")

        with dpg.tooltip("range_tooltip"):
            dpg.add_text("-1 indicates no upper limit")

    with dpg.group(tag="data", horizontal=False):
        with dpg.plot(
            label="Plots",
            tag="plots",
            height=600,
            width=-1,
            crosshairs=True,
            anti_aliased=True,
        ):
            dpg.add_plot_legend(location=9)
            dpg.add_plot_axis(dpg.mvXAxis, label="Energy, kEv")
            dpg.add_plot_axis(dpg.mvYAxis, label="Counts", tag="y_axis")

        with dpg.table(
            label="Results table",
            tag="results_table",
            user_data={},
            header_row=True,
            hideable=False,
            resizable=True,
            clipper=True,
            freeze_columns=1,
            freeze_rows=1,
            scrollX=True,
            scrollY=True,
            sortable=False,
            delay_search=True,
            policy=dpg.mvTable_SizingFixedFit,
            borders_outerH=True,
            borders_innerV=True,
            borders_innerH=True,
            borders_outerV=True,
            reorderable=True,
            precise_widths=True,
            height=-1,
            width=-1,
        ):
            dpg.add_table_column(label="File #")

    pdz_file_dialog_callback(
        "",
        {"file_path_name": "/home/puglet5/Documents/PROJ/test_data/smalts pdz"},
    )
    csv_file_dialog_callback(
        "",
        {
            "selections": {
                "1": "/home/puglet5/Documents/PROJ/XRFSplitter/test/fixtures/Results.csv"
            }
        },
    )


dpg.bind_theme(global_theme)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("primary", True)
dpg.start_dearpygui()
dpg.destroy_context()
