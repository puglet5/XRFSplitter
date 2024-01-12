import functools
import logging
import logging.config
import os
from turtle import delay

import coloredlogs
import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd

from app.utils import *

logging.basicConfig(filename="./log/main.log", filemode="a")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")

dpg.create_context()
dpg.create_viewport(
    title="xrf_splitter", width=2560, height=1440, vsync=True, decorated=False
)


@timeit
def sort_callback(_sender: int, sort_specs: None | list[list[int]]):
    if sort_specs is None:
        return

    sort_col_id = sort_specs[0][0]
    sort_col_label: str = dpg.get_item_configuration(sort_col_id)["label"]
    table_data: TableData | None = dpg.get_item_user_data("results_table")

    if table_data is None or not table_data:
        return

    table_data = table_data.copy()

    unsorted_df = table_data["df"]

    arr = unsorted_df[sort_col_label].tolist()
    sorted_arr_ids = [
        b[0]
        for b in sorted(enumerate(arr), key=pd_sorter, reverse=sort_specs[0][1] < 0)
    ]

    if list(range(unsorted_df.shape[0])) == sorted_arr_ids:
        return

    df = unsorted_df.iloc[sorted_arr_ids, :]

    table_data["df"] = df
    dpg.set_item_user_data("results_table", table_data)
    show_selected_rows(df)


def on_key_ctrl():
    if dpg.is_key_down(dpg.mvKey_Q):
        dpg.stop_dearpygui()
    if dpg.is_key_down(dpg.mvKey_Alt):
        if dpg.is_key_pressed(dpg.mvKey_M):
            menubar_visible = dpg.get_item_configuration("primary")["menubar"]
            dpg.configure_item("primary", menubar=(not menubar_visible))


with dpg.handler_registry():
    dpg.add_key_down_handler(dpg.mvKey_Control, callback=on_key_ctrl)


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


def toggle_table_columns(keys: list[str], state: bool):
    table_data: TableData | None = dpg.get_item_user_data("results_table")
    if table_data is None:
        return

    table_data = table_data.copy()

    with dpg.mutex():
        original_df = table_data["original_df"]
        current_df = table_data["df"]
        if state:
            columns_to_show = current_df.columns.tolist() + keys
            original_columns = original_df.columns.tolist()
            ordered_intersection = sorted(
                set(original_columns) & set(columns_to_show),
                key=original_columns.index,
            )
            filtered_df = original_df[columns_to_show]
        else:
            filtered_df = current_df.drop(keys, axis=1)

        if filtered_df.equals(current_df):
            return

        table_data["df"] = filtered_df
        dpg.set_item_user_data("results_table", table_data)

    show_selected_rows(filtered_df)


def select_rows(df: pd.DataFrame):
    row_range: tuple[int, int] = (dpg.get_value("from"), dpg.get_value("to"))

    if row_range[1] == -1:
        df = df.loc[pd.to_numeric(df["File #"]) >= row_range[0]]
    else:
        df = df.loc[
            pd.to_numeric(df["File #"]).isin(range(row_range[0], row_range[1] + 1))
        ]

    return df


def show_selected_rows(df: pd.DataFrame):
    with dpg.mutex():
        filtered_df = select_rows(df)
        populate_table(filtered_df)


def row_select_callback(s, a):
    spectrum_label = dpg.get_item_configuration(s)["label"]
    data = dpg.get_item_user_data("primary")
    if data is None or not data:
        return

    table_data: TableData | None = dpg.get_item_user_data("results_table")
    if table_data is None or not table_data:
        return

    if not table_data.get("selections"):
        table_data["selections"] = {}

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
        table_data["selections"].pop(spectrum_label)
    elif a and file:
        add_plot(f"{folder}/{file}")
        table_data["selections"][spectrum_label] = s
    elif a:
        table_data["selections"][spectrum_label] = s

    elif not a:
        table_data["selections"].pop(spectrum_label)


@timeit
def populate_table(df: pd.DataFrame):
    try:
        dpg.delete_item("results_table", children_only=True)
    except Exception:
        logger.warn("No table found")

    arr = df.to_numpy()
    cols = df.columns.to_numpy()

    table_data: TableData | None = dpg.get_item_user_data("results_table")
    if not table_data:
        return

    if not table_data["selections"]:
        selections = {}

    selections = table_data["selections"].copy()

    for col in cols:
        dpg.add_table_column(
            label=col,
            parent="results_table",
            prefer_sort_ascending=False,
            prefer_sort_descending=True,
        )

    for i in range(arr.shape[0]):
        with dpg.table_row(use_internal_label=False, parent="results_table"):
            dpg.add_selectable(
                label=f"{arr[i,0]}",
                span_columns=True,
                disable_popup_close=True,
                use_internal_label=False,
                default_value=(arr[i, 0] in selections),
                callback=row_select_callback,
                tag=selections.get(arr[i, 0], 0),
            )
            for j in range(1, arr.shape[1]):
                dpg.add_selectable(
                    label=f"{arr[i,j]}",
                    disable_popup_close=True,
                    use_internal_label=False,
                )

    if dpg.get_value("highlight"):
        unhighlight_table(df)
        highlight_table(df)
    else:
        unhighlight_table(df)


def deselect_all():
    table_data = dpg.get_item_user_data("results_table")
    if table_data is None or not table_data:
        table_data = {"selections": {}}

    selections: dict = table_data["selections"]

    for s in list(selections.values()):
        dpg.set_value(s, False)
        row_select_callback(s, False)


def setup_table(path: Path):
    keys = RESULT_KEYS

    if dpg.does_item_exist("results_table"):
        dpg.delete_item("results_table")

    csv_data = construct_data(path)
    selected_data = select_data(
        csv_data,
        [0],
        keys,
    )
    df = (
        pd.read_csv(
            data_to_csv(selected_data, keys),
            dtype=str,
            delimiter=",",
            header="infer",
            index_col=False,
            skipinitialspace=False,
        )
        .fillna("")
        .iloc[::-1]
    )

    table_data: TableData = {
        "original_df": df,
        "df": df,
        "path": path,
        "selections": {},
    }

    dpg.add_table(
        user_data=table_data,
        label="Results",
        tag="results_table",
        no_saved_settings=True,
        header_row=True,
        hideable=False,
        resizable=True,
        clipper=True,
        freeze_columns=1,
        freeze_rows=1,
        scrollX=True,
        scrollY=True,
        sortable=True,
        callback=lambda s, a: sort_callback(s, a),
        delay_search=False,
        policy=dpg.mvTable_SizingFixedFit,
        borders_outerH=True,
        borders_innerV=True,
        borders_innerH=True,
        borders_outerV=True,
        reorderable=False,
        precise_widths=False,
        height=-1,
        width=-1,
    )

    # show_selected_rows(df)
    cols_to_hide = list(set(RESULTS_JUNK) | set(RESULT_KEYS_ERR))
    toggle_table_columns(cols_to_hide, False)
    enable_table_controls()
    dpg.configure_item("results_table", sortable=True)


def csv_file_dialog_callback(_, app_data):
    path = Path(list(app_data["selections"].values())[0])
    setup_table(path)


def toggle_highlight_table(df):
    if dpg.get_value("highlight"):
        unhighlight_table(df)
        highlight_table(df)
    else:
        unhighlight_table(df)


def unhighlight_table(df):
    df = select_rows(df)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            dpg.unhighlight_table_cell("results_table", i, j)


def highlight_table(df):
    # table_rows = dpg.get_item_children("results_table", 1)
    # if not table_rows:
    #     return

    unhighlight_table(df)
    df = select_rows(df)
    col_ids = [df.columns.get_loc(c) for c in RESULT_ELEMENTS if c in df]
    arr = df.iloc[:, col_ids].replace(["< LOD", ""], 0).to_numpy().astype(float)
    for row_i, row in enumerate(arr):
        # if not dpg.is_item_visible(table_rows[row_i]):
        #     continue
        t = np.nan_to_num(row / np.max(row), nan=0.0)
        t = np.log(t + 0.01)
        t = np.interp(t, (t.min(), t.max()), (0.0, 1.0))
        for val, column in zip(t, col_ids):
            sample = dpg.sample_colormap(dpg.mvPlotColormap_Jet, val)
            norm = [255.0, 255.0, 255.0, min(val * 100.0 + 20.0, 100.0)]
            color = [int(sample[i] * norm[i]) for i in range(len(sample))]
            dpg.highlight_table_cell("results_table", row_i, column, color)


@timeit
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

with dpg.window(
    label="xrfsplitter",
    tag="primary",
    autosize=False,
    menubar=False,
    user_data={},
    delay_search=True,
    no_background=True,
):
    with dpg.menu_bar(tag="menu_bar"):
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
            dpg.add_menu_item(
                label="Show Colormap Registry",
                callback=lambda: dpg.show_item("__demo_colormap_registry"),
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

    dpg.add_colormap_registry(
        label="Demo Colormap Registry", tag="__demo_colormap_registry"
    )

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
        with dpg.group(tag="table_controls", horizontal=True):
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
                callback=lambda _s, _a: toggle_highlight_table(
                    dpg.get_item_user_data("results_table").get("df").copy()  # type: ignore
                ),
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
                    default_value=1788,
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
                    default_value=1872,
                    min_clamped=True,
                    max_clamped=True,
                    on_enter=True,
                    callback=lambda s, _a: show_selected_rows(
                        dpg.get_item_user_data("results_table").get("df")  # type:ignore
                    ),
                )
                dpg.add_text("(?)", tag="range_tooltip", color=(200, 200, 200, 100))

                with dpg.tooltip("range_tooltip"):
                    dpg.add_text("-1 indicates no upper limit")
            dpg.add_button(label="Deselect all", callback=lambda _s, _a: deselect_all())
        with dpg.table(
            label="Results table",
            tag="results_table",
            user_data={},
            header_row=True,
            hideable=False,
            resizable=False,
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
