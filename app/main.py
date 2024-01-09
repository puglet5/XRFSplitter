import functools
import logging
import logging.config
import time

import dearpygui.dearpygui as dpg
import pandas as pd

from app.utils import *

# logging.config.fileConfig(fname=r"logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

dpg.create_context()
dpg.create_viewport(title="xrf_splitter", width=1920, height=1080)


def sort_callback(sender: int, sort_specs: None | list[list[int]]):
    if sort_specs is None:
        return

    sort_col_id = sort_specs[0][0]
    sort_col_label: str = dpg.get_item_configuration(sort_col_id)["label"]
    table_data = dpg.get_item_user_data("results")

    if table_data is None:
        return

    df: pd.DataFrame = table_data["df"].copy()
    key = functools.cmp_to_key(sorter)

    arr = df[sort_col_label].tolist()
    sorted_arr_ids = [
        b[0] for b in sorted(enumerate(arr), key=key, reverse=sort_specs[0][1] < 0)
    ]

    df = pd.DataFrame(df.reset_index(drop=True), index=sorted_arr_ids)
    df.reset_index(drop=True, inplace=True)

    with dpg.mutex():
        table_data["df"] = df
        dpg.set_item_user_data("results", table_data)
        show_selected_rows()


def on_key_lq():
    if dpg.is_key_down(dpg.mvKey_Q):
        dpg.stop_dearpygui()


with dpg.handler_registry():
    dpg.add_key_press_handler(dpg.mvKey_Control, callback=on_key_lq)


def enable_table_controls():
    dpg.enable_item("err col checkbox")
    dpg.show_item("err col checkbox")
    dpg.enable_item("junk col checkbox")
    dpg.show_item("junk col checkbox")


def toggle_table_columns(keys: list[str], state: bool):
    table_data = dpg.get_item_user_data("results")
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

        with dpg.mutex():
            table_data["df"] = df
            dpg.set_item_user_data("results", table_data)
            show_selected_rows()


def select_rows(df: pd.DataFrame):
    row_range: tuple[int, int] = (dpg.get_value("from"), dpg.get_value("to"))
    if row_range[1] == -1:
        df = df.loc[pd.to_numeric(df["File #"]) >= row_range[0]]
    else:
        df = df.loc[
            pd.to_numeric(df["File #"]).isin(range(row_range[0], row_range[1] + 1))
        ]

    return df


def show_selected_rows():
    table_data = dpg.get_item_user_data("results")
    if table_data is not None:
        df: pd.DataFrame = table_data["df"]
        df = select_rows(df)

        with dpg.mutex():
            populate_table(df)


def populate_table(df: pd.DataFrame):
    try:
        dpg.delete_item("results", children_only=True)
    except:
        pass

    arr = df.to_numpy()

    for i in range(df.shape[1]):
        dpg.add_table_column(
            label=df.columns[i],
            tag=f"{df.columns[i]}",
            parent="results",
            prefer_sort_ascending=False,
            prefer_sort_descending=True,
        )
    for i in range(df.shape[0]):
        with dpg.table_row(parent="results"):
            for j in range(df.shape[1]):
                dpg.add_selectable(label=f"{arr[i,j]}", span_columns=True)


def create_table(path: Path):
    keys = RESULT_KEYS

    try:
        dpg.delete_item("results")
    except:
        pass

    csv_data = construct_data(path)
    selected_data = select_data(
        csv_data,
        [0],
        keys,
    )
    df = pd.read_csv(data_to_csv(selected_data, keys), dtype=str).fillna("")
    # df = df.apply(pd.to_numeric, errors="coerce", downcast="signed").fillna(df)

    dpg.add_table(
        user_data={"original_df": df, "df": df, "path": path},
        label="Results",
        tag="results",
        parent="data",
        header_row=True,
        hideable=False,
        resizable=True,
        clipper=True,
        freeze_columns=1,
        freeze_rows=1,
        scrollX=True,
        scrollY=True,
        sortable=True,
        callback=sort_callback,
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
        populate_table(df)

    enable_table_controls()


def csv_file_dialog_callback(_, app_data):
    create_table(Path(app_data["file_path_name"]))
    return None


def pdz_file_dialog_callback(_, app_data):
    pdz = PDZFile(app_data["file_path_name"])
    spectra_sum = [
        pdz.spectra[1].counts[i] + pdz.spectra[2].counts[i]
        for i, _ in enumerate(pdz.spectra[1].counts)
    ]

    with dpg.plot(
        label="Plots",
        tag="plots",
        parent="data",
        before="results",
        height=600,
        width=-1,
        crosshairs=True,
        anti_aliased=True,
    ):
        dpg.add_plot_legend(location=9)
        dpg.add_plot_axis(dpg.mvXAxis, label="Energy, kEv")
        dpg.add_plot_axis(dpg.mvYAxis, label="Counts", tag="y_axis")

        dpg.add_line_series(
            pdz.spectra[1].energies,
            pdz.spectra[1].counts,
            label=f"[1] {pdz.name}",
            parent="y_axis",
        )

        dpg.add_line_series(
            pdz.spectra[2].energies,
            pdz.spectra[2].counts,
            label=f"[2] {pdz.name}",
            parent="y_axis",
        )

        dpg.add_line_series(
            pdz.spectra[2].energies,
            spectra_sum,
            label=f"[1+2] {pdz.name}",
            parent="y_axis",
        )

    return None


def file_dialog_cancel_callback(_s, _a):
    ...


with dpg.file_dialog(
    directory_selector=False,
    show=False,
    callback=pdz_file_dialog_callback,
    tag="pdz_dialog",
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
    default_filename="results",
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

with dpg.window(label="xrfsplitter", tag="primary"):
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
            label="Select spectra",
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

    dpg.add_text("File # range")
    with dpg.group(horizontal=True):
        dpg.add_text("From")
        dpg.add_input_int(
            tag="from",
            width=100,
            max_value=10000,
            min_value=1,
            min_clamped=True,
            max_clamped=True,
            default_value=1,
            on_enter=True,
            callback=lambda s_, a_: show_selected_rows(),
        )
        dpg.add_text("to")
        dpg.add_input_int(
            tag="to",
            width=100,
            max_value=10000,
            min_value=-1,
            default_value=-1,
            min_clamped=True,
            max_clamped=True,
            on_enter=True,
            callback=lambda s_, a_: show_selected_rows(),
        )
        dpg.add_text("?", tag="range_tooltip")

        with dpg.tooltip("range_tooltip"):
            dpg.add_text("-1 indicates no upper limit")

    dpg.add_group(tag="data", horizontal=False)

    pdz_file_dialog_callback(
        "",
        {
            "file_path_name": "/home/puglet5/Documents/PROJ/XRFSplitter/test/fixtures/01968-GeoMining.pdz"
        },
    )
    csv_file_dialog_callback(
        "",
        {
            "file_path_name": "/home/puglet5/Documents/PROJ/XRFSplitter/test/fixtures/results.csv"
        },
    )

dpg.bind_theme(global_theme)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("primary", True)
dpg.start_dearpygui()
dpg.destroy_context()
