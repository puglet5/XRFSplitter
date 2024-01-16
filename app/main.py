import logging
import logging.config
import os

import coloredlogs
import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd

from app.utils import *

logging.basicConfig(filename="./log/main.log", filemode="a")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")

dpg.create_context()
dpg.create_viewport(title="xrf_splitter", width=1920, height=1080, vsync=True)
dpg.configure_app(
    # docking=True,
    # docking_space=False,
    # init_file="./dgp.ini"
)

TABLE = "results_table"
WINDOW = "primary_window"


def save_init():
    dpg.save_init_file("./dpg.ini")


@log_exec_time
def sort_callback(sender: int | str, sort_specs: None | list[list[int]]):
    if sort_specs is None:
        return

    sort_col_label: str = dpg.get_item_configuration(sort_specs[0][0])["label"]

    reverse = sort_specs[0][1] < 0

    current = table_data.current

    sorted = table_data.sort(sort_col_label, reverse)

    if current.equals(sorted):
        return

    show_selected_rows(table_data.current)


def on_key_ctrl():
    if dpg.is_key_pressed(dpg.mvKey_Q):
        dpg.stop_dearpygui()
    if dpg.is_key_pressed(dpg.mvKey_C):
        selected_co_clipboard()
    if dpg.is_key_down(dpg.mvKey_Shift):
        if dpg.is_key_pressed(dpg.mvKey_A):
            select_all_rows()
        if dpg.is_key_pressed(dpg.mvKey_D):
            deselect_all_rows()
    if dpg.is_key_down(dpg.mvKey_Alt):
        if dpg.is_key_pressed(dpg.mvKey_M):
            menubar_visible = dpg.get_item_configuration(WINDOW)["menubar"]
            dpg.configure_item(WINDOW, menubar=(not menubar_visible))


def toggle_lod():
    show_selected_rows(table_data.toggle_lod(dpg.get_value("lod_checkbox")))


def enable_table_controls():
    # dpg.enable_item("err_col_checkbox")
    # dpg.show_item("err_col_checkbox")
    dpg.enable_item("junk_col_checkbox")
    dpg.show_item("junk_col_checkbox")
    dpg.enable_item("table_highlight_checkbox")
    dpg.show_item("table_highlight_checkbox")
    dpg.enable_item("lod_checkbox")
    dpg.show_item("lod_checkbox")
    dpg.set_value("err_col_checkbox", False)
    dpg.set_value("junk_col_checkbox", False)
    dpg.set_value("lod_checkbox", True)


def toggle_table_columns(
    keys: list[str],
    state: bool,
):
    table_data.toggle_columns(keys, state)
    toggle_lod()


def select_rows(
    df: pd.DataFrame,
):
    row_range: tuple[int, int] = (dpg.get_value("from"), dpg.get_value("to"))
    return table_data.select_rows_range(df, row_range)


def show_selected_rows(df: pd.DataFrame):
    populate_table(select_rows(df))


@log_exec_time
def select_all_rows():
    rows: list[int] = dpg.get_item_children(TABLE, 1)  # type:ignore
    rows_n = len(rows)

    for i, r in enumerate(rows):
        dpg.set_value("table_progress", i / rows_n)
        if cells := dpg.get_item_children(r, 1):
            cell = cells[0]
            dpg.set_value(cell, True)
            row_select_callback(cell, True)

    dpg.set_value("table_progress", 0)


def row_select_callback(cell: int, value: bool):
    global pdz_folder

    spectrum_label = dpg.get_item_label(cell)
    if spectrum_label is None:
        return

    selections = table_data.selections

    files = [
        filename for filename in os.listdir(pdz_folder) if spectrum_label in filename
    ]
    file = files[0] if files else None

    if value:
        selections[spectrum_label] = cell
        if file is not None:
            path = Path(f"{pdz_folder}/{file}")
            add_plot(path)
    else:
        selections.pop(spectrum_label, None)
        if file is not None:
            dpg.delete_item(file)


@log_exec_time
def populate_table(df: pd.DataFrame):
    with dpg.mutex():
        try:
            dpg.delete_item(TABLE, children_only=True)
        except Exception:
            logger.warn(f"No table found: {TABLE}")

        arr = df.to_numpy()
        cols = df.columns.to_numpy()

        for col in cols:
            dpg.add_table_column(
                label=col,
                parent=TABLE,
                prefer_sort_ascending=False,
                prefer_sort_descending=True,
            )

    selections = table_data.selections

    row_n = arr.shape[0]
    for i in range(row_n):
        dpg.set_value("table_progress", i / row_n)
        with dpg.table_row(use_internal_label=False, parent=TABLE):
            label = arr[i, 0]
            dpg.add_selectable(
                label=label,
                span_columns=True,
                default_value=(label in selections),
                callback=row_select_callback,
                tag=selections.get(arr[i, 0], 0),
            )
            for j in range(1, arr.shape[1]):
                dpg.add_selectable(
                    label=arr[i, j],
                    disable_popup_close=True,
                    use_internal_label=False,
                )

    dpg.set_value("table_progress", 0)

    if dpg.get_value("table_highlight_checkbox"):
        highlight_table(df)


@log_exec_time
def deselect_all_rows():
    cells = list(table_data.selections.values())
    cells_n = len(cells)

    for i, cell in enumerate(cells):
        dpg.set_value("table_progress", i / cells_n)
        if dpg.does_item_exist(cell):
            dpg.set_value(cell, False)
            row_select_callback(cell, False)

    dpg.set_value("table_progress", 0)


def setup_table(csv_path: Path):
    if dpg.does_item_exist(TABLE):
        dpg.delete_item(TABLE)

    global table_data
    table_data = TableData(csv_path)

    dpg.add_table(
        label="Results",
        parent="table_wrapper",
        tag=TABLE,
        no_saved_settings=True,
        header_row=True,
        hideable=False,
        resizable=False,
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
        reorderable=False,
        precise_widths=True,
        height=-1,
        width=-1,
    )

    cols_to_hide = list(set(RESULTS_JUNK) | set(RESULT_KEYS_ERR))
    toggle_table_columns(cols_to_hide, False)
    enable_table_controls()
    dpg.configure_item(TABLE, sortable=True)


def csv_file_dialog_callback(_, app_data: dict):
    selections = app_data.get("selections", None)
    if selections is None:
        return
    selection = list(selections.values())[0]
    setup_table(Path(selection))


def toggle_highlight_table(df: pd.DataFrame):
    if dpg.get_value("table_highlight_checkbox"):
        unhighlight_table(
            df,
        )
        highlight_table(
            df,
        )
    else:
        unhighlight_table(
            df,
        )


def unhighlight_table(
    df: pd.DataFrame,
):
    df = select_rows(df)
    df_n = df.shape[0]
    for i in range(df.shape[0]):
        dpg.set_value("table_progress", i / df_n)
        for j in range(df.shape[1]):
            dpg.unhighlight_table_cell(TABLE, i, j)
    dpg.set_value("table_progress", 0)


def selected_co_clipboard():
    table_data.selected_to_clipboard()


def highlight_table(
    df: pd.DataFrame,
):
    unhighlight_table(df)
    df = select_rows(df)
    col_ids = [df.columns.get_loc(c) for c in RESULT_ELEMENTS if c in df]
    arr = df.iloc[:, col_ids].replace(["< LOD", ""], 0).to_numpy().astype(float)
    arr_n = len(arr)
    for row_i, row in enumerate(arr):
        dpg.set_value("table_progress", row_i / arr_n)
        t = np.nan_to_num(row / np.max(row), nan=0.0)
        t = np.log(t + 0.01)
        t = np.interp(t, (t.min(), t.max()), (0.0, 1.0))
        for val, column in zip(t, col_ids):
            sample = dpg.sample_colormap(dpg.mvPlotColormap_Jet, val)
            norm = [255.0, 255.0, 255.0, min(val * 100.0 + 20.0, 100.0)]
            color = [int(sample[i] * norm[i]) for i in range(len(sample))]
            dpg.highlight_table_cell(TABLE, row_i, column, color)
    dpg.set_value("table_progress", 0)


def add_plot(path: Path):
    pdz = PDZFile(path)
    spectra_sum = [
        pdz.spectra[1].counts[i] + pdz.spectra[2].counts[i]
        for i, _ in enumerate(pdz.spectra[1].counts)
    ]

    if dpg.does_item_exist(pdz.name):
        return

    dpg.add_line_series(
        pdz.spectra[2].energies,
        spectra_sum,
        tag=pdz.name,
        label=f"[1+2] {pdz.name}",
        parent="y_axis",
    )

    dpg.fit_axis_data("x_axis")
    dpg.fit_axis_data("y_axis")


def pdz_file_dialog_callback(_, app_data: dict[str, str]):
    global pdz_folder
    pdz_folder = app_data.get("file_path_name", "")


with dpg.handler_registry():
    dpg.add_key_down_handler(dpg.mvKey_Control, callback=on_key_ctrl)

with dpg.file_dialog(
    directory_selector=True,
    show=False,
    callback=pdz_file_dialog_callback,
    tag="pdz_dialog",
    file_count=1,
    width=700,
    height=400,
    modal=True,
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
    tag=WINDOW,
    autosize=True,
    user_data={},
    delay_search=True,
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
        with dpg.child_window(border=False, width=350, tag="sidebar"):
            with dpg.child_window(width=-1, height=60):
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Select spectra folder",
                        callback=lambda: dpg.show_item("pdz_dialog"),
                    )

                    dpg.add_button(
                        label="Select results table",
                        callback=lambda: dpg.show_item("csv_dialog"),
                    )

            with dpg.child_window(width=-1, height=-1):
                with dpg.group(tag="table_controls", horizontal=False):
                    dpg.add_checkbox(
                        label="Show Err columns",
                        default_value=True,
                        tag="err_col_checkbox",
                        callback=lambda _s, a: toggle_table_columns(RESULT_KEYS_ERR, a),
                        enabled=False,
                        show=False,
                    )

                    dpg.add_checkbox(
                        label="Show junk columns",
                        default_value=True,
                        tag="junk_col_checkbox",
                        callback=lambda _s, a: toggle_table_columns(RESULTS_JUNK, a),
                        enabled=False,
                        show=False,
                    )

                    dpg.add_checkbox(
                        label="Show '< LOD'",
                        default_value=True,
                        tag="lod_checkbox",
                        callback=lambda _s, a: toggle_lod(),
                        enabled=False,
                        show=False,
                    )

                    dpg.add_checkbox(
                        label="Highlight table",
                        tag="table_highlight_checkbox",
                        default_value=False,
                        callback=lambda _s, _a: toggle_highlight_table(
                            table_data.current
                        ),
                        show=False,
                        enabled=(
                            not dpg.get_value("err_col_checkbox")
                            and not dpg.get_value("junk_col_checkbox")
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
                            default_value=1,
                            on_enter=True,
                            callback=lambda _s, _a: toggle_lod(),
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
                            callback=lambda s, _a: toggle_lod(),
                        )
                        dpg.add_text(
                            "(?)", tag="range_tooltip", color=(200, 200, 200, 100)
                        )

                        with dpg.tooltip("range_tooltip"):
                            dpg.add_text("-1 indicates no upper limit")

                    dpg.add_button(
                        label="Select all",
                        callback=lambda _s, _a: select_all_rows(),
                    )

                    dpg.add_button(
                        label="Deselect all",
                        callback=lambda _s, _a: deselect_all_rows(),
                    )

        with dpg.child_window(border=False, width=-1, tag="data"):
            with dpg.collapsing_header(label="PDZ Plots", default_open=False):
                with dpg.plot(
                    tag="plots",
                    crosshairs=True,
                    anti_aliased=True,
                    height=300,
                    width=-1,
                ):
                    dpg.add_plot_legend(location=9)
                    dpg.add_plot_axis(dpg.mvXAxis, label="Energy, kEv", tag="x_axis")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Counts", tag="y_axis")
                dpg.add_spacer(height=4)
                dpg.add_separator()
                dpg.add_spacer(height=4)

            with dpg.collapsing_header(
                label="Results Table", default_open=True, tag="table_wrapper"
            ):
                dpg.add_progress_bar(tag="table_progress", width=-1, height=10)
                with dpg.table(
                    label="Results table",
                    tag=TABLE,
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


dpg.bind_theme(global_theme)
dpg.setup_dearpygui()

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
highlight_table(table_data.current)  # type: ignore
dpg.set_value("table_highlight_checkbox", True)

dpg.show_viewport()
dpg.set_primary_window(WINDOW, True)
dpg.start_dearpygui()
dpg.destroy_context()
