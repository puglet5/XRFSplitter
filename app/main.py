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
dpg.create_viewport(title="xrf_splitter", width=1920, height=1080)
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
def sort_callback(_sender: int | str, sort_specs: None | list[list[int]]):
    if sort_specs is None:
        return

    sort_col_id = sort_specs[0][0]
    sort_col_label: str = dpg.get_item_configuration(sort_col_id)["label"]
    table_data: TableData | None = dpg.get_item_user_data(TABLE)

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
    dpg.set_item_user_data(TABLE, table_data)
    show_selected_rows(df)


def on_key_ctrl():
    if dpg.is_key_pressed(dpg.mvKey_Q):
        dpg.stop_dearpygui()
    if dpg.is_key_pressed(dpg.mvKey_C):
        selected_co_clipboard()
    if not dpg.is_key_down(dpg.mvKey_Shift) and dpg.is_key_pressed(dpg.mvKey_A):
        select_all_rows()
    if dpg.is_key_down(dpg.mvKey_Shift):
        if dpg.is_key_pressed(dpg.mvKey_A):
            deselect_all_rows()
    if dpg.is_key_down(dpg.mvKey_Alt):
        if dpg.is_key_pressed(dpg.mvKey_M):
            menubar_visible = dpg.get_item_configuration(WINDOW)["menubar"]
            dpg.configure_item(WINDOW, menubar=(not menubar_visible))


with dpg.handler_registry():
    dpg.add_key_down_handler(dpg.mvKey_Control, callback=on_key_ctrl)
    dpg.add_key_press_handler(dpg.mvKey_Back, callback=lambda: deselect_all_rows(TABLE))


def toggle_lod(state: bool):
    table_data: TableData | None = dpg.get_item_user_data(TABLE)
    if table_data is None:
        return

    table_data = table_data.copy()

    with dpg.mutex():
        original_df = table_data["original_df"]
        current_df = table_data["df"]
        if state:
            filtered_df = current_df.copy()
            filtered_df.update(
                original_df, overwrite=False, filter_func=lambda x: x == ""
            )
        else:
            filtered_df = current_df.replace("< LOD", "")

        # if filtered_df.equals(current_df):
        #     return

        table_data["df"] = filtered_df
        dpg.set_item_user_data(TABLE, table_data)

    show_selected_rows(filtered_df)


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


def toggle_table_columns(keys: list[str], state: bool, table=TABLE):
    table_data: TableData | None = dpg.get_item_user_data(table)
    if table_data is None:
        return

    table_data = table_data.copy()

    original_df = table_data["original_df"]
    current_df = table_data["df"]
    if state:
        columns_to_show = current_df.columns.tolist() + keys
        original_columns = original_df.columns.tolist()
        ordered_intersection = sorted(
            set(original_columns) & set(columns_to_show),
            key=original_columns.index,
        )
        filtered_df = original_df[ordered_intersection]
    else:
        filtered_df = current_df.drop(keys, axis=1)

    if filtered_df.equals(current_df):
        return

    table_data["df"] = filtered_df
    dpg.set_item_user_data(table, table_data)

    toggle_lod(dpg.get_value("lod_checkbox"))


def select_rows(df: pd.DataFrame):
    row_range: tuple[int, int] = (dpg.get_value("from"), dpg.get_value("to"))

    if row_range[1] == -1:
        df = df.loc[pd.to_numeric(df["File #"]) >= row_range[0]]
    else:
        df = df.loc[
            pd.to_numeric(df["File #"]).isin(range(row_range[0], row_range[1] + 1))
        ]

    return df


def show_selected_rows(df: pd.DataFrame, table=TABLE):
    filtered_df = select_rows(df)
    populate_table(filtered_df, table)


@log_exec_time
def select_all_rows(table=TABLE):
    children = dpg.get_item_children(table, 1)
    if children and isinstance(children, list):
        rows = children
    else:
        return

    table_data: TableData | None = dpg.get_item_user_data(table)
    if table_data is None or not table_data:
        return

    rows_n = len(rows)

    for i, r in enumerate(rows):
        dpg.set_value("table_progress", i / rows_n)
        if cells := dpg.get_item_children(r, 1):
            cell = cells[0]
            if not dpg.get_value(cell):
                dpg.set_value(cell, True)
                row_select_callback(cell, True, table=TABLE, td_up=table_data)
    dpg.set_value("table_progress", 0)


def row_select_callback(
    cell: int | str, value: bool, table=TABLE, td_up: TableData | None = None
):
    spectrum_label = dpg.get_item_label(cell)
    if spectrum_label is None:
        return

    data: dict | None = dpg.get_item_user_data(WINDOW)
    if data is None or not data:
        return

    folder: str = data.get("pdz_folder", None)
    if folder is None or not folder:
        return

    if td_up is None:
        table_data: TableData | None = dpg.get_item_user_data(table)
        if table_data is None or not table_data:
            return
    else:
        table_data = td_up

    selections = table_data.get("selections", {})

    files = [filename for filename in os.listdir(folder) if spectrum_label in filename]
    file = files[0] if files else None

    if value:
        selections[spectrum_label] = cell
        if file is not None:
            path = Path(f"{folder}/{file}")
            add_plot(path)
    else:
        selections.pop(spectrum_label, None)
        if file is not None:
            dpg.delete_item(file)


@log_exec_time
def populate_table(df: pd.DataFrame, table=TABLE):
    with dpg.mutex():
        try:
            dpg.delete_item(table, children_only=True)
        except Exception:
            logger.warn(f"No table found: {table}")

        arr = df.to_numpy()
        cols = df.columns.to_numpy()

        table_data: TableData | None = dpg.get_item_user_data(table)
        if not table_data:
            return

        if not table_data["selections"]:
            selections = {}

        selections = table_data["selections"].copy()

        for col in cols:
            dpg.add_table_column(
                label=col,
                parent=table,
                prefer_sort_ascending=False,
                prefer_sort_descending=True,
            )

    row_n = arr.shape[0]
    for i in range(row_n):
        dpg.set_value("table_progress", i / row_n)
        dpg.lock_mutex()
        with dpg.table_row(use_internal_label=False, parent=table):
            dpg.add_selectable(
                label=f"{arr[i,0]}",
                span_columns=True,
                disable_popup_close=True,
                use_internal_label=False,
                default_value=(arr[i, 0] in selections),
                callback=lambda s, a: row_select_callback(s, a),
                tag=selections.get(arr[i, 0], 0),
            )
            for j in range(1, arr.shape[1]):
                dpg.add_selectable(
                    label=f"{arr[i,j]}",
                    disable_popup_close=True,
                    use_internal_label=False,
                )
        dpg.unlock_mutex()

    dpg.set_value("table_progress", 0)

    with dpg.mutex():
        if dpg.get_value("table_highlight_checkbox"):
            highlight_table(df, table)
        else:
            unhighlight_table(df, table)


@log_exec_time
def deselect_all_rows(table=TABLE):
    table_data: TableData | None = dpg.get_item_user_data(table)
    if table_data is None or not table_data:
        return

    selections: dict = table_data.get("selections", None)
    if not selections:
        return

    cells = list(selections.values())
    cells_n = len(cells)

    for i, cell in enumerate(cells):
        dpg.set_value("table_progress", i / cells_n)
        if dpg.does_item_exist(cell):
            dpg.set_value(cell, False)
            row_select_callback(cell, False, table=TABLE, td_up=table_data)

    dpg.set_value("table_progress", 0)


def setup_table(csv_path: Path, table=TABLE):
    keys = RESULT_KEYS

    if dpg.does_item_exist(table):
        dpg.delete_item(table)

    csv_data = construct_data(csv_path)
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
        "path": csv_path,
        "selections": {},
    }

    dpg.add_table(
        user_data=table_data,
        label="Results",
        parent="table_wrapper",
        tag=table,
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
    dpg.configure_item(table, sortable=True)


def csv_file_dialog_callback(_, app_data: dict):
    selections = app_data.get("selections", None)
    if selections is None:
        return
    selection = list(selections.values())[0]
    setup_table(Path(selection))


def toggle_highlight_table(df: pd.DataFrame):
    if dpg.get_value("table_highlight_checkbox"):
        unhighlight_table(df, table=TABLE)
        highlight_table(df, table=TABLE)
    else:
        unhighlight_table(df, table=TABLE)


def unhighlight_table(df: pd.DataFrame, table=TABLE):
    df = select_rows(df)
    df_n = df.shape[0]
    for i in range(df.shape[0]):
        dpg.set_value("table_progress", i / df_n)
        for j in range(df.shape[1]):
            dpg.unhighlight_table_cell(table, i, j)
    dpg.set_value("table_progress", 0)


def selected_co_clipboard(table=TABLE):
    table_data: TableData | None = dpg.get_item_user_data(table)
    if table_data is None or not table_data:
        return

    selections: dict = table_data.get("selections", None)
    df = table_data.get("df", None)

    if not selections:
        return

    selected_rows = df[df["File #"].isin(selections.keys())]
    selected_rows.to_clipboard(excel=True, index=False)


def highlight_table(df: pd.DataFrame, table=TABLE):
    # table_rows = dpg.get_item_children(TABLE, 1)
    # if not table_rows:
    #     return

    unhighlight_table(df, table)
    df = select_rows(df)
    col_ids = [df.columns.get_loc(c) for c in RESULT_ELEMENTS if c in df]
    arr = df.iloc[:, col_ids].replace(["< LOD", ""], 0).to_numpy().astype(float)
    arr_n = len(arr)
    for row_i, row in enumerate(arr):
        dpg.set_value("table_progress", row_i / arr_n)
        # if not dpg.is_item_visible(table_rows[row_i]):
        #     continue
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


def pdz_file_dialog_callback(_, app_data: dict):
    data: dict | None = dpg.get_item_user_data(WINDOW)
    if data is None:
        data = {}
    pdz_folder = app_data.get("file_path_name", None)
    dpg.set_item_user_data(WINDOW, {**data, **{"pdz_folder": pdz_folder}})


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
                        callback=lambda _s, a: toggle_lod(a),
                        enabled=False,
                        show=False,
                    )

                    dpg.add_checkbox(
                        label="Highlight table",
                        tag="table_highlight_checkbox",
                        default_value=False,
                        callback=lambda _s, _a: toggle_highlight_table(
                            dpg.get_item_user_data(TABLE).get("df", None).copy()  # type: ignore
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
                            default_value=1772,
                            on_enter=True,
                            callback=lambda _s, _a: show_selected_rows(
                                dpg.get_item_user_data(TABLE).get(  # type:ignore
                                    "df", None
                                )
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
                                dpg.get_item_user_data(TABLE).get(  # type:ignore
                                    "df", None
                                )
                            ),
                        )
                        dpg.add_text(
                            "(?)", tag="range_tooltip", color=(200, 200, 200, 100)
                        )

                        with dpg.tooltip("range_tooltip"):
                            dpg.add_text("-1 indicates no upper limit")

                    dpg.add_button(
                        label="Deselect all",
                        callback=lambda _s, _a: deselect_all_rows(TABLE),
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
highlight_table(dpg.get_item_user_data(TABLE).get("df", None).copy())  # type: ignore
dpg.set_value("table_highlight_checkbox", True)

dpg.show_viewport()
dpg.set_primary_window(WINDOW, True)
dpg.start_dearpygui()
dpg.destroy_context()
