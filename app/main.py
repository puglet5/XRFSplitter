import logging
import logging.config
from operator import call

import dearpygui.dearpygui as dpg
import pandas as pd

from app.utils import *

# logging.config.fileConfig(fname=r"logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

dpg.create_context()
dpg.create_viewport(title="xrf_splitter", width=1920, height=1080)


def sort_callback(sender: int, sort_specs: None | list[list[int]]):
    def _sorter(e):
        return e[1]

    if sort_specs is None:
        return

    rows = dpg.get_item_children(sender, 1)

    if rows is None:
        return

    sort_col_id = sort_specs[0][0]

    cols: list[int] = dpg.get_item_children(sender)[0]  # type: ignore

    sort_column = sort_col_id - cols[0]

    sortable_list = []

    for row in rows:
        cells: list[int] = dpg.get_item_children(row, 1)  # type: ignore
        sortable_list.append([row, dpg.get_item_label(cells[sort_column])])

    sortable_list.sort(key=_sorter, reverse=sort_specs[0][1] < 0)
    new_order = [pair[0] for pair in sortable_list]

    dpg.reorder_items(sender, 1, new_order)


def show_table_columns(keys: list[str]):
    for k in RESULT_KEYS:
        dpg.configure_item(k, enabled=False)
        dpg.hide_item(k)

    for k in keys:
        dpg.configure_item(k, enabled=True)
        dpg.show_item(k)

    print([i for i, e in enumerate(keys) if e in RESULT_KEYS])


def toggle_err_columns():
    if dpg.get_item_configuration("Li Err")["show"]:
        show_table_columns(RESULT_KEYS_NO_ERR)
    else:
        show_table_columns(RESULT_KEYS)


def on_key_lq():
    if dpg.is_key_down(dpg.mvKey_Q):
        dpg.stop_dearpygui()


with dpg.handler_registry():
    dpg.add_key_press_handler(dpg.mvKey_Control, callback=on_key_lq)


def create_table(path: Path):
    if dpg.get_value("err col checkbox"):
        keys = RESULT_KEYS
    else:
        keys = RESULT_KEYS_NO_ERR

    try:
        dpg.delete_item("results")
    except:
        pass

    csv_data = construct_data(path)
    selected_data = select_data(csv_data, [0, 3000], keys)
    df = pd.read_csv(data_to_csv(selected_data, keys), dtype=str).fillna("")
    df = df.apply(pd.to_numeric, errors="coerce", downcast="signed").fillna(df)
    arr = df.to_numpy()

    with dpg.table(
        user_data=path,
        label="Results",
        tag="results",
        parent="stuff",
        header_row=True,
        resizable=True,
        clipper=False,
        freeze_columns=1,
        scrollX=True,
        scrollY=True,
        sortable=True,
        callback=sort_callback,
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
        for i in range(df.shape[1]):
            dpg.add_table_column(label=df.columns[i], tag=f"{df.columns[i]}")
        for i in range(df.shape[0]):
            with dpg.table_row():
                for j in range(df.shape[1]):
                    dpg.add_selectable(label=f"{arr[i,j]}", span_columns=True)

    dpg.enable_item("err col checkbox")
    dpg.enable_item("empty col checkbox")

    dpg.show_item("err col checkbox")
    dpg.show_item("empty col checkbox")


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
        parent="stuff",
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


def file_dialog_cancel_callback():
    return None


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
    with dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_style(
            dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
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
        callback=lambda _s, _a: toggle_err_columns(),
        enabled=False,
        show=False,
    )
    dpg.add_checkbox(
        label="Remove empty columns",
        default_value=False,
        tag="empty col checkbox",
        enabled=False,
        show=False,
    )

    dpg.add_group(tag="stuff", horizontal=False)

dpg.bind_theme(global_theme)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("primary", True)
dpg.start_dearpygui()
dpg.destroy_context()
