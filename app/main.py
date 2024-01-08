import logging
import logging.config

import dearpygui.dearpygui as dpg
import pandas as pd

from app.utils import *

# logging.config.fileConfig(fname=r"logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

filepath = "test/fixtures/01968-GeoMining.pdz"
pdz = PDZFile(filepath)

csv_data = construct_data(Path("test/fixtures/results.csv"))
selected_data = select_data(csv_data, [0, 3000])
df = pd.read_csv(data_to_csv(selected_data), dtype=str).fillna("")
df = df.apply(pd.to_numeric, errors="coerce", downcast="signed").fillna(df)
arr = df.to_numpy()


def sort_callback(sender: int, sort_specs: None | list[list[int]]):
    if sort_specs is None:
        return

    rows = dpg.get_item_children(sender, 1)

    sort_col_id = sort_specs[0][0]

    cols: list[int] = dpg.get_item_children(sender)[0]  # type: ignore

    sort_column = sort_col_id - cols[0]

    sortable_list = []
    if rows is None:
        return

    for row in rows:
        cells: list[int] = dpg.get_item_children(row, 1)  # type: ignore
        first_cell = cells[sort_column]
        sortable_list.append([row, dpg.get_item_label(first_cell)])

    def _sorter(e):
        return e[1]

    sortable_list.sort(key=_sorter, reverse=sort_specs[0][1] < 0)

    # create list of just sorted row ids
    new_order = []
    for pair in sortable_list:
        new_order.append(pair[0])

    dpg.reorder_items(sender, 1, new_order)


dpg.create_context()
dpg.create_viewport(title="xrf_splitter", width=1920, height=1080)


def on_key_lq():
    if dpg.is_key_down(dpg.mvKey_Q):
        dpg.stop_dearpygui()


with dpg.handler_registry():
    dpg.add_key_press_handler(dpg.mvKey_Control, callback=on_key_lq)


def file_dialog_callback(_, app_data):
    return app_data["file_path_name"]


def file_dialog_cancel_callback():
    return None


with dpg.file_dialog(
    directory_selector=False,
    show=False,
    callback=file_dialog_callback,
    tag="pdz_dialog",
    cancel_callback=file_dialog_callback,
    width=700,
    height=400,
):
    dpg.add_file_extension(".pdz")


with dpg.file_dialog(
    directory_selector=False,
    show=False,
    callback=file_dialog_callback,
    tag="csv_dialog",
    cancel_callback=file_dialog_callback,
    width=700,
    height=400,
):
    dpg.add_file_extension(".csv")

with dpg.viewport_menu_bar():
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
            label="Show Documentation", callback=lambda: dpg.show_tool(dpg.mvTool_Doc)
        )
        dpg.add_menu_item(
            label="Show Debug", callback=lambda: dpg.show_tool(dpg.mvTool_Debug)
        )
        dpg.add_menu_item(
            label="Show Style Editor", callback=lambda: dpg.show_tool(dpg.mvTool_Style)
        )
        dpg.add_menu_item(
            label="Show Font Manager", callback=lambda: dpg.show_tool(dpg.mvTool_Font)
        )
        dpg.add_menu_item(
            label="Show Item Registry",
            callback=lambda: dpg.show_tool(dpg.mvTool_ItemRegistry),
        )

with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvLineSeries):
        dpg.add_theme_style(
            dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
        )

with dpg.window(label="Example Window", tag="Primary"):
    dpg.add_button(
        label="Select spectra",
        callback=lambda: dpg.show_item("pdz_dialog"),
    )
    dpg.add_button(
        label="Select results table",
        callback=lambda: dpg.show_item("csv_dialog"),
    )

    with dpg.plot(
        label="Plots", height=600, width=800, crosshairs=True, anti_aliased=True
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
            [
                pdz.spectra[1].counts[i] + pdz.spectra[2].counts[i]
                for i, _ in enumerate(pdz.spectra[1].counts)
            ],
            label=f"[1+2] {pdz.name}",
            parent="y_axis",
        )

    with dpg.table(
        label="Results",
        header_row=True,
        resizable=False,
        clipper=False,
        scrollX=True,
        sortable=True,
        callback=sort_callback,
        policy=dpg.mvTable_SizingFixedFit,
        borders_outerH=True,
        borders_innerV=True,
        borders_innerH=True,
        borders_outerV=True,
        reorderable=True,
        precise_widths=True,
    ):
        for i in range(df.shape[1]):
            dpg.add_table_column(label=df.columns[i], tag=f"{df.columns[i]}")
        for i in range(df.shape[0]):
            with dpg.table_row():
                for j in range(df.shape[1]):
                    dpg.add_selectable(label=f"{arr[i,j]}", span_columns=True)

dpg.bind_theme(global_theme)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary", True)
dpg.start_dearpygui()
dpg.destroy_context()
