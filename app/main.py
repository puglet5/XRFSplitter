import logging
import os
import re
import uuid

import coloredlogs
import dearpygui.dearpygui as dpg
import numpy as np

from app.utils import *

logging.basicConfig(filename="./log/main.log", filemode="a")
logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")

dpg.create_context()
dpg.create_viewport(title="xrf_splitter", width=1920, height=1080, vsync=True)
dpg.configure_app(
    wait_for_input=False
    # docking=True,
    # docking_space=False,
    # init_file="./dgp.ini"
)

TABLE = "results_table"
WINDOW = "primary_window"
global pca_plots_last_visible
pca_plots_last_visible: int = 0

global last_row_selected
last_row_selected: int | str = 0


def save_init():
    dpg.save_init_file("./dpg.ini")


@log_exec_time
def sort_callback(sender: int | str, sort_specs: None | list[list[int]]):
    if sort_specs is None:
        return

    sort_col_label: str | None = dpg.get_item_label(sort_specs[0][0])

    if sort_col_label is None:
        return

    reverse = sort_specs[0][1] < 0
    current = table_data.current.copy()
    sorted = table_data.sort(sort_col_label, reverse)

    if current.equals(sorted):
        logger.info(f"Table already sorted: {table_data.sorted_by}")
        return

    table_data.sorted_by = (sort_col_label, reverse)

    populate_table()


def on_key_ctrl():
    if dpg.is_key_pressed(dpg.mvKey_Q):
        dpg.stop_dearpygui()
    if dpg.is_key_pressed(dpg.mvKey_C):
        if table_data.selections:
            table_data.selected_to_clipboard()
    if dpg.is_key_down(dpg.mvKey_Shift):
        if dpg.is_key_pressed(dpg.mvKey_A):
            select_all_rows()
        if dpg.is_key_pressed(dpg.mvKey_D):
            deselect_all_rows()
    if dpg.is_key_down(dpg.mvKey_Alt):
        if dpg.is_key_pressed(dpg.mvKey_M):
            menubar_visible = dpg.get_item_configuration(WINDOW)["menubar"]
            dpg.configure_item(WINDOW, menubar=(not menubar_visible))


def enable_table_controls():
    dpg.enable_item("err_col_checkbox")
    dpg.show_item("err_col_checkbox")
    dpg.enable_item("junk_col_checkbox")
    dpg.show_item("junk_col_checkbox")
    dpg.enable_item("table_highlight_checkbox")
    dpg.show_item("table_highlight_checkbox")
    dpg.enable_item("lod_checkbox")
    dpg.show_item("lod_checkbox")
    dpg.enable_item("empty_cols_checkbox")
    dpg.show_item("empty_cols_checkbox")
    dpg.enable_item("empty_rows_checkbox")
    dpg.show_item("empty_rows_checkbox")
    dpg.set_value("err_col_checkbox", False)
    dpg.set_value("junk_col_checkbox", False)
    dpg.set_value("lod_checkbox", True)
    dpg.set_value("empty_cols_checkbox", True)
    dpg.set_value("empty_rows_checkbox", True)


@log_exec_time
@progress_bar
def select_all_rows():
    rows: list[int] = dpg.get_item_children(TABLE, 1)  # type:ignore
    rows_n = len(rows)

    for i, r in enumerate(rows):
        yield (i / rows_n)
        if cells := dpg.get_item_children(r, 1):
            cell = cells[0]
            if not dpg.get_value(cell):
                dpg.set_value(cell, True)
                row_select_callback(cell, True, check_keys=False, propagate=False)


def pca_plots_is_visible():
    return dpg.get_frame_count() - pca_plots_last_visible < 10


def ctrl_select_rows(row_clicked):
    rows: list[int] = dpg.get_item_children(TABLE, 1)  # type:ignore
    if row_clicked not in rows or last_row_selected not in rows:
        return

    row_clicked_i = rows.index(row_clicked)
    last_row_i = rows.index(last_row_selected)

    if last_row_i == row_clicked_i:
        return

    if last_row_i > row_clicked_i:
        select_from, select_to = row_clicked_i, last_row_i
    else:
        select_from, select_to = last_row_i, row_clicked_i

    for r in rows[select_from:select_to]:
        cells: list[int] | None = dpg.get_item_children(r, 1)  # type:ignore
        if cells is None:
            return

        cell = cells[0]
        if not dpg.get_value(cell):
            dpg.set_value(cell, True)
            row_select_callback(cell, True, False, False)


@log_exec_time
def row_select_callback(cell: int, value: bool, check_keys=True, propagate=True):
    global last_row_selected
    if propagate:
        if check_keys:
            if not dpg.is_key_down(dpg.mvKey_Shift):
                deselect_all_rows()
            if dpg.is_key_down(dpg.mvKey_Control):
                ctrl_select_rows(dpg.get_item_parent(cell))

        last_row_selected = dpg.get_item_parent(cell) or 0

    if (spectrum_label := dpg.get_item_label(cell)) is None:
        return

    selections = table_data.selections

    with dpg.mutex():
        if value:
            files = [
                filename
                for filename in os.listdir(plot_data.pdz_folder)
                if filename.endswith(".pdz")
            ]

            patt = re.compile(f"^([0]+){spectrum_label}-")
            files_re = [s for s in files if patt.match(s)]
            file = files_re[0] if files else None
            selections[spectrum_label] = cell
            if file is not None:
                path = Path(f"{plot_data.pdz_folder}/{file}")
                add_pdz_plot(path, spectrum_label)
                dpg.bind_item_handler_registry(cell, "row_hover_handler")
                if len(plot_data.pdz_data) > 3:
                    if pca_plots_is_visible():
                        update_pca_plot()
        else:
            selections.pop(spectrum_label, None)
            remove_pdz_plot(spectrum_label)
            if len(plot_data.pdz_data) > 3:
                if pca_plots_is_visible():
                    update_pca_plot()
            else:
                remove_pca_plot()


def prepare_data():
    table_data.toggle_columns(RESULTS_JUNK, dpg.get_value("junk_col_checkbox"))
    table_data.toggle_columns(RESULT_KEYS_ERR, dpg.get_value("err_col_checkbox"))
    table_data.selected_rows_range = (dpg.get_value("from"), dpg.get_value("to"))
    table_data.select_rows_range(
        show_empty_cols=bool(dpg.get_value("empty_cols_checkbox")),
        show_empty_rows=bool(dpg.get_value("empty_rows_checkbox")),
    )
    table_data.toggle_lod(dpg.get_value("lod_checkbox"))


@log_exec_time
@progress_bar
def populate_table():
    global last_row_selected

    prepare_data()

    with dpg.mutex():
        if dpg.get_value("table_highlight_checkbox"):
            unhighlight_table()
        try:
            dpg.delete_item(TABLE, children_only=True)
        except Exception:
            logger.warn(f"No table found: {TABLE}")

        arr = table_data.current.to_numpy()
        cols = table_data.current.columns.tolist()

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
        yield (i / row_n)
        dpg.lock_mutex()
        row_id = uuid.uuid4().int & (1 << 64) - 1
        with dpg.table_row(use_internal_label=False, parent=TABLE, tag=row_id):
            label = arr[i, 0]
            if previously_selected := label in selections:
                last_row_selected = row_id
            dpg.add_selectable(
                label=label,
                span_columns=True,
                default_value=previously_selected,
                callback=lambda s, a: row_select_callback(s, a),
                tag=selections.get(arr[i, 0], 0),
            )

            for j in range(1, arr.shape[1]):
                dpg.add_selectable(
                    label=arr[i, j],
                    disable_popup_close=True,
                    use_internal_label=False,
                )
        dpg.unlock_mutex()

    if dpg.get_value("table_highlight_checkbox"):
        highlight_table()


@log_exec_time
@progress_bar
def deselect_all_rows():
    cells = list(table_data.selections.values())
    cells_n = len(cells)

    for i, cell in enumerate(cells):
        yield (i / cells_n)
        if dpg.does_item_exist(cell):
            dpg.set_value(cell, False)
            row_select_callback(cell, False, check_keys=False, propagate=False)


def remove_pca_plot():
    dpg.delete_item("pca_y_axis", children_only=True)
    dpg.delete_item("pca_plots", children_only=True, slot=0)
    dpg.configure_item("pca_x_axis", label="PC1")
    dpg.configure_item("pca_y_axis", label="PC2")


def update_pca_plot():
    plot_data.generate_pca_data()
    dpg.delete_item("pca_y_axis", children_only=True)
    dpg.delete_item("pca_plots", children_only=True, slot=0)

    if (
        plot_data.pca_shapes is None
        or plot_data.pca_info is None
        or plot_data.pca_data is None
    ):
        return

    for i in plot_data.pca_shapes:
        for j in i:
            x, y = j.T.tolist()
            if x and y:
                dpg.add_area_series(
                    x,
                    y,
                    parent="pca_y_axis",
                    fill=(30, 120, 200, 20),
                )

    x, y = plot_data.pca_data.T.tolist()
    dpg.add_scatter_series(
        x,
        y,
        parent="pca_y_axis",
    )

    for x, y, label in zip(x, y, plot_data.pdz_data.keys()):
        dpg.add_plot_annotation(
            tag=f"ann_{label}",
            label=label,
            clamped=False,
            default_value=(x, y),
            offset=(-1, 1),
            color=[255, 255, 255, 100],
            parent="pca_plots",
        )

    variance = plot_data.pca_info.explained_variance_ratio_
    sum = variance.sum()

    dpg.configure_item("pca_x_axis", label=f"PC1 ({variance[0]/sum*100:,.2f}%)")
    dpg.configure_item("pca_y_axis", label=f"PC2 ({variance[1]/sum*100:,.2f}%)")

    # dpg.fit_axis_data("pca_x_axis")
    # dpg.fit_axis_data("pca_y_axis")


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

    enable_table_controls()
    populate_table()

    dpg.configure_item(TABLE, sortable=True)


def csv_file_dialog_callback(_, app_data: dict):
    selections = app_data.get("selections", None)
    if selections is None:
        return
    selection = list(selections.values())[0]
    setup_table(Path(selection))


def toggle_highlight_table():
    if dpg.get_value("table_highlight_checkbox"):
        unhighlight_table()
        highlight_table()
    else:
        unhighlight_table()


def unhighlight_table():
    df = table_data.current
    df_n = df.shape[0]
    err = None
    for i in range(df_n):
        for j in range(df.shape[1]):
            try:
                dpg.unhighlight_table_cell(TABLE, i, j)
            except Exception as e:
                err = e
                break
    if err is not None:
        logger.warning(f"Couldn't properly unhighlight table: {TABLE}, {err}")


def row_hover_callback(_s, row):
    row_label = dpg.get_item_label(row)
    if row_label is None:
        return

    if not (annotations := dpg.get_item_children("pca_plots", 0)):
        return

    if len(annotations) < 4:
        return

    for a in annotations:
        dpg.configure_item(a, color=[255, 255, 255, 100])

    if not dpg.does_item_exist(f"ann_{row_label}"):
        return

    dpg.configure_item(f"ann_{row_label}", color=(0, 119, 200, 255))


def collapsible_clicked_callback(s, a):
    plots_visible: bool = dpg.get_item_state("plots_wrapper")[
        "clicked"
    ] != dpg.is_item_visible("plots_tabs")
    table_visible = dpg.is_item_visible(TABLE)

    vp_height = dpg.get_viewport_height()

    if plots_visible and not table_visible:
        dpg.configure_item("pca_plots", height=-50)
        dpg.configure_item("pdz_plots", height=-50)

    if plots_visible and table_visible:
        dpg.configure_item("pca_plots", height=vp_height // 2)
        dpg.configure_item("pdz_plots", height=vp_height // 2)
        dpg.configure_item(TABLE, height=-1)

    if not plots_visible and table_visible:
        dpg.configure_item(TABLE, height=-1)


def window_resize_callback(s, a):
    plots_visible = dpg.is_item_visible("plots_tabs")
    table_visible = dpg.is_item_visible(TABLE)

    vp_height = dpg.get_viewport_height()

    if plots_visible and not table_visible:
        dpg.configure_item("pca_plots", height=-50)
        dpg.configure_item("pdz_plots", height=-50)

    if plots_visible and table_visible:
        dpg.configure_item("pca_plots", height=vp_height // 2)
        dpg.configure_item("pdz_plots", height=vp_height // 2)
        dpg.configure_item(TABLE, height=-1)

    if not plots_visible and table_visible:
        dpg.configure_item(TABLE, height=-1)


def pca_plots_visible_callback(s, a):
    global pca_plots_last_visible
    if not pca_plots_is_visible():
        try:
            update_pca_plot()
        except:
            logger.warning("Couldn't update PCA plots")
        finally:
            pca_plots_last_visible = dpg.get_frame_count()

    pca_plots_last_visible = dpg.get_frame_count()


with dpg.item_handler_registry(tag="row_hover_handler"):
    dpg.add_item_hover_handler(callback=row_hover_callback)

with dpg.item_handler_registry(tag="collapsible_clicked_handler"):
    dpg.add_item_clicked_handler(callback=collapsible_clicked_callback)

with dpg.item_handler_registry(tag="window_resize_handler"):
    dpg.add_item_resize_handler(callback=window_resize_callback)

with dpg.item_handler_registry(tag="pca_plots_visible_handler"):
    dpg.add_item_visible_handler(callback=pca_plots_visible_callback)


def highlight_table():
    df = table_data.current
    col_ids = [df.columns.get_loc(c) for c in RESULT_ELEMENTS if c in df]
    arr = df.iloc[:, col_ids].replace(["< LOD", ""], 0).to_numpy().astype(float)
    for row_i, row in enumerate(arr):
        t = np.nan_to_num(row / np.max(row), nan=0.0)
        t = np.log(t + 0.01)
        t = np.interp(t, (t.min(), t.max()), (0.0, 1.0))
        for val, column in zip(t, col_ids):
            sample = dpg.sample_colormap(dpg.mvPlotColormap_Jet, val)
            norm = [255.0, 255.0, 255.0, min(val * 100.0 + 20.0, 100.0)]
            color = [int(sample[i] * norm[i]) for i in range(len(sample))]
            dpg.highlight_table_cell(TABLE, row_i, column, color)


def add_pdz_plot(path: Path, label: str):
    if dpg.does_item_exist(f"{label}_plot"):
        return

    try:
        pdz = PDZFile(path)
    except:
        logger.error(f"Error decoding .pdz file: {path}")
        return

    plot_data.pdz_data[label] = pdz
    x, y = pdz.plot_data

    dpg.add_line_series(
        x,
        y,
        tag=f"{label}_plot",
        label=f"{pdz.spectra_used} {pdz.name}",
        parent="y_axis",
    )

    dpg.fit_axis_data("x_axis")
    dpg.fit_axis_data("y_axis")


def remove_pdz_plot(label: str):
    plot_data.pdz_data.pop(label, None)
    plot_label = f"{label}_plot"
    if dpg.does_item_exist(plot_label):
        dpg.delete_item(plot_label)


def pdz_file_dialog_callback(_, app_data: dict[str, str]):
    global plot_data
    plot_data = PlotData(app_data.get("file_path_name", ""))


def setup_dev():
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

with dpg.theme() as pca_theme:
    with dpg.theme_component(dpg.mvAreaSeries):
        dpg.add_theme_color(
            dpg.mvPlotCol_Line, (255, 255, 255, 100), category=dpg.mvThemeCat_Plots
        )

    with dpg.theme_component(dpg.mvScatterSeries):
        dpg.add_theme_color(
            dpg.mvPlotCol_MarkerFill,
            (255, 255, 255, 255),
            category=dpg.mvThemeCat_Plots,
        )
        dpg.add_theme_color(
            dpg.mvPlotCol_MarkerOutline,
            (255, 255, 255, 255),
            category=dpg.mvThemeCat_Plots,
        )


with dpg.window(
    label="xrfsplitter", tag=WINDOW, horizontal_scrollbar=False, no_scrollbar=True
):
    with dpg.menu_bar(tag="menu_bar"):
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save As")

            with dpg.menu(label="Settings"):
                dpg.add_menu_item(label="Setting 1", check=True)
                dpg.add_menu_item(label="Setting 2")

        with dpg.menu(label="Window"):
            dpg.add_menu_item(
                label="Wait For Input",
                check=True,
                tag="wait_for_input_menu",
                callback=lambda s, a: dpg.configure_app(wait_for_input=a),
            )
            dpg.add_menu_item(
                label="Toggle Fullscreen",
                callback=lambda: dpg.toggle_viewport_fullscreen(),
            )
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
                        callback=populate_table,
                        enabled=False,
                        show=False,
                    )

                    dpg.add_checkbox(
                        label="Show junk columns",
                        default_value=True,
                        tag="junk_col_checkbox",
                        callback=populate_table,
                        enabled=False,
                        show=False,
                    )

                    dpg.add_checkbox(
                        label="Show '< LOD'",
                        default_value=True,
                        tag="lod_checkbox",
                        callback=populate_table,
                        enabled=False,
                        show=False,
                    )

                    dpg.add_checkbox(
                        label="Show empty columns",
                        default_value=True,
                        tag="empty_cols_checkbox",
                        callback=populate_table,
                        enabled=False,
                        show=False,
                    )

                    dpg.add_checkbox(
                        label="Show empty rows",
                        default_value=True,
                        tag="empty_rows_checkbox",
                        callback=populate_table,
                        enabled=False,
                        show=False,
                    )

                    dpg.add_checkbox(
                        label="Highlight table",
                        tag="table_highlight_checkbox",
                        default_value=True,
                        callback=toggle_highlight_table,
                        show=False,
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
                            callback=populate_table,
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
                            callback=populate_table,
                        )
                        dpg.add_text(
                            "(?)", tag="range_tooltip", color=(200, 200, 200, 100)
                        )

                        with dpg.tooltip("range_tooltip"):
                            dpg.add_text("-1 indicates no upper limit")

                    dpg.add_button(
                        label="Select all",
                        callback=select_all_rows,
                    )

                    dpg.add_button(
                        label="Deselect all",
                        callback=deselect_all_rows,
                    )

        with dpg.child_window(border=False, width=-1, tag="data"):
            with dpg.collapsing_header(
                label="Plots", default_open=False, tag="plots_wrapper"
            ):
                with dpg.tab_bar(tag="plots_tabs"):
                    with dpg.tab(label="PDZ"):
                        with dpg.plot(
                            tag="pdz_plots",
                            crosshairs=True,
                            anti_aliased=True,
                            height=-50,
                            width=-1,
                        ):
                            dpg.add_plot_legend(location=9)
                            dpg.add_plot_axis(
                                dpg.mvXAxis, label="Energy, kEv", tag="x_axis"
                            )
                            dpg.add_plot_axis(dpg.mvYAxis, label="Counts", tag="y_axis")
                    with dpg.tab(label="PCA"):
                        with dpg.plot(
                            tag="pca_plots",
                            crosshairs=True,
                            anti_aliased=True,
                            height=-50,
                            width=-1,
                            equal_aspects=True,
                        ):
                            dpg.add_plot_legend(location=9)
                            dpg.add_plot_axis(
                                dpg.mvXAxis, label="PC2", tag="pca_x_axis"
                            )
                            dpg.add_plot_axis(
                                dpg.mvYAxis, label="PC1", tag="pca_y_axis"
                            )

                # dpg.add_spacer(height=4)
                # dpg.add_separator()
                # dpg.add_spacer(height=4)

            with dpg.group(tag="table_wrapper"):
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
                    dpg.add_table_column(label=ID_COL)

dpg.bind_theme(global_theme)
dpg.bind_item_theme("pca_plots", pca_theme)
dpg.bind_item_handler_registry("table_wrapper", "collapsible_clicked_handler")
dpg.bind_item_handler_registry("plots_wrapper", "collapsible_clicked_handler")
dpg.bind_item_handler_registry(WINDOW, "window_resize_handler")
dpg.bind_item_handler_registry("pca_plots", "pca_plots_visible_handler")

dpg.set_frame_callback(3, setup_dev)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_viewport_vsync(True)
dpg.set_primary_window(WINDOW, True)
dpg.start_dearpygui()
dpg.destroy_context()
