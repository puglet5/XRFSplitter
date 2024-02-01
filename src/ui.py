from dataclasses import dataclass, field
import gc
import os
from pathlib import Path
import re
from typing import Literal
import uuid

import numpy as np
import dearpygui.dearpygui as dpg

from src.utils import COLUMN_PRESETS, ID_COL, LABEL_PAD, RESULT_ELEMENTS, RESULTS_INFO, TABLE_SIZE_APPROXIMATION_FACTOR_KB, PDZFile, PlotData, TableData, log_exec_time, logger, progress_bar



TOOLTIP_DELAY_SEC = 0.1
EMPTY_CELL_COLOR = [0.0, 0.0, 170.0000050663948, 20.0]


@dataclass
class UI:
    table_data: TableData = field(init=False)
    plot_data: PlotData = field(init=False)
    row_range: tuple[int, int] = field(init=False)
    table_tag: str = field(init=False, default="results_table")
    window_tag: str = field(init=False, default="primary_window")
    pca_plot_last_frame_visible: int = field(init=False, default=0)
    pdz_plot_last_frame_visible: int = field(init=False, default=0)
    last_row_selected: int = field(init=False, default=0)
    last_idle_frame: int = field(init=False, default=0)

    def __post_init__(self):
        dpg.create_context()
        dpg.create_viewport(title="xrf_splitter", width=1920, height=1080, vsync=True)
        dpg.configure_app(wait_for_input=False)
        self.setup_themes()
        self.setup_handler_registries()
        self.setup_file_dialogs()
        self.setup_layout()
        self.bind_themes()
        self.bind_item_handlers()

    def setup_themes(self):
        with dpg.theme() as self.global_theme:
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

            with dpg.theme_component(dpg.mvCheckbox, enabled_state=False):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, [120, 120, 120, 0])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, [120, 120, 120, 0])
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, [120, 120, 120])

        with dpg.theme() as self.pca_theme:
            with dpg.theme_component(dpg.mvAreaSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (255, 255, 255, 100),
                    category=dpg.mvThemeCat_Plots,
                )

            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_Marker,
                    dpg.mvPlotMarker_Cross,
                    category=dpg.mvThemeCat_Plots,
                )
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

    def save_state_init(self, path: str):
        dpg.save_init_file(path)

    def load_state_json(self):
        ...

    def save_state_json(self):
        ...

    def setup_handler_registries(self):
        with dpg.handler_registry():
            dpg.add_key_down_handler(dpg.mvKey_Control, callback=self.on_key_ctrl)
            dpg.add_key_down_handler(dpg.mvKey_Escape, callback=self.hide_modals)
            # dpg.add_mouse_move_handler(callback=mouse_move_callback)

        with dpg.item_handler_registry(tag="row_hover_handler"):
            dpg.add_item_hover_handler(callback=self.row_hover_callback)

        with dpg.item_handler_registry(tag="collapsible_clicked_handler"):
            dpg.add_item_clicked_handler(callback=self.collapsible_clicked_callback)

        with dpg.item_handler_registry(tag="window_resize_handler"):
            dpg.add_item_resize_handler(callback=self.window_resize_callback)

        with dpg.item_handler_registry(tag="pca_plot_visible_handler"):
            dpg.add_item_visible_handler(callback=self.pca_plot_visible_callback)

        with dpg.item_handler_registry(tag="row_threshold_slider_handler"):
            dpg.add_item_deactivated_after_edit_handler(callback=self.populate_table)

    def hide_modals(self):
        if dpg.is_item_visible("settings_modal"):
            dpg.hide_item("settings_modal")

    def bind_item_handlers(self):
        dpg.bind_theme(self.global_theme)
        dpg.bind_item_theme("pca_plot", self.pca_theme)
        dpg.bind_item_handler_registry("table_wrapper", "collapsible_clicked_handler")
        dpg.bind_item_handler_registry("plots_wrapper", "collapsible_clicked_handler")
        dpg.bind_item_handler_registry(self.window_tag, "window_resize_handler")
        dpg.bind_item_handler_registry("pca_plot", "pca_plot_visible_handler")
        dpg.bind_item_handler_registry(
            "row_threshold_slider", "row_threshold_slider_handler"
        )

    def bind_themes(self):
        dpg.bind_theme(self.global_theme)
        dpg.bind_item_theme("pca_plot", self.pca_theme)

    def start(self, dev=False):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_vsync(True)
        dpg.set_primary_window(self.window_tag, True)
        try:
            if dev:
                dpg.set_frame_callback(1, self.setup_dev)
            dpg.start_dearpygui()
        except Exception as e:
            logger.fatal(e)
        finally:
            self.stop()

    def stop(self):
        dpg.stop_dearpygui()
        dpg.destroy_context()

    def setup_table(self, csv_path: Path):
        if dpg.does_item_exist(self.table_tag):
            dpg.delete_item(self.table_tag)

        self.table_data = TableData(csv_path)

        dpg.add_table(
            label="Results",
            parent="table_wrapper",
            tag=self.table_tag,
            header_row=True,
            hideable=False,
            resizable=False,
            clipper=True,
            freeze_columns=1,
            freeze_rows=1,
            scrollX=True,
            scrollY=True,
            sortable=False,
            callback=lambda s, a: self.sort_callback(s, a),
            delay_search=True,
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

        self.enable_table_controls()
        self.populate_table()

        dpg.configure_item(self.table_tag, sortable=True)

    def setup_dev(self):
        self.pdz_file_dialog_callback(
            "",
            {"file_path_name": "/home/puglet5/Documents/PROJ/test_data/smalts pdz"},
        )
        self.csv_file_dialog_callback(
            "",
            {
                "selections": {
                    "1": "/home/puglet5/Documents/PROJ/XRFSplitter/sandbox/Results.csv"
                }
            },
        )

    def csv_file_dialog_callback(self, _sender, app_data: dict):
        selections = app_data.get("selections", None)
        if selections is None:
            return
        file_selection = list(selections.values())[0]
        self.setup_table(Path(file_selection))

        dpg.configure_item(
            "from",
            min_value=self.table_data.first_row,
            max_value=self.table_data.last_row,
        )
        dpg.configure_item(
            "to",
            max_value=self.table_data.last_row,
        )

    def pdz_file_dialog_callback(self, _sender, app_data: dict[str, str]):
        self.plot_data = PlotData(app_data.get("file_path_name", ""))

    def setup_file_dialogs(self):
        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=self.pdz_file_dialog_callback,
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
            callback=self.csv_file_dialog_callback,
            tag="csv_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".csv")
            dpg.add_file_extension("*")

    def highlight_table(self):
        df = self.table_data.current
        col_ids = sorted(df.columns.get_loc(c) for c in RESULT_ELEMENTS if c in df)
        arr = df.iloc[:, col_ids].replace(["< LOD", ""], 0).to_numpy().astype(float)
        for row_i, row in enumerate(arr):
            t = np.nan_to_num(row / np.max(row), nan=0)
            t = np.log(t + 0.01)
            t = np.interp(t, (t.min(), t.max()), (0, 1))
            for val, column in zip(t, col_ids):
                sample = dpg.sample_colormap(dpg.mvPlotColormap_Jet, val)
                color = [
                    a * b
                    for a, b in zip(sample, [255, 255, 255, min(val * 100 + 20, 100)])
                ]

                if color == EMPTY_CELL_COLOR:
                    continue

                dpg.highlight_table_cell(self.table_tag, row_i, column, color)

    def add_pdz_plot(self, path: Path, label: str):
        if dpg.does_item_exist(f"{label}_plot"):
            return

        try:
            pdz = PDZFile(path)
        except:
            logger.error(f"Error decoding .pdz file: {path}")
            return

        self.plot_data.pdz_data[label] = pdz
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

    def remove_pdz_plot(self, label: str):
        self.plot_data.pdz_data.pop(label, None)
        plot_label = f"{label}_plot"
        if dpg.does_item_exist(plot_label):
            dpg.delete_item(plot_label)

    def row_hover_callback(self, _sender, row):
        row_label = dpg.get_item_label(row)
        if row_label is None:
            return

        if not (annotations := dpg.get_item_children("pca_plot", 0)):
            return

        if len(annotations) < 4:
            return

        for a in annotations:
            dpg.configure_item(a, color=[255, 255, 255, 100])

        if not dpg.does_item_exist(f"ann_{row_label}"):
            return

        dpg.configure_item(f"ann_{row_label}", color=(0, 119, 200, 255))

    def collapsible_clicked_callback(self, _sender, _data):
        plots_visible: bool = dpg.get_item_state("plots_wrapper")[
            "clicked"
        ] != dpg.is_item_visible("plots_tabs")
        table_visible = dpg.is_item_visible(self.table_tag)

        vp_height = dpg.get_viewport_height()

        if plots_visible and not table_visible:
            dpg.configure_item("pca_plot", height=-50)
            dpg.configure_item("pdz_plots", height=-50)

        if plots_visible and table_visible:
            dpg.configure_item("pca_plot", height=vp_height // 2)
            dpg.configure_item("pdz_plots", height=vp_height // 2)
            dpg.configure_item(self.table_tag, height=-1)

        if not plots_visible and table_visible:
            dpg.configure_item(self.table_tag, height=-1)

    def window_resize_callback(self, _sender, _data):
        plots_visible = dpg.is_item_visible("plots_tabs")
        table_visible = dpg.is_item_visible(self.table_tag)

        vp_height = dpg.get_viewport_height()

        if plots_visible and not table_visible:
            dpg.configure_item("pca_plot", height=-50)
            dpg.configure_item("pdz_plots", height=-50)

        if plots_visible and table_visible:
            dpg.configure_item("pca_plot", height=vp_height // 2)
            dpg.configure_item("pdz_plots", height=vp_height // 2)
            dpg.configure_item(self.table_tag, height=-1)

        if not plots_visible and table_visible:
            dpg.configure_item(self.table_tag, height=-1)

        if dpg.is_item_visible("settings_modal"):
            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            dpg.configure_item("settings_modal", pos=[w // 2 - 350, h // 2 - 200])

    def pca_plot_visible_callback(self, _sender, _data):
        if not self.pca_plot_is_visible():
            try:
                self.update_pca_plot()
            except Exception as e:
                logger.warning(f"Couldn't update PCA plots: {e}")
            finally:
                self.pca_plot_last_frame_visible = dpg.get_frame_count()

        self.pca_plot_last_frame_visible = dpg.get_frame_count()

    def toggle_highlight_table(self):
        if dpg.get_value("table_highlight_checkbox"):
            self.unhighlight_table()
            self.highlight_table()
        else:
            self.unhighlight_table()

    def unhighlight_table(self):
        table_children: dict | None = dpg.get_item_children(self.table_tag)  # type: ignore
        if table_children is None:
            return
        cols: list[int] = table_children.get(0, [])
        rows: list[int] = table_children.get(1, [])
        err = None
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                try:
                    dpg.unhighlight_table_cell(self.table_tag, i, j)
                except Exception as e:
                    err = e
                    break
        if err is not None:
            logger.warning(
                f"Couldn't properly unhighlight table: {self.table_tag}, {err}"
            )

    def sort_callback(self, _sender: int | str, sort_specs: None | list[list[int]]):
        if sort_specs is None:
            return

        sort_col = sort_specs[0][0]

        if not dpg.does_item_exist(sort_col):
            return

        sort_col_label: str | None = dpg.get_item_label(sort_col)

        if sort_col_label is None:
            return

        reverse = sort_specs[0][1] < 0
        current = self.table_data.current
        sorted = self.table_data.sort(sort_col_label, reverse)

        if current[ID_COL].equals(sorted[ID_COL]):
            logger.info(f"Table already sorted: {self.table_data.sorted_by}")
            return

        self.table_data.sorted_by = (sort_col_label, reverse)

        self.populate_table(skip_equals_check=True)

    def cycle_table_presets(self, direction: Literal["left", "right"]):
        current_preset = dpg.get_value("column_preset_combo")
        presets = list(COLUMN_PRESETS.keys())
        current_preset_index = presets.index(current_preset)
        if direction == "left":
            if current_preset_index < len(presets) - 1:
                dpg.set_value("column_preset_combo", presets[current_preset_index + 1])
            else:
                dpg.set_value("column_preset_combo", presets[0])
        if direction == "right":
            if current_preset_index > 0:
                dpg.set_value("column_preset_combo", presets[current_preset_index - 1])
            else:
                dpg.set_value("column_preset_combo", presets[-1])
        self.populate_table()

    def on_key_ctrl(self):
        if dpg.is_key_pressed(dpg.mvKey_Right):
            self.cycle_table_presets("right")
        if dpg.is_key_pressed(dpg.mvKey_Left):
            self.cycle_table_presets("left")
        if dpg.is_key_pressed(dpg.mvKey_Q):
            dpg.stop_dearpygui()
            dpg.destroy_context()
        if dpg.is_key_pressed(dpg.mvKey_C):
            if self.table_data.selections:
                self.table_data.selected_to_clipboard()
        if dpg.is_key_pressed(dpg.mvKey_Comma):
            if not dpg.is_item_visible("settings_modal"):
                self.show_settings_modal()
            else:
                dpg.hide_item("settings_modal")
        if dpg.is_key_down(dpg.mvKey_Shift):
            if dpg.is_key_pressed(dpg.mvKey_A):
                self.select_all_rows()
            if dpg.is_key_pressed(dpg.mvKey_D):
                self.deselect_all_rows()

        if dpg.is_key_down(dpg.mvKey_Alt):
            if dpg.is_key_down(dpg.mvKey_Shift):
                if dpg.is_key_pressed(dpg.mvKey_M):
                    dpg.show_tool(dpg.mvTool_Metrics)
            elif dpg.is_key_pressed(dpg.mvKey_M):
                menubar_visible = dpg.get_item_configuration(self.window_tag)["menubar"]
                dpg.configure_item(self.window_tag, menubar=(not menubar_visible))

    def enable_table_controls(self):
        dpg.enable_item("table_highlight_checkbox")
        dpg.show_item("table_highlight_checkbox")
        dpg.enable_item("lod_checkbox")
        dpg.show_item("lod_checkbox")
        dpg.set_value("lod_checkbox", False)

    @log_exec_time
    @progress_bar
    def select_all_rows(self):
        rows: list[int] = dpg.get_item_children(self.table_tag, 1)  # type:ignore
        rows_n = len(rows)

        for i, row in enumerate(rows):
            yield (i / rows_n)
            if dpg.is_key_pressed(dpg.mvKey_Escape):
                break
            if cell := dpg.get_item_user_data(row):
                if not dpg.get_value(cell):
                    dpg.set_value(cell, True)
                    self.row_select_callback(
                        cell, True, check_keys=False, propagate=False
                    )

    def pca_plot_is_visible(self):
        return dpg.get_frame_count() - self.pca_plot_last_frame_visible < 10

    @progress_bar
    def ctrl_select_rows(self, row_clicked: int):
        rows: list[int] = dpg.get_item_children(self.table_tag, 1)  # type:ignore
        if row_clicked not in rows or self.last_row_selected not in rows:
            return

        row_clicked_i = rows.index(row_clicked)
        last_row_i = rows.index(self.last_row_selected)

        if last_row_i == row_clicked_i:
            return

        if last_row_i > row_clicked_i:
            select_from, select_to = row_clicked_i, last_row_i
        else:
            select_from, select_to = last_row_i, row_clicked_i

        rows_to_select = rows[select_from:select_to]
        rows_to_select_n = len(rows_to_select)

        for i, r in enumerate(rows_to_select):
            yield (i / rows_to_select_n)
            if dpg.is_key_down(dpg.mvKey_Escape):
                break

            cell: int | None = dpg.get_item_user_data(r)
            if cell is None:
                return

            if not dpg.get_value(cell):
                dpg.set_value(cell, True)
                self.row_select_callback(cell, True, False, False)

    def row_select_callback(
        self,
        cell: int,
        value: bool,
        check_keys=True,
        propagate=True,
        update_plots: Literal["all", "pca", "pdz", "none"] = "all",
    ):
        if propagate:
            if check_keys:
                if not dpg.is_key_down(dpg.mvKey_Shift):
                    self.deselect_all_rows()
                if dpg.is_key_down(dpg.mvKey_Control):
                    row_clicked = dpg.get_item_parent(cell)
                    if row_clicked is not None:
                        self.ctrl_select_rows(row_clicked)

            self.last_row_selected = dpg.get_item_parent(cell) or 0

        if (spectrum_label := dpg.get_item_label(cell)) is None:
            return

        selections = self.table_data.selections
        dpg.bind_item_handler_registry(cell, 0)

        with dpg.mutex():
            if value:
                files = [
                    filename
                    for filename in os.listdir(self.plot_data.pdz_folder)
                    if filename.endswith(".pdz")
                ]

                patt = re.compile(f"^([0]+){spectrum_label}-")
                files_re = [s for s in files if patt.match(s)]
                file = files_re[0] if files_re else None
                selections[spectrum_label] = cell
                if file is not None:
                    path = Path(self.plot_data.pdz_folder, file)
                    if update_plots == "pdz" or update_plots == "all":
                        self.add_pdz_plot(path, spectrum_label)
                    if update_plots == "pca" or update_plots == "all":
                        if len(self.plot_data.pdz_data) > 3:
                            if self.pca_plot_is_visible():
                                self.update_pca_plot()
            else:
                selections.pop(spectrum_label, None)
                if update_plots == "pdz" or update_plots == "all":
                    self.remove_pdz_plot(spectrum_label)
                if update_plots == "pca" or update_plots == "all":
                    if len(self.plot_data.pdz_data) > 3:
                        if self.pca_plot_is_visible():
                            self.update_pca_plot()
                    else:
                        self.remove_pca_plot()

    @log_exec_time
    def prepare_data(self):
        range_from = dpg.get_value("from")
        range_to = dpg.get_value("to")
        if range_to < range_from and range_to != -1:
            dpg.set_value("to", range_from)
            range_to = range_from

        self.table_data.selected_rows_range = (range_from, range_to)
        self.table_data.select_rows_range()

        column_preset: str = dpg.get_value("column_preset_combo")
        for column, state in COLUMN_PRESETS[column_preset]:
            self.table_data.toggle_columns(column, state)

        self.table_data.toggle_lod(dpg.get_value("lod_checkbox"))

        row_preset: str = dpg.get_value("row_preset_combo")
        if row_preset == "All rows":
            ...
        if row_preset == "Non-empty rows":
            self.table_data.filter_empty_rows()
        if row_preset == "Valid rows":
            self.table_data.filter_empty_rows()
            if not column_preset == "Info":
                self.table_data.filter_invalid_rows(
                    dpg.get_value("row_threshold_slider")
                )
        if row_preset == "Valid rows, normalized":
            self.table_data.filter_empty_rows()
            if not column_preset == "Info":
                self.table_data.filter_invalid_rows(
                    dpg.get_value("row_threshold_slider")
                )
                self.table_data.normalize_rows()
        if column_preset == "Non-empty elements":
            self.table_data.toggle_columns("empty", False)

        if dpg.get_value("fill_with_zeros_checkbox"):
            self.table_data.fill_with_zeros()

    def clear_plots(self):
        self.plot_data.clear()
        dpg.delete_item("pca_y_axis", children_only=True)
        dpg.delete_item("pca_plot", children_only=True, slot=0)
        dpg.configure_item("pca_x_axis", label="PC1")
        dpg.configure_item("pca_y_axis", label="PC2")
        dpg.delete_item("y_axis", children_only=True, slot=1)

    @log_exec_time
    @progress_bar
    def populate_table(self, skip_equals_check=False):
        """
        Populates table `self.table_tag` with data from global `table_data`.

        Called on sort, row/column selection and '< LOD' toggle.

        Regenerates previously selected rows.

        Rehighlights table if `table_highlight_checkbox` is set.
        """

        if not skip_equals_check:
            pre = self.table_data.current.copy()
            self.prepare_data()
            if self.table_data.current.equals(pre):
                return

        with dpg.mutex():
            if dpg.get_value("table_highlight_checkbox"):
                self.unhighlight_table()
            try:
                dpg.delete_item(self.table_tag, children_only=True)
            except Exception:
                logger.warn(f"No table found: {self.table_tag}")

            arr = self.table_data.current.to_numpy()
            cols = self.table_data.current.columns.tolist()

            for col in cols:
                dpg.add_table_column(
                    label=col,
                    parent=self.table_tag,
                    prefer_sort_ascending=False,
                    prefer_sort_descending=True,
                )

            selections = self.table_data.selections

            row_n = arr.shape[0]
            for i in range(row_n):
                yield (i / row_n)
                row_id = uuid.uuid4().int & (1 << 64) - 1
                first_cell_label = arr[i, 0]
                is_previously_selected = first_cell_label in selections
                if is_previously_selected:
                    self.last_row_selected = row_id
                    first_cell_id = selections[first_cell_label]
                else:
                    first_cell_id = uuid.uuid4().int & (1 << 64) - 1

                with dpg.table_row(
                    parent=self.table_tag, tag=row_id, user_data=first_cell_id
                ):
                    dpg.add_selectable(
                        label=first_cell_label,
                        span_columns=True,
                        default_value=is_previously_selected,
                        callback=lambda s, a: self.row_select_callback(s, a),
                        tag=first_cell_id,
                    )
                    dpg.bind_item_handler_registry(dpg.last_item(), 0)

                    for j in range(1, arr.shape[1]):
                        dpg.add_selectable(label=arr[i, j])

            if dpg.get_value("column_preset_combo") == "Info":
                dpg.disable_item("table_highlight_checkbox")
                self.unhighlight_table()
            else:
                dpg.enable_item("table_highlight_checkbox")
                if dpg.get_value("table_highlight_checkbox"):
                    self.toggle_highlight_table()

            dpg.set_value(
                "table_dimensions",
                f"{self.table_data.current.shape}, apx. {self.table_data.current.memory_usage(index=True, deep=True).sum()/TABLE_SIZE_APPROXIMATION_FACTOR_KB:.0f} kB",
            )

    @progress_bar
    def deselect_all_rows(self):
        cells = list(self.table_data.selections.values())
        cells_n = len(cells)

        for i, cell in enumerate(cells):
            yield (i / cells_n)
            if dpg.does_item_exist(cell):
                dpg.set_value(cell, False)
        self.table_data.selections = {}
        self.clear_plots()
        gc.collect()

    def remove_pca_plot(self):
        if self.plot_data.pca_data is not None:
            dpg.delete_item("pca_y_axis", children_only=True)
            dpg.delete_item("pca_plot", children_only=True, slot=0)
            dpg.configure_item("pca_x_axis", label="PC1")
            dpg.configure_item("pca_y_axis", label="PC2")

    def update_pca_plot(self):
        self.plot_data.generate_pca_data()
        dpg.delete_item("pca_y_axis", children_only=True)
        dpg.delete_item("pca_plot", children_only=True, slot=0)

        if (
            self.plot_data.pca_shapes is None
            or self.plot_data.pca_info is None
            or self.plot_data.pca_data is None
        ):
            return

        for i in self.plot_data.pca_shapes:
            for j in i:
                x, y = j.T.tolist()
                if x and y:
                    dpg.add_area_series(
                        x,
                        y,
                        parent="pca_y_axis",
                        fill=(30, 120, 200, 20),
                    )

        x, y, *_ = self.plot_data.pca_data.T.tolist()
        dpg.add_scatter_series(
            x,
            y,
            parent="pca_y_axis",
        )

        for x, y, label in zip(x, y, self.plot_data.pdz_data.keys()):
            dpg.add_plot_annotation(
                tag=f"ann_{label}",
                label=label,
                clamped=False,
                default_value=(x, y),
                offset=(-1, 1),
                color=[255, 255, 255, 100],
                parent="pca_plot",
            )

        variance = self.plot_data.pca_info.explained_variance_ratio_
        sum = variance.sum()

        dpg.configure_item("pca_x_axis", label=f"PC1 ({variance[0]/sum*100:,.2f}%)")
        dpg.configure_item("pca_y_axis", label=f"PC2 ({variance[1]/sum*100:,.2f}%)")

        for row in self.table_data.selections.values():
            dpg.bind_item_handler_registry(row, "row_hover_handler")

        # dpg.fit_axis_data("pca_x_axis")
        # dpg.fit_axis_data("pca_y_axis")

    def setup_layout(self):
        with dpg.window(
            label="xrfsplitter",
            tag=self.window_tag,
            horizontal_scrollbar=False,
            no_scrollbar=True,
        ):
            with dpg.menu_bar(tag="menu_bar"):
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Save As", shortcut="(Ctrl+S)")
                    dpg.add_menu_item(label="Save As", shortcut="(Ctrl+Shift+S)")

                with dpg.menu(label="Edit"):
                    dpg.add_menu_item(label="To clipboard", shortcut="(Ctrl+C)")

                    dpg.add_menu_item(
                        label="Preferences",
                        shortcut="(Ctrl+,)",
                        callback=self.show_settings_modal,
                    )

                with dpg.menu(label="Window"):
                    dpg.add_menu_item(
                        label="Wait For Input",
                        check=True,
                        tag="wait_for_input_menu",
                        shortcut="(Ctrl+Shift+Alt+W)",
                        callback=lambda s, a: dpg.configure_app(wait_for_input=a),
                    )
                    dpg.add_menu_item(
                        label="Toggle Fullscreen",
                        shortcut="(Win+F)",
                        callback=lambda: dpg.toggle_viewport_fullscreen(),
                    )
                with dpg.menu(label="Tools"):
                    with dpg.menu(label="Developer"):
                        dpg.add_menu_item(
                            label="Show About",
                            callback=lambda: dpg.show_tool(dpg.mvTool_About),
                        )
                        dpg.add_menu_item(
                            label="Show Metrics",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Metrics),
                            shortcut="(Ctrl+Alt+M)",
                        )
                        dpg.add_menu_item(
                            label="Show Documentation",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Doc),
                        )
                        dpg.add_menu_item(
                            label="Show Debug",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Debug),
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
                with dpg.child_window(border=False, width=350, tag="sidebar"):
                    with dpg.child_window(width=-1, height=60, show=False):
                        with dpg.group(horizontal=True):
                            dpg.add_button(
                                label="Select spectra folder",
                                callback=lambda: dpg.show_item("pdz_dialog"),
                            )

                            dpg.add_button(
                                label="Select results table",
                                callback=lambda: dpg.show_item("csv_dialog"),
                            )
                    dpg.add_progress_bar(tag="table_progress", width=-1, height=19)
                    with dpg.tooltip("table_progress", delay=TOOLTIP_DELAY_SEC):
                        dpg.add_text("Current operation progress")

                    with dpg.child_window(width=-1, height=-1):
                        with dpg.group(tag="table_controls", horizontal=False):
                            with dpg.group(horizontal=True):
                                dpg.add_text("Show '< LOD'".rjust(LABEL_PAD))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Shows '< LOD' values in cells
                                            Default: off
                                        """,
                                        wrap=400,
                                    )
                                dpg.add_checkbox(
                                    default_value=False,
                                    tag="lod_checkbox",
                                    callback=self.populate_table,
                                    enabled=False,
                                    show=False,
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Row sum threshold".rjust(LABEL_PAD))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Sets threshold to filter valid rows.\nCtrl+LMB to edit directly
                                            Default: 70.0
                                        """,
                                        wrap=400,
                                    )
                                dpg.add_slider_float(
                                    width=-1,
                                    min_value=0,
                                    max_value=99,
                                    default_value=70,
                                    format="%.1f",
                                    clamped=True,
                                    tag="row_threshold_slider",
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Highlight table".rjust(LABEL_PAD))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Highlights table cells row-wise based on cell's value relative to row's total (normalized, blue to red from lowest to highest value)
                                            Default: on
                                        """,
                                        wrap=400,
                                    )
                                dpg.add_checkbox(
                                    tag="table_highlight_checkbox",
                                    default_value=True,
                                    callback=self.toggle_highlight_table,
                                    show=False,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Fill with zeros".rjust(LABEL_PAD))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Fills empty cells with zeros. Doesn't remove '< LOD' values
                                            Default: off
                                        """,
                                        wrap=400,
                                    )
                                dpg.add_checkbox(
                                    tag="fill_with_zeros_checkbox",
                                    default_value=False,
                                    callback=self.populate_table,
                                    show=True,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Column preset".rjust(LABEL_PAD))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Selects a set of columns to be shown:
                                                         - All elements: show column for every element found in the original table
                                                         - Non-empty elements: only show columns for elements with at least one value (not zero, empty or '< LOD') in selected rows
                                                         - Info: show columns with data acquisition information. Empty columns are always shown
                                                        
                                                        Columns with error estimates are not included in the table
                                                        
                                                            Default: Non-empty elements
                                                        """,
                                        wrap=400,
                                    )
                                dpg.add_combo(
                                    width=-1,
                                    items=[
                                        "All elements",
                                        "Non-empty elements",
                                        "Info",
                                    ],
                                    default_value="Non-empty elements",
                                    callback=self.populate_table,
                                    tag="column_preset_combo",
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Row preset".rjust(LABEL_PAD))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Selects a set of rows to be shown:
                                                         - All rows: show all rows, no filters
                                                         - Non-empty rows: only show rows with at least value in selected columns
                                                         - Valid rows: show rows, sums of which pass threshold. Max is 100
                                                         - Valid rows, normalized: show valid rows with values normalized to total
                                                        
                                                            Default: Valid rows
                                                        """,
                                        wrap=400,
                                    )
                                dpg.add_combo(
                                    width=-1,
                                    items=[
                                        "All rows",
                                        "Non-empty rows",
                                        "Valid rows",
                                        "Valid rows, normalized",
                                    ],
                                    default_value="Valid rows, normalized",
                                    callback=self.populate_table,
                                    tag="row_preset_combo",
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("File range".rjust(LABEL_PAD))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        "Select rows with first column [File #] value within the range",
                                        wrap=400,
                                    )
                                with dpg.group(horizontal=False):
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("from".rjust(4))
                                        dpg.add_input_int(
                                            tag="from",
                                            width=-1,
                                            max_value=10000,
                                            min_value=1,
                                            min_clamped=True,
                                            max_clamped=True,
                                            default_value=2100,
                                            on_enter=True,
                                            callback=self.populate_table,
                                        )
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("to".rjust(4))
                                        with dpg.tooltip(
                                            dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                        ):
                                            dpg.add_text(
                                                "Value of -1 indicates no upper limit (up to last row)",
                                                wrap=400,
                                            )
                                        dpg.add_input_int(
                                            tag="to",
                                            width=-1,
                                            max_value=10000,
                                            min_value=-1,
                                            default_value=-1,
                                            min_clamped=True,
                                            max_clamped=True,
                                            on_enter=True,
                                            callback=self.populate_table,
                                        )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Table dimensions".rjust(LABEL_PAD))
                                dpg.add_text(
                                    default_value="(0, 0)", tag="table_dimensions"
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
                                    dpg.add_plot_axis(
                                        dpg.mvYAxis, label="Counts", tag="y_axis"
                                    )
                            with dpg.tab(label="PCA"):
                                with dpg.plot(
                                    tag="pca_plot",
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

                    with dpg.group(tag="table_wrapper"):
                        with dpg.table(
                            label="Results table",
                            tag=self.table_tag,
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

            with dpg.window(
                label="Settings",
                tag="settings_modal",
                show=False,
                no_move=True,
                no_collapse=True,
                modal=True,
                width=700,
                height=400,
                no_resize=True,
            ):
                with dpg.tab_bar():
                    with dpg.tab(label="General"):
                        with dpg.child_window(
                            label="Appearance",
                            width=-1,
                            height=-1,
                            menubar=True,
                            no_scrollbar=True,
                        ):
                            with dpg.menu_bar():
                                with dpg.menu(label="UI", enabled=False):
                                    pass
                            with dpg.group(horizontal=True):
                                dpg.add_text("Theme")
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """UI theme
                                                Default: "Dark"
                                            """,
                                        wrap=400,
                                    )
                                dpg.add_combo(
                                    items=["Light", "Dark"],
                                    default_value="Dark",
                                    width=100,
                                )
                    with dpg.tab(label="Plots"):
                        with dpg.child_window(
                            label="PCA",
                            width=-1,
                            height=110,
                            menubar=True,
                            no_scrollbar=True,
                        ):
                            with dpg.menu_bar():
                                with dpg.menu(label="PCA", enabled=False):
                                    pass
                            with dpg.group(horizontal=True):
                                dpg.add_text("Minimum selections".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Sets minimum number of rows selected to render PCA plot
                                                Default: 4
                                            """,
                                        wrap=400,
                                    )
                                dpg.add_input_int(
                                    width=90,
                                    default_value=4,
                                    min_value=4,
                                    max_value=100,
                                    min_clamped=True,
                                    max_clamped=True,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Axes ratios equal".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Sets minimum number of rows selected to render PCA plot
                                                Default: 4
                                            """,
                                        wrap=400,
                                    )
                                dpg.add_checkbox()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Annotation mode".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Sets minimum number of rows selected to render PCA plot
                                                Default: 4
                                            """,
                                        wrap=400,
                                    )
                                dpg.add_combo(
                                    width=100,
                                    items=["All", "Closest", "None"],
                                    default_value="All",
                                )
                        with dpg.child_window(
                            label="PDZ",
                            width=-1,
                            height=-1,
                            menubar=True,
                            no_scrollbar=True,
                        ):
                            with dpg.menu_bar():
                                with dpg.menu(label="PDZ", enabled=False):
                                    pass
                            with dpg.group(horizontal=True):
                                dpg.add_text("Update on select".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Sets minimum number of rows selected to render PCA plot
                                                Default: 4
                                            """,
                                        wrap=400,
                                    )
                                dpg.add_checkbox()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Maximum plots shown".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Sets minimum number of rows selected to render PCA plot
                                                Default: 4
                                            """,
                                        wrap=400,
                                    )
                                dpg.add_input_int(
                                    width=90,
                                    default_value=10,
                                    min_value=2,
                                    max_value=100,
                                    min_clamped=True,
                                    max_clamped=True,
                                )
                    with dpg.tab(label="Table"):
                        with dpg.child_window(
                            label="Defaults",
                            width=-1,
                            height=120,
                            menubar=True,
                            no_scrollbar=True,
                        ):
                            with dpg.menu_bar():
                                with dpg.menu(label="Defaults", enabled=False):
                                    pass

                        with dpg.child_window(
                            label="Columns",
                            width=-1,
                            height=-1,
                            menubar=True,
                            no_scrollbar=True,
                        ):
                            with dpg.menu_bar():
                                with dpg.menu(label="Columns", enabled=False):
                                    pass
                            with dpg.group(horizontal=True):
                                dpg.add_text("Elements".ljust(8))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Sets minimum number of rows selected to render PCA plot
                                                Default: 4
                                            """,
                                        wrap=400,
                                    )
                                dpg.add_listbox(items=RESULT_ELEMENTS)

                            with dpg.group(horizontal=True):
                                dpg.add_text("Info".ljust(8))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        """Sets minimum number of rows selected to render PCA plot
                                                Default: 4
                                            """,
                                        wrap=400,
                                    )
                                dpg.add_listbox(items=RESULTS_INFO)

                    with dpg.tab(label="Hotkeys"):
                        with dpg.table():
                            dpg.add_table_column(label="Action")
                            dpg.add_table_column(label="Shortcut")
                            dpg.add_table_row()
                            dpg.add_table_row()
                    with dpg.tab(label="Import/Export"):
                        with dpg.child_window(
                            label="USB",
                            width=-1,
                            height=140,
                            menubar=True,
                            no_scrollbar=True,
                        ):
                            with dpg.menu_bar():
                                with dpg.menu(label="USB", enabled=False):
                                    pass
                            with dpg.group(horizontal=True):
                                dpg.add_text("Connect automatically".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        "",
                                        wrap=400,
                                    )
                                dpg.add_checkbox(default_value=True)
                            with dpg.group(horizontal=True):
                                dpg.add_text("Use default file structure".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        "",
                                        wrap=400,
                                    )
                                dpg.add_checkbox(default_value=True)
                            with dpg.group(horizontal=True):
                                dpg.add_text(".pdz folder name".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        "",
                                        wrap=400,
                                    )
                                dpg.add_input_text(default_value="DATA", width=200)
                            with dpg.group(horizontal=True):
                                dpg.add_text("Results table name".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        "",
                                        wrap=400,
                                    )
                                dpg.add_input_text(
                                    default_value="Results.csv", width=200
                                )

                        with dpg.child_window(
                            label="Export",
                            width=-1,
                            height=-1,
                            menubar=True,
                            no_scrollbar=True,
                        ):
                            with dpg.menu_bar():
                                with dpg.menu(label="Export", enabled=False):
                                    pass
                            with dpg.group(horizontal=True):
                                dpg.add_text("Include header".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        "",
                                        wrap=400,
                                    )
                                dpg.add_checkbox(default_value=True)
                            with dpg.group(horizontal=True):
                                dpg.add_text("Fill in zeroes".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        "",
                                        wrap=400,
                                    )
                                dpg.add_checkbox(default_value=False)
                            with dpg.group(horizontal=True):
                                dpg.add_text("Default path".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        "",
                                        wrap=400,
                                    )
                                dpg.add_input_text(default_value="Documents", width=200)
                            with dpg.group(horizontal=True):
                                dpg.add_text("Default format".ljust(28))
                                with dpg.tooltip(
                                    dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                ):
                                    dpg.add_text(
                                        "",
                                        wrap=400,
                                    )
                                dpg.add_combo(
                                    items=[".csv", ".xlsx"],
                                    default_value=".csv",
                                    width=200,
                                )

    def show_settings_modal(self):
        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        dpg.configure_item("settings_modal", pos=[w // 2 - 350, h // 2 - 200])
        dpg.show_item("settings_modal")
