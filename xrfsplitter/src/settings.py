from attrs import define, field
from typing import Callable

import dearpygui.dearpygui as dpg


@define
class Setting[T]:
    tag: str | int
    default_value: T
    value: T = field(init=False)
    callback: Callable

    def __attrs_post_init__(self):
        self.value = self.default_value

    def update(self):
        self.value = dpg.get_value(self.tag)
        self.callback()

    def set(self, value: T):
        dpg.set_value(self.tag, value)
        self.value = value

    @property
    def as_dict(self):
        return {
            "tag": self.tag,
            "default_value": self.default_value,
            "callback": self.update,
        }

    def disable(self):
        dpg.disable_item(self.tag)

    def enable(self):
        dpg.enable_item(self.tag)


@define
class Settings:
    table_highlighted: Setting[bool]
    lod_shown: Setting[bool]
    row_threshold: Setting[float]
    filled_with_zeros: Setting[bool]
    row_preset: Setting[str]
    column_preset: Setting[str]
    file_range_from: Setting[int]
    file_range_to: Setting[int]
    max_plots_shown: Setting[int]
