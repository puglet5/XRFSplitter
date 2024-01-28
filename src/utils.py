# PDZReader https://github.com/zebhall/PDZReader

import csv
import functools
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime as dt
from functools import cached_property, partial, wraps
from io import StringIO
from pathlib import Path
from struct import unpack
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    NotRequired,
    ParamSpec,
    TypedDict,
    TypeVar,
)

import coloredlogs
import dearpygui.dearpygui as dpg
import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

matplotlib.use("agg")
plt.ioff()
pd.set_option("future.no_silent_downcasting", True)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=Path(ROOT_DIR, "log/main.log"), filemode="a")
coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

LABEL_PAD = 23
TABLE_SIZE_APPROXIMATION_FACTOR_KB = 15000

Result = TypedDict(
    "Result",
    {
        "File #": str,
        "Alloy 1": NotRequired[str],
        "Alloy 2": NotRequired[str],
        "Alloy 3": NotRequired[str],
        "Application": NotRequired[str],
        "Cal Check": NotRequired[str],
        "DateTime": NotRequired[str],
        "ElapsedTime": NotRequired[str],
        "Field1": NotRequired[str],
        "Field2": NotRequired[str],
        "ID": NotRequired[str],
        "Included in Average": NotRequired[str],
        "Match Qual 1": NotRequired[str],
        "Match Qual 2": NotRequired[str],
        "Match Qual 3": NotRequired[str],
        "Method": NotRequired[str],
        "Multiplier": NotRequired[str],
        "Name": NotRequired[str],
        "Operator": NotRequired[str],
        "Li": NotRequired[str],
        "Li Err": NotRequired[str],
        "Be": NotRequired[str],
        "Be Err": NotRequired[str],
        "C": NotRequired[str],
        "C Err": NotRequired[str],
        "Mg": NotRequired[str],
        "Mg Err": NotRequired[str],
        "MgO": NotRequired[str],
        "MgO Err": NotRequired[str],
        "Al": NotRequired[str],
        "Al Err": NotRequired[str],
        "Al2O3": NotRequired[str],
        "Al2O3 Err": NotRequired[str],
        "Si": NotRequired[str],
        "Si Err": NotRequired[str],
        "SiO2": NotRequired[str],
        "SiO2 Err": NotRequired[str],
        "P": NotRequired[str],
        "P Err": NotRequired[str],
        "S": NotRequired[str],
        "S Err": NotRequired[str],
        "Cl": NotRequired[str],
        "Cl Err": NotRequired[str],
        "K": NotRequired[str],
        "K Err": NotRequired[str],
        "K2O": NotRequired[str],
        "K2O Err": NotRequired[str],
        "Ca": NotRequired[str],
        "Ca Err": NotRequired[str],
        "Ti": NotRequired[str],
        "Ti Err": NotRequired[str],
        "V": NotRequired[str],
        "V Err": NotRequired[str],
        "Cr": NotRequired[str],
        "Cr Err": NotRequired[str],
        "Mn": NotRequired[str],
        "Mn Err": NotRequired[str],
        "Fe": NotRequired[str],
        "Fe Err": NotRequired[str],
        "Co": NotRequired[str],
        "Co Err": NotRequired[str],
        "Ni": NotRequired[str],
        "Ni Err": NotRequired[str],
        "Cu": NotRequired[str],
        "Cu Err": NotRequired[str],
        "Zn": NotRequired[str],
        "Zn Err": NotRequired[str],
        "Ga": NotRequired[str],
        "Ga Err": NotRequired[str],
        "As": NotRequired[str],
        "As Err": NotRequired[str],
        "Se": NotRequired[str],
        "Se Err": NotRequired[str],
        "Rb": NotRequired[str],
        "Rb Err": NotRequired[str],
        "Sr": NotRequired[str],
        "Sr Err": NotRequired[str],
        "Y": NotRequired[str],
        "Y Err": NotRequired[str],
        "Zr": NotRequired[str],
        "Zr Err": NotRequired[str],
        "Nb": NotRequired[str],
        "Nb Err": NotRequired[str],
        "Mo": NotRequired[str],
        "Mo Err": NotRequired[str],
        "Ru": NotRequired[str],
        "Ru Err": NotRequired[str],
        "Rh": NotRequired[str],
        "Rh Err": NotRequired[str],
        "Pd": NotRequired[str],
        "Pd Err": NotRequired[str],
        "Ag": NotRequired[str],
        "Ag Err": NotRequired[str],
        "Cd": NotRequired[str],
        "Cd Err": NotRequired[str],
        "In": NotRequired[str],
        "In Err": NotRequired[str],
        "Sn": NotRequired[str],
        "Sn Err": NotRequired[str],
        "Sb": NotRequired[str],
        "Sb Err": NotRequired[str],
        "Te": NotRequired[str],
        "Te Err": NotRequired[str],
        "Ba": NotRequired[str],
        "Ba Err": NotRequired[str],
        "La": NotRequired[str],
        "La Err": NotRequired[str],
        "Ce": NotRequired[str],
        "Ce Err": NotRequired[str],
        "Hf": NotRequired[str],
        "Hf Err": NotRequired[str],
        "Ta": NotRequired[str],
        "Ta Err": NotRequired[str],
        "W": NotRequired[str],
        "W Err": NotRequired[str],
        "Re": NotRequired[str],
        "Re Err": NotRequired[str],
        "Os": NotRequired[str],
        "Os Err": NotRequired[str],
        "Ir": NotRequired[str],
        "Ir Err": NotRequired[str],
        "Pt": NotRequired[str],
        "Pt Err": NotRequired[str],
        "Au": NotRequired[str],
        "Au Err": NotRequired[str],
        "Hg": NotRequired[str],
        "Hg Err": NotRequired[str],
        "Tl": NotRequired[str],
        "Tl Err": NotRequired[str],
        "Pb": NotRequired[str],
        "Pb Err": NotRequired[str],
        "Bi": NotRequired[str],
        "Bi Err": NotRequired[str],
        "Th": NotRequired[str],
        "Th Err": NotRequired[str],
        "U": NotRequired[str],
        "U Err": NotRequired[str],
    },
)
Results = dict[str, Result]
ELEMENT_SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
ELEMENT_NAMES = [
    "Hydrogen",
    "Helium",
    "Lithium",
    "Beryllium",
    "Boron",
    "Carbon",
    "Nitrogen",
    "Oxygen",
    "Fluorine",
    "Neon",
    "Sodium",
    "Magnesium",
    "Aluminium",
    "Silicon",
    "Phosphorus",
    "Sulfur",
    "Chlorine",
    "Argon",
    "Potassium",
    "Calcium",
    "Scandium",
    "Titanium",
    "Vanadium",
    "Chromium",
    "Manganese",
    "Iron",
    "Cobalt",
    "Nickel",
    "Copper",
    "Zinc",
    "Gallium",
    "Germanium",
    "Arsenic",
    "Selenium",
    "Bromine",
    "Krypton",
    "Rubidium",
    "Strontium",
    "Yttrium",
    "Zirconium",
    "Niobium",
    "Molybdenum",
    "Technetium",
    "Ruthenium",
    "Rhodium",
    "Palladium",
    "Silver",
    "Cadmium",
    "Indium",
    "Tin",
    "Antimony",
    "Tellurium",
    "Iodine",
    "Xenon",
    "Caesium",
    "Barium",
    "Lanthanum",
    "Cerium",
    "Praseodymium",
    "Neodymium",
    "Promethium",
    "Samarium",
    "Europium",
    "Gadolinium",
    "Terbium",
    "Dysprosium",
    "Holmium",
    "Erbium",
    "Thulium",
    "Ytterbium",
    "Lutetium",
    "Hafnium",
    "Tantalum",
    "Tungsten",
    "Rhenium",
    "Osmium",
    "Iridium",
    "Platinum",
    "Gold",
    "Mercury",
    "Thallium",
    "Lead",
    "Bismuth",
    "Polonium",
    "Astatine",
    "Radon",
    "Francium",
    "Radium",
    "Actinium",
    "Thorium",
    "Protactinium",
    "Uranium",
    "Neptunium",
    "Plutonium",
    "Americium",
    "Curium",
    "Berkelium",
    "Californium",
    "Einsteinium",
    "Fermium",
    "Mendelevium",
    "Nobelium",
    "Lawrencium",
    "Rutherfordium",
    "Dubnium",
    "Seaborgium",
    "Bohrium",
    "Hassium",
    "Meitnerium",
    "Darmstadtium",
    "Roentgenium",
    "Copernicium",
    "Nihonium",
    "Flerovium",
    "Moscovium",
    "Livermorium",
    "Tennessine",
    "Oganesson",
]
RESULT_KEYS: list[str] = list(Result.__dict__["__annotations__"].keys())
RESULT_KEYS_ERR = list(filter(lambda s: " Err" in s, RESULT_KEYS))
RESULT_ELEMENTS = [
    "Pb",
    "Sr",
    "Os",
    "P",
    "Mo",
    "Be",
    "In",
    "Hg",
    "Bi",
    "Se",
    "Al2O3",
    "Sn",
    "Cd",
    "Y",
    "C",
    "Ru",
    "Pt",
    "Au",
    "Cl",
    "Rh",
    "Ir",
    "Fe",
    "Te",
    "La",
    "Rb",
    "MgO",
    "Al",
    "Li",
    "Tl",
    "Th",
    "Hf",
    "Ag",
    "V",
    "Mg",
    "Ti",
    "Zn",
    "Si",
    "Ni",
    "Nb",
    "Sb",
    "U",
    "Cu",
    "W",
    "Zr",
    "Re",
    "Ta",
    "As",
    "Pd",
    "Ba",
    "Ga",
    "SiO2",
    "Ce",
    "S",
    "Ca",
    "Co",
    "Cr",
    "K2O",
    "K",
    "Mn",
]
RESULTS_INFO = [
    "DateTime",
    "Method",
    "Operator",
    "ID",
    "Field1",
    "Field2",
    "Multiplier",
    "Cal Check",
    "Name",
    "Application",
    "ElapsedTime",
    "Alloy 1",
    "Match Qual 1",
    "Alloy 2",
    "Match Qual 2",
    "Alloy 3",
    "Match Qual 3",
    "Included in Average",
]
FMTS = Literal["B", "h", "i", "I", "f", "s", "10", "5"]
ID_COL = "File #"

COLUMN_PRESETS: dict[str, list[tuple[list[str] | Literal["empty"], bool]]] = {
    "Info": [
        (RESULT_ELEMENTS, False),
        (RESULT_KEYS_ERR, False),
        (RESULTS_INFO, True),
    ],
    "Non-empty elements": [
        (RESULTS_INFO, False),
        (RESULT_KEYS_ERR, False),
        ("empty", False),
    ],
    "All elements": [
        (RESULTS_INFO, False),
        (RESULT_KEYS_ERR, False),
        ("empty", True),
    ],
}

T = TypeVar("T")
P = ParamSpec("P")


def element_z_to_symbol(z: int) -> str | None:
    if z == 0:
        return ""
    elif z <= 118:
        return ELEMENT_SYMBOLS[z - 1]
    else:
        logger.error("Error: Z out of range")
        return None


def element_z_to_name(z) -> str | None:
    if z <= 118:
        return ELEMENT_NAMES[z - 1]
    else:
        logger.error("Error: Z out of range")
        return None


def log_exec_time(f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def _wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = f(*args, **kwargs)
        logger.debug(f"{f.__name__}: {time.perf_counter() - start_time} s.")
        return result

    return _wrapper  # type:ignore


def progress_bar(
    f: Callable[P, Generator[float, None, None]]
) -> Callable[P, T]:  # type:ignore
    @wraps(f)
    def _wrapper(*args, **kwargs):
        progress_generator = f(*args, **kwargs)
        try:
            while True:
                progress = next(progress_generator)
                dpg.set_value("table_progress", progress)
                dpg.configure_item("table_progress", overlay=f"{progress*100:.0f}%")
        except StopIteration as result:
            dpg.set_value("table_progress", 0)
            dpg.configure_item("table_progress", overlay="")
            return result.value
        except TypeError:
            dpg.set_value("table_progress", 0)
            dpg.configure_item("table_progress", overlay="")
            return None

    return _wrapper  # type:ignore


@dataclass
class XRFSpectrum:
    datetime: dt = field(default=dt(1970, 1, 1, 0), init=False)
    counts: list[float] = field(default_factory=list, init=False)
    energies: list[float] = field(default_factory=list, init=False)
    source_voltage: float = field(default=0.0, init=False)  # in kV
    source_current: float = field(default=0.0, init=False)  # in uA
    filter_layer_1_element_z: int = field(default=0, init=False)  # Z num
    filter_layer_1_thickness: int = field(default=0, init=False)  # in um
    filter_layer_2_element_z: int = field(default=0, init=False)  # Z num
    filter_layer_2_thickness: int = field(default=0, init=False)  # in um
    filter_layer_3_element_z: int = field(default=0, init=False)  # Z num
    filter_layer_3_thickness: int = field(default=0, init=False)  # in um
    detector_temp_celsius: float = field(default=0.0, init=False)
    ambient_temp_fahrenheit: float = field(default=0.0, init=False)
    ambient_temp_celsius: float = field(default=0.0, init=False)
    nose_temp_celsius: float = field(default=0.0, init=False)
    nose_pressure: float = field(default=0.0, init=False)
    energy_per_channel: float = field(default=20.0, init=False)  # in eV
    vacuum_state: int = field(default=0, init=False)
    _energy_channel_start: float = field(default=0, init=False)  # in eV
    _n_channels: int = field(default=0, init=False)
    _filter_n: int = field(default=0, init=False)
    _time_elapsed_total: float = field(default=0.0, init=False)
    _time_live: float = field(default=0.0, init=False)
    _counts_valid: int = field(default=0, init=False)
    _counts_raw: int = field(default=0, init=False)

    def calculate_energies_list(self):
        self.energies = list(
            ((i * self.energy_per_channel + self._energy_channel_start) * 0.001)
            for i in range(0, self._n_channels)
        )
        return self.energies


@dataclass
class PDZFile:
    pdz_file_path: Path
    name: str = field(default="", init=False)
    anode_element_z: bytes = field(default=b"", init=False)
    tube_name: str = field(default="", init=False)
    tube_number: int = field(default=0, init=False)
    spectra: list[XRFSpectrum] = field(default_factory=list, init=False)
    phase_count: int = field(default=0, init=False)
    datetime: dt = field(default=dt(1970, 1, 1, 0), init=False)
    instrument_serial_number: str = field(default="", init=False)
    instrument_build_number: str = field(default="", init=False)
    user: str = field(default="", init=False)
    detector_type: str = field(default="", init=False)
    collimator_type: str = field(default="", init=False)
    spectra_used: Literal["[1]", "[1+2]"] = field(init=False)
    _assay_time_live: float = field(default=0.0, init=False)
    _assay_time_total: float = field(default=0.0, init=False)
    _other: Any = field(default=None, init=False)
    _firmware_vers_list_len: int = field(default=0, init=False)
    _pdz_file_version: int = field(default=25, init=False)

    def __repr__(self):
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    def __post_init__(self):
        self.name = os.path.basename(self.pdz_file_path)
        self._dispatch: dict[str, Callable[[], Any]] = {
            "B": partial(self._read_bytes, "B", 1),
            "h": partial(self._read_bytes, "<h", 2),
            "i": partial(self._read_bytes, "<i", 4),
            "I": partial(self._read_bytes, "<I", 4),
            "f": partial(self._read_bytes, "<f", 4),
            "s": self._read_string,
            "10": partial(self._read_n_bytes, 10),
            "5": partial(self._read_n_bytes, 5),
        }

        self._pdz_attr_formats_1: list[tuple[str, FMTS]] = [
            ("_pdz_file_version", "h"),
            ("_other", "I"),
            ("_other", "10"),
            ("_other", "I"),
            ("_other", "h"),
            ("_other", "I"),
            ("instrument_serial_number", "s"),
            ("instrument_build_number", "s"),
            ("anode_element_z", "B"),
            ("_other", "5"),
            ("detector_type", "s"),
            ("tube_name", "s"),
            ("tube_number", "h"),
            ("collimator_type", "s"),
            ("_firmware_vers_list_len", "i"),
        ]

        self._pdz_attr_formats_2: list[tuple[str, FMTS]] = [
            ("_other", "h"),
            ("_other", "i"),
            ("_other", "i"),
            ("_other", "i"),
            ("_other", "i"),
            ("_other", "i"),
            ("_other", "i"),
            ("_other", "f"),
            ("_other", "f"),
            ("_other", "f"),
            ("_other", "f"),
            ("_assay_time_live", "f"),
            ("_assay_time_total", "f"),
            ("measurement_mode", "s"),
            ("_other", "i"),
            ("user", "s"),
            ("_other", "h"),
        ]

        self._spectrum_attr_formats: list[tuple[str, FMTS]] = [
            ("_other", "i"),
            ("_other", "f"),
            ("_counts_raw", "i"),
            ("_counts_valid", "i"),
            ("_other", "f"),
            ("_other", "f"),
            ("_time_elapsed_total", "f"),
            ("_other", "f"),
            ("_other", "f"),
            ("_other", "f"),
            ("_time_live", "f"),
            ("source_voltage", "f"),
            ("source_current", "f"),
            ("filter_layer_1_element_z", "h"),
            ("filter_layer_1_thickness", "h"),
            ("filter_layer_2_element_z", "h"),
            ("filter_layer_2_thickness", "h"),
            ("filter_layer_3_element_z", "h"),
            ("filter_layer_3_thickness", "h"),
            ("_filter_n", "h"),
            ("detector_temp_celsius", "f"),
            ("ambient_temp_fahrenheit", "f"),
            ("vacuum_state", "i"),
            ("energy_per_channel", "f"),
            ("_other", "h"),
            ("_energy_channel_start", "f"),
            ("_spectrum_year", "h"),
            ("_spectrum_month", "h"),
            ("_spectrum_datetimedayofweek", "h"),
            ("_spectrum_day", "h"),
            ("_spectrum_hour", "h"),
            ("_spectrum_minute", "h"),
            ("_spectrum_second", "h"),
            ("_spectrum_millisecond", "h"),
            ("nose_pressure", "f"),
            ("_n_channels", "h"),
            ("nose_temp_celsius", "h"),
            ("_other", "h"),
            ("name", "s"),
            ("_other", "h"),
        ]

        self._read_pdz_data()
        self.datetime = self.spectra[0].datetime

    def _read_bytes(self, fmt: str, size: int):
        return unpack(fmt, self._pdz_file_reader.read(size))[0]

    def _read_n_bytes(self, size: int):
        return self._pdz_file_reader.read(size)

    def _read_string(self):
        return self._pdz_file_reader.read(self._read_bytes("<i", 4) * 2).decode("utf16")

    def _read_spectrum_params(self, spectrum: XRFSpectrum):
        for attr, fmt in self._spectrum_attr_formats:
            value = self._dispatch[fmt]()
            spectrum.__setattr__(attr, value)

        spectrum.ambient_temp_celsius = (spectrum.ambient_temp_fahrenheit - 32) / 1.8

    def _read_spectrum_counts(self, spectrum: XRFSpectrum):
        for _ in range(spectrum._n_channels):
            spectrum.counts.append(self._read_bytes("<i", 4))

    def _append_spectrum(self):
        self.spectra.append(XRFSpectrum())
        self._read_spectrum_params(self.spectra[-1])
        self._read_spectrum_counts(self.spectra[-1])
        self.spectra[-1].calculate_energies_list()
        self.phase_count += 1

    def _read_pdz_data(self):
        with open(self.pdz_file_path, "rb") as self._pdz_file_reader:
            for attr, fmt in self._pdz_attr_formats_1:
                self.__setattr__(attr, self._dispatch[fmt]())

            for _ in range(self._firmware_vers_list_len):
                self._read_bytes("h", 2)
                self._read_string()

            self.anode_element_symbol = element_z_to_symbol(int(self.anode_element_z))
            self.anode_element_name = element_z_to_name(int(self.anode_element_z))
            self.tube_type = f"{self.tube_name}:{self.tube_number}"

            for attr, fmt in self._pdz_attr_formats_2:
                self.__setattr__(attr, self._dispatch[fmt]())

            self._append_spectrum()
            if self._read_bytes("h", 2) == 3:
                self._append_spectrum()
                if self._read_bytes("h", 2) == 3:
                    self._append_spectrum()

    @cached_property
    def plot_data(self):
        if len(self.spectra) == 3:
            counts = [
                a + b for a, b in zip(self.spectra[1].counts, self.spectra[2].counts)
            ]
            self.spectra_used = "[1+2]"
        else:
            counts = self.spectra[0].counts
            self.spectra_used = "[1]"

        x = self.spectra[0].energies
        y = counts

        return x, y


@dataclass
class PlotData:
    pdz_folder: str | Path
    pdz_data: dict[str, PDZFile] = field(init=False)
    pca_data: npt.NDArray[np.float_] | None = field(init=False)
    pca_shapes: list[list[npt.NDArray[np.float_]]] | None = field(init=False)
    pca_info: PCA | None = field(init=False)

    def __post_init__(self):
        self.clear()

    def clear(self):
        self.pdz_data = {}
        self.pca_info = None
        self.pca_shapes = None
        self.pca_data = None
        self.pca = PCA(n_components=2, svd_solver="full")

    def generate_pca_data(self):
        if len(self.pdz_data) < 3:
            self.pca_data = None
            self.pca_info = None
            self.pca_shapes = None

        counts = [pdz.plot_data[1] for pdz in self.pdz_data.values()]
        data = minmax_scale(np.array(counts), feature_range=(0, 1), axis=1)

        pca_data = self.pca.fit_transform(data)
        x, y, *_ = pca_data.T
        pad = (x.max() + y.max() - x.min() - y.min()) ** 0.5
        xmin, xmax = x.min() - pad, x.max() + pad
        ymin, ymax = y.min() - pad, y.max() + pad

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = st.gaussian_kde([x, y])
        f = np.reshape(kernel(positions).T, xx.shape)

        allsegs = plt.contour(xx, yy, f).allsegs

        self.pca_info = self.pca
        self.pca_shapes = allsegs
        self.pca_data = pca_data

        plt.clf()


@dataclass
class TableData:
    path: str | Path
    original: DataFrame = field(init=False)
    current: DataFrame = field(init=False)
    selections: dict[str, int] = field(init=False)
    shown_cols: list[str] = field(init=False)
    shown_rows: list[str] = field(init=False)
    empty_cols: list[str] = field(init=False)
    original_cols: list[str] = field(init=False)
    sorted_by: tuple[str, bool] = field(init=False, default=(ID_COL, True))
    lod_shown: bool = field(init=False, default=True)
    last_row: int = field(init=True, default=10000)
    first_row: int = field(init=True, default=1)

    def __post_init__(self):
        self.selections = {}
        self.selected_rows_range = (1, -1)
        columns_to_drop = list(set(RESULTS_INFO) | set(RESULT_KEYS_ERR))
        raw_csv = self._raw_to_csv(self._results_to_array(self._construct()))
        self.original = pd.read_csv(
            raw_csv,
            dtype=str,
            delimiter=",",
            header=0,
            index_col=False,
            na_filter=False,
        )[::-1].replace(["0.0000"], "")

        self.current = self.original.drop(columns=columns_to_drop).copy()
        self.original_cols = self.original.columns.tolist()
        self.shown_cols = self.current.columns.tolist()
        self.shown_rows = self.current.iloc[:, 0].tolist()
        self.empty_cols = []
        self.first_row = int(
            pd.to_numeric(self.original.iloc[-1, 0], errors="raise", downcast="integer")
        )
        self.last_row = int(
            pd.to_numeric(self.original.iloc[0, 0], errors="raise", downcast="integer")
        )

    def _sorter(self, x, y):
        strx = str(x[1])
        stry = str(y[1])

        try:
            x = float(strx)
            y = float(stry)

            if x < y:
                return -1
            elif x > y:
                return 1
            else:
                return 0

        except ValueError:
            x = strx
            y = stry

        if x == "" and y == "< LOD":
            return -1
        if y == "" and x == "< LOD":
            return 1
        if x == "" or x == "< LOD" and y != x:
            return -1
        if y == "" or y == "< LOD" and x != y:
            return 1
        if x < y:
            return -1
        elif x > y:
            return 1
        else:
            return 0

    def _construct(self):
        with open(self.path, "r") as f:
            lines_arr = list(
                csv.reader(
                    f, skipinitialspace=True, delimiter=",", quoting=csv.QUOTE_NONE
                )
            )

        header_ids: list[int] = []
        for i, line in enumerate(lines_arr):
            if line[0] == ID_COL:
                header_ids.append(i)

        els_per_header: list[int] = [
            header_ids[n] - header_ids[n - 1] for n in range(1, len(header_ids))
        ] + [len(lines_arr) - header_ids[-1]]

        lines_same_header: list[list[list[str]]] = []
        for i, e in zip(header_ids, els_per_header):
            lines_same_header.append(lines_arr[i : i + e])

        results_data: Results = {}
        for lines in lines_same_header:
            for l in lines[1:]:
                results_data[l[0]] = dict(zip(lines[0], l))  # type: ignore

        return results_data

    def _results_to_array(self, data: Results):
        max_data_number = max([int(s) for s in data.keys()])

        data_selected = [
            [data.get(str(i), {}).get(k) for k in RESULT_KEYS]
            for i in range(0, max_data_number + 1)
        ]

        data_strings: list[list[str]] = [
            [i or "" for i in res] for res in data_selected if any(res)
        ]

        return data_strings

    def _raw_to_csv(self, data):
        data.insert(0, RESULT_KEYS)
        sio = StringIO()
        csv_writer = csv.writer(sio, quotechar="'")
        csv_writer.writerows(data)
        sio.seek(0)
        return sio

    def toggle_lod(self, state: bool):
        if state:
            self.current.update(
                self.original, overwrite=False, filter_func=lambda x: x == ""
            )
        else:
            self.current = self.current.replace("< LOD", "")

        self.lod_shown = state

    def filter_lod(self, df: DataFrame, state: bool):
        if state:
            df.update(self.original, overwrite=False, filter_func=lambda x: x == "")
        else:
            df = df.replace("< LOD", "")

        self.lod_shown = state

        return df

    def toggle_columns(self, keys: list[str] | Literal["empty"], state: bool):
        if state:
            if isinstance(keys, list):
                columns_to_show = self.shown_cols + keys
            else:
                columns_to_show = self.shown_cols + self.empty_cols

            ordered_intersection = sorted(
                set(self.original_cols) & set(columns_to_show),
                key=self.original_cols.index,
            )
            self.current = self.original[ordered_intersection].reindex(
                index=self.current.index
            )
        else:
            if isinstance(keys, list):
                self.current = self.current.drop(keys, axis=1, errors="ignore")
            elif keys == "empty":
                filtered = (
                    self.filter_lod(self.current.copy(), False)
                    .replace("", np.nan)
                    .dropna(how="all", axis=1)
                    .fillna("")
                )

                if empty_cols := self.current.columns.difference(
                    filtered.columns
                ).tolist():
                    self.empty_cols = empty_cols

                self.current = self.current.drop(empty_cols, axis=1)

        self.shown_cols = self.current.columns.tolist()

    def select_rows_range(
        self,
        row_range: tuple[int, int] | None = None,
    ):
        if row_range is None:
            row_range = self.selected_rows_range
        if row_range[1] != -1 and row_range[1] < row_range[0]:
            return
        if row_range[1] == -1:
            filtered = self.original.loc[
                pd.to_numeric(self.original[ID_COL]) >= row_range[0]
            ]
        else:
            filtered = self.original.loc[
                pd.to_numeric(self.original[ID_COL]).isin(
                    range(row_range[0], row_range[1] + 1)
                )
            ]

        self.shown_cols = filtered.columns.tolist()
        self.shown_rows = filtered.iloc[:, 0].tolist()

        self.current = filtered
        self.sort(*self.sorted_by)

    def filter_empty_rows(self):
        if self.current.empty:
            return

        filtered = self.current
        non_empty = filtered.replace("", np.nan)[self.shown_cols]
        non_empty = non_empty.dropna(
            subset=non_empty.columns.difference([ID_COL]), how="all", axis=0
        ).fillna("")
        filtered = filtered.loc[non_empty[ID_COL].index, :]

        self.current = filtered

    def filter_invalid_rows(self, threshold: float):
        df = (
            self.current[self.current.columns.intersection(RESULT_ELEMENTS)]
            .replace(["", "< LOD"], "0.0000")
            .astype(float)
        )
        mask = df.sum(axis=1) > threshold
        self.current = self.current[mask]

    def normalize_rows(self):
        elements = self.current.columns.intersection(RESULT_ELEMENTS)
        df = self.current[elements].replace(["", "< LOD"], "0.0000").astype(float)
        row_total = df.sum(axis=1) / 100
        normalized = df.div(row_total, axis=0)
        normalized = normalized.map("{:,.4f}".format)
        normalized.update(
            self.current, filter_func=lambda x: x == "0.0000", overwrite=False
        )
        self.current[elements] = normalized

    def select_rows_list(self, row_list: list[str] | None):
        if row_list is None:
            row_list = self.shown_rows
        filtered = self.current.loc[self.current[ID_COL].isin(row_list)]
        return filtered

    def sort(self, column: str, reverse: bool):
        col = self.current[column].tolist()
        sorted_arr_ids = [
            b[0]
            for b in sorted(
                enumerate(col), key=functools.cmp_to_key(self._sorter), reverse=reverse
            )
        ]

        self.current = self.current.iloc[sorted_arr_ids, :]
        self.sorted_by = (column, reverse)

        return self.current

    def append_errs(self):
        new_df = DataFrame()

        err_df = self.original[RESULT_KEYS_ERR].replace("", "0.0000")

        for i, col in enumerate(self.current[RESULT_ELEMENTS].columns.tolist()):
            new_df[col] = (
                self.current[RESULT_ELEMENTS]
                .iloc[:, i]
                .str.cat(err_df.iloc[:, i], sep="±")
            )
            new_df = new_df.replace(["±", "±0.0000"], "")

        return new_df

    def selected_to_clipboard(self):
        selected_rows = self.current[self.current[ID_COL].isin(self.selections)]
        selected_rows.to_clipboard(excel=True, index=False)

    def fill_with_zeros(self):
        self.current = self.current.replace("", "0.0000")
