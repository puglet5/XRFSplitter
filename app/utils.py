import csv
import logging as log
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime as dt
from functools import partial
from io import StringIO
from pathlib import Path
from struct import unpack
from typing import Any, Callable, Literal, NotRequired, TypedDict

Result = TypedDict(
    "Result",
    {
        "File #": str,
        "DateTime": str,
        "Operator": str,
        "Name": str,
        "ID": str,
        "Field1": str,
        "Field2": str,
        "Application": str,
        "Method": str,
        "ElapsedTime": str,
        "Alloy 1": str,
        "Match Qual 1": str,
        "Alloy 2": str,
        "Match Qual 2": str,
        "Alloy 3": str,
        "Match Qual 3": str,
        "Li": NotRequired[str],
        "Li Err": NotRequired[str],
        "Be": NotRequired[str],
        "Be Err": NotRequired[str],
        "C": NotRequired[str],
        "C Err": NotRequired[str],
        "Mg": NotRequired[str],
        "Mg Err": NotRequired[str],
        "Al": NotRequired[str],
        "Al Err": NotRequired[str],
        "Si": NotRequired[str],
        "Si Err": NotRequired[str],
        "P": NotRequired[str],
        "P Err": NotRequired[str],
        "S": NotRequired[str],
        "S Err": NotRequired[str],
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
        "Tl": NotRequired[str],
        "Tl Err": NotRequired[str],
        "Pb": NotRequired[str],
        "Pb Err": NotRequired[str],
        "Bi": NotRequired[str],
        "Bi Err": NotRequired[str],
        "U": NotRequired[str],
        "U Err": NotRequired[str],
        "Multiplier": str,
        "Cal Check": str,
    },
)
Results = dict[str, Result]
RESULT_KEYS: list[str] = list(Result.__dict__["__annotations__"].keys())
RESULT_KEYS_NO_ERR = list(filter(lambda s: " Err" not in s, RESULT_KEYS))
FMTS = Literal["B", "h", "i", "I", "f", "s", "10", "5"]
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


def construct_data(filepath: Path):
    with open(filepath, "r") as f:
        lines_arr = list(
            csv.reader(f, skipinitialspace=True, delimiter=",", quoting=csv.QUOTE_NONE)
        )

    header_ids: list[int] = []
    for i, line in enumerate(lines_arr):
        if line[0] == "File #":
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


def select_data(data: Results, data_range: list[int] = [0], keys=RESULT_KEYS):
    max_data_number = max([int(s) for s in data.keys()]) + 1
    if data_range[0] is not None and len(data_range) == 1:
        data_range.append(max_data_number)

    data_selected = [
        [data.get(str(i), {}).get(k) for k in keys] for i in range(*data_range)
    ]
    data_strings: list[list[str]] = [
        [i or "" for i in res] for res in data_selected if any(res)
    ]

    return data_strings


def data_to_csv(data: list[list[str]], keys=RESULT_KEYS):
    data.insert(0, keys)
    sio = StringIO()
    csv_writer = csv.writer(sio, quotechar="'")
    csv_writer.writerows(data)
    sio.seek(0)
    return sio


def write_csv(buf: StringIO, path: Path):
    with open(path, "w") as f:
        buf.seek(0)
        shutil.copyfileobj(buf, f)


def element_z_to_symbol(z: int) -> str:
    """Returns 1-2 character Element symbol as a string"""
    if z == 0:
        return ""
    elif z <= 118:
        return ELEMENT_SYMBOLS[z - 1]
    else:
        log.error("Error: Z out of range")
        return "ERR"


def element_z_to_name(z):
    if z <= 118:
        return ELEMENT_NAMES[z - 1]
    else:
        log.error("Error: Z out of range")
        return None


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
    pdz_file_path: str
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
