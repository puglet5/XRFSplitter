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
from typing import Dict, NotRequired, TypedDict

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

Results = Dict[str, Result]

result_keys: list[str] = list(Result.__dict__["__annotations__"].keys())
result_keys_no_err = list(filter(lambda s: " Err" not in s, result_keys))


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


def select_data(data: Results, data_range: list[int] = [0]):
    max_data_number = max([int(s) for s in data.keys()]) + 1
    if data_range[0] is not None and len(data_range) == 1:
        data_range.append(max_data_number)

    data_selected = [
        [data.get(str(i), {}).get(k) for k in result_keys] for i in range(*data_range)
    ]
    data_strings: list[list[str]] = [
        [i or "" for i in res] for res in data_selected if any(res)
    ]

    return data_strings


def data_to_csv(data: list[list[str]]):
    data.insert(0, result_keys)
    sio = StringIO()
    csvWriter = csv.writer(sio)
    csvWriter.writerows(data)
    sio.seek(0)
    return sio


def write_csv(buf: StringIO, path: Path):
    with open(path, "w") as f:
        buf.seek(0)
        shutil.copyfileobj(buf, f)


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


def element_z_to_symbol(Z: int) -> str:
    """Returns 1-2 character Element symbol as a string"""
    if Z == 0:
        return ""
    elif Z <= 118:
        return ELEMENT_SYMBOLS[Z - 1]
    else:
        log.error("Error: Z out of range")
        return "ERR"


def element_z_to_name(Z):
    if Z <= 118:
        return ELEMENT_NAMES[Z - 1]
    else:
        log.error("Error: Z out of range")
        return None


@dataclass
class XRFSpectrum:
    name: str = ""
    datetime: dt = dt(1970, 1, 1, 0)
    counts: list[int] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    energy_channel_start: float = 0  # in eV
    n_channels: int = 0
    source_voltage: float = 0.0  # in kV
    source_current: float = 0.0  # in uA
    filter_layer_1_element_z: int = 0  # Z num
    filter_layer_1_thickness: int = 0  # in um
    filter_layer_2_element_z: int = 0  # Z num
    filter_layer_2_thickness: int = 0  # in um
    filter_layer_3_element_z: int = 0  # Z num
    filter_layer_3_thickness: int = 0  # in um
    filter_n = 0
    detector_temp_celsius: float = 0.0
    ambient_temp_fahrenheit: float = 0.0
    ambient_temp_celsius: float = 0.0
    nose_temp_celsius: float = 0.0
    nose_pressure: float = 0.0
    energy_per_channel: float = 20.0  # in eV
    time_elapsed_total: float = 0.0
    time_live: float = 0.0
    vacuum_state: int = 0
    counts_valid: int = 0
    counts_raw: int = 0

    def calculate_energies_list(self):
        self.energies = list(
            ((i * self.energy_per_channel + self.energy_channel_start) * 0.001)
            for i in range(0, self.n_channels)
        )
        return self.energies


@dataclass
class PDZFile:
    pdz_file_path: str
    anode_element_z: bytes = b""
    tube_name: str = ""
    tube_number: int = 0
    firmware_vers_list_len: int = 0

    def __post_init__(self):
        self.name = os.path.basename(self.pdz_file_path)
        self.dispatch = {
            "B": partial(self.read_bytes, "B", 1),
            "h": partial(self.read_bytes, "<h", 2),
            "i": partial(self.read_bytes, "<i", 4),
            "I": partial(self.read_bytes, "<I", 4),
            "f": partial(self.read_bytes, "<f", 4),
            "s": self.read_string,
            "10": partial(self.read_n_bytes, 10),
            "5": partial(self.read_n_bytes, 5),
        }

        self.pdz_attr_formats_1 = [
            ("pdz_file_version", "h"),
            ("other", "I"),
            ("other", "10"),
            ("other", "I"),
            ("other", "h"),
            ("other", "I"),
            ("instrument_serial_number", "s"),
            ("instrument_build_number", "s"),
            ("anode_element_z", "B"),
            ("other", "5"),
            ("detector_type", "s"),
            ("tube_name", "s"),
            ("tube_number", "h"),
            ("collimator_type", "s"),
            ("firmware_vers_list_len", "i"),
        ]

        self.pdz_attr_formats_2 = [
            ("other", "h"),
            ("other", "i"),
            ("other", "i"),
            ("other", "i"),
            ("other", "i"),
            ("other", "i"),
            ("other", "i"),
            ("other", "f"),
            ("other", "f"),
            ("other", "f"),
            ("other", "f"),
            ("assay_time_live", "f"),
            ("assay_time_total", "f"),
            ("measurement_mode", "s"),
            ("other", "i"),
            ("user", "s"),
            ("other", "h"),
        ]

        self.spectrum_attr_formats = [
            ("other", "i"),
            ("other", "f"),
            ("counts_raw", "i"),
            ("counts_valid", "i"),
            ("other", "f"),
            ("other", "f"),
            ("time_elapsed_total", "f"),
            ("other", "f"),
            ("other", "f"),
            ("other", "f"),
            ("time_live", "f"),
            ("source_voltage", "f"),
            ("source_current", "f"),
            ("filter_layer_1_element_z", "h"),
            ("filter_layer_1_thickness", "h"),
            ("filter_layer_2_element_z", "h"),
            ("filter_layer_2_thickness", "h"),
            ("filter_layer_3_element_z", "h"),
            ("filter_layer_3_thickness", "h"),
            ("filter_n", "h"),
            ("detector_temp_celsius", "f"),
            ("ambient_temp_fahrenheit", "f"),
            ("vacuum_state", "i"),
            ("energy_per_channel", "f"),
            ("other", "h"),
            ("energy_channel_start", "f"),
            ("spectrum_year", "h"),
            ("spectrum_month", "h"),
            ("spectrum_datetimedayofweek", "h"),
            ("spectrum_day", "h"),
            ("spectrum_hour", "h"),
            ("spectrum_minute", "h"),
            ("spectrum_second", "h"),
            ("spectrum_millisecond", "h"),
            ("nose_pressure", "f"),
            ("n_channels", "h"),
            ("nose_temp_celsius", "h"),
            ("other", "h"),
            ("name", "s"),
            ("other", "h"),
        ]

        self.read_pdz_data()
        self.datetime = self.spectrum_1.datetime

    def read_bytes(self, fmt, size):
        return unpack(fmt, self.pdz_file_reader.read(size))[0]

    def read_n_bytes(self, size):
        return self.pdz_file_reader.read(size)

    def read_string(self):
        return self.pdz_file_reader.read(self.read_bytes("<i", 4) * 2).decode("utf16")

    def read_spectrum_params(self, spectrum: XRFSpectrum):
        for attr, fmt in self.spectrum_attr_formats:
            value = self.dispatch[fmt]()
            spectrum.__setattr__(attr, value)

        spectrum.ambient_temp_celsius = (spectrum.ambient_temp_fahrenheit - 32) / 1.8

    def read_spectrum_counts(self, spectrum: XRFSpectrum):
        for _ in range(spectrum.n_channels):
            spectrum.counts.append(self.read_bytes("<i", 4))

    def read_pdz_data(self):
        with open(self.pdz_file_path, "rb") as self.pdz_file_reader:
            for attr, fmt in self.pdz_attr_formats_1:
                self.__setattr__(attr, self.dispatch[fmt]())

            for _ in range(self.firmware_vers_list_len):
                self.read_bytes("h", 2)
                self.read_string()

            self.anode_element_symbol = element_z_to_symbol(int(self.anode_element_z))
            self.anode_element_name = element_z_to_name(int(self.anode_element_z))
            self.tube_type = f"{self.tube_name}:{self.tube_number}"

            for attr, fmt in self.pdz_attr_formats_2:
                self.__setattr__(attr, self.dispatch[fmt]())

            self.spectrum_1 = XRFSpectrum()
            self.read_spectrum_params(self.spectrum_1)
            self.read_spectrum_counts(self.spectrum_1)
            self.spectrum_1.calculate_energies_list()
            self.phasecount = 1

            if self.read_bytes("h", 2) == 3:
                self.phasecount += 1
                self.spectrum_2 = XRFSpectrum()
                self.read_spectrum_params(self.spectrum_2)
                self.read_spectrum_counts(self.spectrum_2)
                self.spectrum_2.calculate_energies_list()

                if self.read_bytes("h", 2) == 3:
                    self.phasecount += 1
                    self.spectrum_3 = XRFSpectrum()
                    self.read_spectrum_params(self.spectrum_3)
                    self.read_spectrum_counts(self.spectrum_3)
                    self.spectrum_3.calculate_energies_list()
