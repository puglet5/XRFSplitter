import csv
import shutil
from io import StringIO
from pathlib import Path
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
