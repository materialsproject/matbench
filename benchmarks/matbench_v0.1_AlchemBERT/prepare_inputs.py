# SPDX-License-Identifier: GPL-3.0-only
"""Prepare identified, target-free text for a prospective AlchemBERT experiment.

The composition functions below are adapted from the archived GPL-3.0 source.
The v2 numeric formatter and task instructions are reused, with zero rendered
explicitly instead of the original None. Structure rendering sorts neighbour
descriptions and site groups, uses image-aware distances, and propagates neighbour
errors instead of silently replacing failed descriptions. This is explicitly a
new deterministic v2 recipe, NOT recovery of the missing historical v1 generator.
The original heuristic composition prose and numeric formatter are retained;
this utility does not establish their chemical accuracy or historical use.

An output directory must not exist. manifest.json is the completion marker and
contains the train/test file hashes and input_provenance needed by train_fold.py.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE_VERSION = "archived-v2-deterministic-1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def archived_module(name):
    """Load only an explicitly named, checksummed archival helper."""
    if name not in {"constants", "converter"}:
        raise ValueError("Unknown archived helper")
    source = ROOT / "upstream" / f"{name}.py"
    provenance = json.loads((ROOT / "provenance.json").read_text())
    if sha256(source) != provenance["upstream_sha256"][source.name]:
        raise ValueError(f"Archived source hash mismatch: {source.name}")
    spec = importlib.util.spec_from_file_location(
        f"alchembert_archived_{name}", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def n2e(value):
    if value == 0:
        return "zero"
    return archived_module("converter").num_to_words(value)


def parse_composition(input_str):
    pattern = r"([A-Za-z]+)([\d.]+)"
    matches = re.findall(pattern, input_str)
    composition = {element: float(percentage) for element, percentage in matches}
    return composition


def describe_alloy(composition):
    main_elements = []
    other_elements = []
    sorted_composition = sorted(
        composition.items(), key=lambda item: item[1], reverse=True
    )
    for element, percentage in sorted_composition:
        if percentage >= 0.01:
            main_elements.append(f"{n2e(percentage)} {element}")
        else:
            other_elements.append(f"{element} {n2e(percentage)}")

    main_elements_str = ", ".join(main_elements)
    other_elements_str = ", ".join(other_elements)

    description = (
        "This is an alloy with the following composition: "
        f"{main_elements_str}, and smaller amounts of {other_elements_str}."
    )
    return description


def describe_chemical_formula(formula):
    element_pattern = re.compile(r"([A-Z][a-z]*)(\d*\.?\d*)")
    group_pattern = re.compile(r"\(([A-Za-z0-9]+)\)(\d*)")

    def parse_formula(formula):
        elements = {}
        for element, qty in element_pattern.findall(formula):
            qty = float(qty) if qty else 1
            elements[element] = elements.get(element, 0) + qty
        return elements

    def parse_groups(formula):
        groups = {}
        for group, qty in group_pattern.findall(formula):
            qty = int(qty) if qty else 1
            groups[group] = qty
        return groups

    groups = parse_groups(formula)
    cleaned_formula = formula
    for group in groups:
        cleaned_formula = cleaned_formula.replace(f"({group}){groups[group]}", "")

    elements = parse_formula(cleaned_formula)

    base_elements = {element: qty for element, qty in elements.items() if qty >= 0.5}
    dopants = {element: qty for element, qty in elements.items() if qty < 0.5}

    description = [f"The chemical formula {formula} consists of:"]

    base_descriptions = []
    dopant_descriptions = []

    for element, qty in base_elements.items():
        base_descriptions.append(f"{n2e(qty)} atoms of {element}")

    for dopant, dopant_qty in dopants.items():
        found_replacement = False
        for base_element, base_qty in list(base_elements.items()):
            if abs(dopant_qty + base_qty - int(dopant_qty + base_qty)) < 1e-4:
                dopant_descriptions.append(
                    f"{n2e(dopant_qty)} of {dopant} replaced part of "
                    f"{base_element} in the structure"
                )
                base_elements[
                    base_element
                ] += dopant_qty  # multiple elements may dope into one position
                if (
                    base_elements[base_element] - int(base_elements[base_element])
                    < 1e-4
                ):
                    base_elements.pop(base_element)
                found_replacement = True
                break
        if not found_replacement:
            dopant_descriptions.append(
                f"{n2e(dopant_qty)} of {dopant} added as a dopant"
            )

    group_descriptions = []
    for group, qty in groups.items():
        group_elements = parse_formula(group)
        group_description = []
        for element, group_qty in group_elements.items():
            if group_qty > 1:
                group_description.append(f"{n2e(group_qty)} atoms of {element}")
            else:
                group_description.append(f"one atom of {element}")
        group_description_str = " and ".join(group_description)
        if qty > 1:
            group_descriptions.append(
                f"{n2e(qty)} units of {group} ({group_description_str})"
            )
        else:
            group_descriptions.append(
                f"one unit of {group} ({group_description_str})"
            )

    if base_descriptions:
        description.extend(base_descriptions)
    if dopant_descriptions:
        description.extend(dopant_descriptions)
    if group_descriptions:
        description.extend(group_descriptions)

    result = "; ".join(description)
    return result


def get_comp_nl(composition):
    return describe_chemical_formula(composition).replace(";", "", 1)


def get_comp_nl_steels(composition):
    return describe_alloy(parse_composition(composition))


def join_unique(descriptions):
    """Stable order replaces the archived unordered set join."""
    return ", ".join(sorted(set(descriptions)))


def get_structure_nl(structure):
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.core import Composition
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
    symmetry = analyzer.get_symmetry_dataset()
    lattice = structure.lattice
    formula = Composition(
        structure.composition.formula
    ).get_reduced_formula_and_factor()[0]
    description = (
        f"The chemical formula is {formula}. Its crystal structure is in the "
        f"{analyzer.get_space_group_symbol()} space group with lattice parameters: "
        f"a = {n2e(lattice.a)} Angstrom, b = {n2e(lattice.b)} Angstrom, "
        f"c = {n2e(lattice.c)} Angstrom, alpha = {n2e(lattice.alpha)} degree, "
        f"beta = {n2e(lattice.beta)} degree, gamma = {n2e(lattice.gamma)} degree. "
        "Detailed atomic interactions include: "
    )
    cnn = CrystalNN()
    position_description = {}
    equivalents = symmetry.equivalent_atoms.tolist()
    for idx, site in enumerate(structure):
        # Deliberately propagate neighbour/unsupported-site errors. A failed
        # material must not silently become a different, incomplete description.
        local_env = cnn.get_nn_info(structure, idx)
        neighbors = sorted(
            f"{nn['site'].specie} "
            f"({n2e(float(nn['site'].nn_distance))} Angstrom away)"
            for nn in local_env
        )
        key = (
            str(site.specie),
            symmetry.wyckoffs[idx],
            symmetry.site_symmetry_symbols[idx],
            equivalents.count(equivalents[idx]),
        )
        position_description.setdefault(key, []).append(", ".join(neighbors))
    for key in sorted(position_description):
        description += (
            f"{key[0]} at Wyckoff position {key[3]}{key[1]} with site symmetry "
            f"'{key[2]}' has neighboring interactions with: "
            f"{join_unique(position_description[key])}; "
        )
    return description.strip("; ")


def render(task_name, value):
    constants = archived_module("constants")
    kind = constants.task_type_dict[task_name][0]
    instruction = (
        f"Please tell me {constants.des_dict[task_name]} "
        f"{constants.units_dict[task_name]}of the following {kind}:"
    )
    if kind == "structure":
        text = get_structure_nl(value)
    elif task_name == "matbench_steels":
        text = get_comp_nl_steels(value)
    else:
        text = get_comp_nl(value)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Template produced empty text")
    return instruction, text


def input_sha256(inputs):
    """Hash official input identities and values, never their targets."""
    from monty.json import MontyEncoder

    digest = hashlib.sha256()
    for sample_id, value in inputs.items():
        encoded = json.dumps(
            [str(sample_id), value],
            cls=MontyEncoder,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest.update(encoded.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _checked_ids(inputs):
    ids = list(inputs.index)
    if any(
        not isinstance(item, str) or not item or item != item.strip() for item in ids
    ):
        raise ValueError("Official sample IDs must be nonempty unpadded strings")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate official sample IDs")
    return set(ids)


def write_inputs(path, task_name, inputs, renderer):
    with path.open("x", encoding="utf-8") as stream:
        for sample_id, value in inputs.items():
            instruction, text = renderer(task_name, value)
            json.dump(
                {"sample_id": sample_id, "instruction": instruction, "input": text},
                stream,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")


def prepare(task, fold, output_dir, renderer=render):
    """Create a new directory; publish its manifest only after both files pass."""
    if type(fold) is not int or fold not in task.folds:
        raise ValueError("Unknown official fold")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    train_inputs, _ = task.get_train_and_val_data(fold)
    test_inputs = task.get_test_data(fold, include_target=False)
    if _checked_ids(train_inputs) & _checked_ids(test_inputs):
        raise ValueError("Official train/test IDs overlap")
    provenance = json.loads((ROOT / "provenance.json").read_text())
    source_names = (
        "constants.py",
        "converter.py",
        "str_matbench_data_generation.py",
    )
    source_hashes = {}
    for name in source_names:
        source_hashes[name] = sha256(ROOT / "upstream" / name)
        if source_hashes[name] != provenance["upstream_sha256"][name]:
            raise ValueError(f"Archived source hash mismatch: {name}")
    input_provenance = {
        "template_version": TEMPLATE_VERSION,
        "historical_v1_reproduction": False,
        "source_commit": provenance["source_commit"],
        "archived_source_sha256": source_hashes,
        "prepare_inputs_sha256": sha256(__file__),
        "train_official_inputs_sha256": input_sha256(train_inputs),
        "test_official_inputs_sha256": input_sha256(test_inputs),
        "changes_from_archived_v2": [
            "Official sample IDs included; no targets serialized",
            "Sort neighbor descriptions, unique descriptions and site groups",
            "Use actual periodic-image neighbor distances; do not rewrap images",
            "Render zero explicitly; the archived formatter returns None",
            "Raise on failed structure descriptions instead of omitting neighbors",
        ],
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "matbench",
                "matminer",
                "pymatgen",
                "inflect",
                "numpy",
                "spglib",
            )
        },
    }
    # mkdir is exclusive: an existing user directory is never replaced. If
    # generation fails, remove only this invocation's newly reserved directory.
    output_dir.mkdir(parents=False, exist_ok=False)
    try:
        write_inputs(
            output_dir / "train.jsonl", task.dataset_name, train_inputs, renderer
        )
        write_inputs(
            output_dir / "test.jsonl", task.dataset_name, test_inputs, renderer
        )
        manifest = {
            "task": task.dataset_name,
            "fold": fold,
            "train_file": "train.jsonl",
            "test_file": "test.jsonl",
            "train_sha256": sha256(output_dir / "train.jsonl"),
            "test_sha256": sha256(output_dir / "test.jsonl"),
            "train_sample_count": len(train_inputs),
            "test_sample_count": len(test_inputs),
            "input_provenance": input_provenance,
        }
        manifest_tmp = output_dir / "manifest.json.tmp"
        with manifest_tmp.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
        # An atomic, exclusive completion marker never exposes partial JSON.
        os.link(manifest_tmp, output_dir / "manifest.json")
        manifest_tmp.unlink()
    except BaseException:
        shutil.rmtree(output_dir)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(archived_module("constants").task_type_dict),
    )
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from matbench.bench import MatbenchBenchmark

    benchmark = MatbenchBenchmark(autoload=False, subset=[args.task])
    task = getattr(benchmark, args.task)
    task.load()
    manifest = prepare(task, args.fold, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
