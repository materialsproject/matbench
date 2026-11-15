from converter import num_to_words as n2e

import multiprocessing as mp
import fire, json, re, gzip
from matbench.bench import MatbenchBenchmark
from pymatgen.io.cif import CifWriter
# from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Composition
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
from constants import units_dict, des_dict, task_type_dict, element_name_dict

def parse_composition(input_str):
    pattern = r'([A-Za-z]+)([\d.]+)'
    matches = re.findall(pattern, input_str)
    composition = {element: float(percentage) for element, percentage in matches}
    return composition

def describe_alloy(composition):
    main_elements = []
    other_elements = []
    sorted_composition = sorted(composition.items(), key=lambda item: item[1], reverse=True)
    for element, percentage in sorted_composition:
        element_name = element_name_dict.get(element, element)
        if percentage >= 0.01:
            main_elements.append(f"{n2e(percentage)} {element}")
        else:
            other_elements.append(f"{element} {n2e(percentage)}")

    main_elements_str = ", ".join(main_elements)
    other_elements_str = ", ".join(other_elements)

    description = f"This is an alloy with the following composition: {main_elements_str}, and smaller amounts of {other_elements_str}."
    return description

def describe_chemical_formula(formula):
    element_pattern = re.compile(r'([A-Z][a-z]*)(\d*\.?\d*)')
    group_pattern = re.compile(r'\(([A-Za-z0-9]+)\)(\d*)')

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
        cleaned_formula = cleaned_formula.replace(f"({group}){groups[group]}", '')

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
                dopant_descriptions.append(f"{n2e(dopant_qty)} of {dopant} replaced part of {base_element} in the structure")
                base_elements[base_element] += dopant_qty # multiple elements may dope into one position
                if base_elements[base_element] - int(base_elements[base_element]) < 1e-4:
                    base_elements.pop(base_element)
                found_replacement = True
                break
        if not found_replacement:
            dopant_descriptions.append(f"{n2e(dopant_qty)} of {dopant} added as a dopant")

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
            group_descriptions.append(f"{n2e(qty)} units of {group} ({group_description_str})")
        else:
            group_descriptions.append(f"one unit of {group} ({group_description_str})")

    if base_descriptions:
        description.extend(base_descriptions)
    if dopant_descriptions:
        description.extend(dopant_descriptions)
    if group_descriptions:
        description.extend(group_descriptions)

    result = "; ".join(description)
    return result

def get_comp_raw(composition):
    return str(composition)

def get_comp_nl(composition):
    return describe_chemical_formula(composition).replace(';', '', 1)

def get_comp_nl_steels(composition):
    return describe_alloy(parse_composition(composition))

def get_structure_cif(structure):
    ret = ""
    
    # analyzer = SpacegroupAnalyzer(structure, symprec=0.01)
    # symmetrized_structure = analyzer.get_symmetrized_structure()
    cif_str = CifWriter(structure, significant_figures=2, symprec=0.1).__str__()
    cif_str = re.sub(r'\s+', ' ', cif_str).strip()
    # cif_str = re.sub(r'(\d+\.\d{2})\d*', r'\1', cif_str)
    ret = cif_str
    return ret

def simplify_formula(formula):
    comp = Composition(formula)
    reduced_formula = comp.get_reduced_formula_and_factor()[0]
    return reduced_formula

def calculate_angle(site, neighbor1, neighbor2):
    vector1 = neighbor1.coords - site.coords
    vector2 = neighbor2.coords - site.coords
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def get_structure_nl_with_angle(structure):
    analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0) # SpacegroupAnalyzer(structure)
    space_group = analyzer.get_space_group_symbol()
    lattice = structure.lattice
    formula = simplify_formula(structure.composition.formula)
    cnn = CrystalNN()

    description = f"The chemical formula is {formula}. Its crystal structure belongs to the {space_group} space group with lattice parameters: "
    description += f"a = {lattice.a:.2f} Angstrom, b = {lattice.b:.2f} Angstrom, c = {lattice.c:.2f} Angstrom, "
    description += f"alpha = {lattice.alpha:.2f} degree, beta = {lattice.beta:.2f} degree, gamma = {lattice.gamma:.2f} degree. "
    description += "Detailed atomic interactions include: "

    position_description = {}
    for idx, site in enumerate(structure):
        wyckoff_position = analyzer.get_symmetry_dataset()['wyckoffs'][idx]
        equivalent_atoms = analyzer.get_symmetry_dataset()["equivalent_atoms"]
        equivalent_cnt = equivalent_atoms.tolist().count(equivalent_atoms[idx])

        site_symmetry = analyzer.get_symmetry_dataset()['site_symmetry_symbols'][idx]
        try:
            local_env = cnn.get_nn_info(structure, idx)
            neighbors_desc = ", ".join([f"{nn['site'].specie} ({site.distance(nn['site']):.2f}Å away)" for nn in local_env])
        except:
            local_env = []
            neighbors_desc = ", "

        neighbors_desc = []
        angles_desc = []
        cur_idx = -1
        for i, neighbor1 in enumerate(local_env):
            if cur_idx == i:
                continue
            if len(local_env) - i == 1:
                neighbors_desc.append(f"{neighbor1['site'].specie} ({site.distance(neighbor1['site']):.2f} Angstrom away)")
                break
            for j, neighbor2 in enumerate(local_env):
                if i < j:  # to avoid duplicate pairs
                    angle = calculate_angle(site, neighbor1['site'], neighbor2['site'])
                    angles_desc.append(f"{neighbor1['site'].specie} ({site.distance(neighbor1['site']):.2f} Angstrom away) and {neighbor2['site'].specie} ({site.distance(neighbor2['site']):.2f} Angstrom away) forming a {neighbor1['site'].specie}-{site.specie}-{neighbor2['site'].specie} angle of {angle:.1f} degree")
                    cur_idx = j
                    break

        key = (site.specie, wyckoff_position, site_symmetry, equivalent_cnt)
        if key not in position_description:
            position_description[key] = {'neighbors': [], 'angles': []}
        position_description[key]['neighbors'].append(", ".join(neighbors_desc))
        position_description[key]['angles'].extend(angles_desc)

    for key, descriptions in position_description.items():
        desc_set = set(descriptions['neighbors'])
        description += f"{key[0]} at Wyckoff position {key[3]}{key[1]} with site symmetry '{key[2]}' has neighboring interactions with " # : {', '.join(desc_set)}. "
        description += f"{', '.join(set(descriptions['angles']))}; "

    return description.strip("; ")

def get_structure_nl(structure):
    analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0) # SpacegroupAnalyzer(structure)
    space_group = analyzer.get_space_group_symbol()
    lattice = structure.lattice
    formula = simplify_formula(structure.composition.formula)
    cnn = CrystalNN()

    description = f"The chemical formula is {formula}. Its crystal structure is in the {space_group} space group with lattice parameters: "
    description += f"a = {n2e(lattice.a)} Angstrom, b = {n2e(lattice.b)} Angstrom, c = {n2e(lattice.c)} Angstrom, "
    description += f"alpha = {n2e(lattice.alpha)} degree, beta = {n2e(lattice.beta)} degree, gamma = {n2e(lattice.gamma)} degree. "
    description += "Detailed atomic interactions include: "

    position_description = {}
    for idx, site in enumerate(structure):
        wyckoff_position = analyzer.get_symmetry_dataset()['wyckoffs'][idx]
        equivalent_atoms = analyzer.get_symmetry_dataset()["equivalent_atoms"]
        equivalent_cnt = equivalent_atoms.tolist().count(equivalent_atoms[idx])

        site_symmetry = analyzer.get_symmetry_dataset()['site_symmetry_symbols'][idx]
        try:
            local_env = cnn.get_nn_info(structure, idx)
            neighbors_desc = ", ".join([f"{nn['site'].specie} ({n2e(site.distance(nn['site']))} Angstrom away)" for nn in local_env])
        except:
            neighbors_desc = ", "

        key = (site.specie, wyckoff_position, site_symmetry, equivalent_cnt)
        if key not in position_description:
            position_description[key] = []
        position_description[key].append(neighbors_desc)

    for key, descriptions in position_description.items():
        desc_set = set(descriptions)
        description += f"{key[0]} at Wyckoff position {key[3]}{key[1]} with site symmetry '{key[2]}' has neighboring interactions with: {', '.join(desc_set)}; "

    return description.strip("; ")

def get_structure_factory(structure, fmt):
    if fmt == "raw":
        return get_structure_cif(structure)
    elif fmt == "nl":
        return get_structure_nl(structure)
    elif fmt == "nl_angle":
        return get_structure_nl_with_angle(structure)
    else:
        pass

def get_composition_factory(composition, fmt, task):
    if fmt == "raw":
        return get_comp_raw(composition)
    elif fmt == "nl":
        if task == "matbench_steels":
            return get_comp_nl_steels(composition)
        else:
            return get_comp_nl(composition)
    else:
        pass

def generate_json(inputs, outputs, task, fold, train_or_test, structure_str_format="", composition_str_format=""):
    data = []
    fmt = ''
    if structure_str_format != "":
        fmt = structure_str_format
        for structure, value in zip(inputs, outputs):

            # if isinstance(value, float):
            #     v = str(round(value, 4))
            # else:
            #     v = str(value)
            v = str(value)

            new_item = {
                "instruction": f"Please tell me {des_dict[task]} {units_dict[task]}of the following structure:",
                "input": get_structure_factory(structure, structure_str_format),
                "output": f"{v}"
            }
            data.append(new_item)
    elif composition_str_format != "":
        fmt = composition_str_format
        for comp, value in zip(inputs, outputs):
            v = str(value)

            new_item = {
                "instruction": f"Please tell me {des_dict[task]} {units_dict[task]}of the following composition:",
                "input": get_composition_factory(comp, composition_str_format, task),
                "output": f"{v}"
            }
            data.append(new_item)
    else:
        raise ValueError("error, structure format and comp format should not be empty both!!!")

    json_str = json.dumps(data, indent=0)
    compressed_data = gzip.compress(json_str.encode('utf-8'))
    
    cnt = len(data)
    file_path = f'data_{fmt}_v2/{train_or_test}_{fold}_{task}_{fmt}_{cnt}.json.gz'
    
    with open(file_path, 'wb') as file:
        file.write(compressed_data)
    
    print(f"Data has been successfully dumped into {file_path}.")

def process_task(task, fold):
    task.load()
    print(task.dataset_name, task_type_dict[task.dataset_name])

    train_inputs, train_outputs = task.get_train_and_val_data(fold)
    test_inputs, test_outputs = task.get_test_data(fold, include_target=True)

    if task_type_dict[task.dataset_name][0] == "structure":
        structure_str_format = "nl"
        generate_json(train_inputs, train_outputs, task.dataset_name, fold, "Train", structure_str_format=structure_str_format)
        generate_json(test_inputs, test_outputs, task.dataset_name, fold, "Test", structure_str_format=structure_str_format)
    else:
        composition_str_format = "nl"
        generate_json(train_inputs, train_outputs, task.dataset_name, fold, "Train", composition_str_format=composition_str_format)
        generate_json(test_inputs, test_outputs, task.dataset_name, fold, "Test", composition_str_format=composition_str_format)

def main():
    mb = MatbenchBenchmark(autoload=False, subset=list(units_dict.keys()))
    # mb = MatbenchBenchmark(autoload=False, subset=["matbench_steels", "matbench_expt_gap", "matbench_expt_is_metal", "matbench_glass"])
    tasks = mb.tasks
    work_lst = []
    for task in tasks:
        for n in task.folds:
            work_lst.append((task, n))
    with mp.Pool(10) as pool:
            results = pool.starmap(process_task, work_lst)

if __name__ == "__main__":
    fire.Fire(main)
