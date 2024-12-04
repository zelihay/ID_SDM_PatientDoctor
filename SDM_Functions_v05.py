import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.image as gimg
import math
import random
import numpy as np
from IPython.display import Image
from graphviz import Source
from itertools import product
import copy
import pandas as pdp
import itertools


import pandas as pd
from datetime import datetime

def calculate_utility_values(model, utility_node_name, parameters):
    """
    This function computes utility values based on parent states, weights, and increments.
    """
    utility_node_id = model.idFromName(utility_node_name)
    utility_parents = model.parents(utility_node_id)
    parent_names = [model.variable(parent_id).name() for parent_id in utility_parents]

    weights = []
    increments = []

    for parent in parent_names:
        found = False
        for row in parameters:
            if row[0] == parent:
                weights.append(float(row[1]))
                increments.append(float(row[2]))
                found = True
                break

        if not found:
            raise ValueError(f"No parameters found for parent '{parent}'")


    assert len(weights) == len(utility_parents), (
        f"Number of weights must match number of parents. "
        f"Found {len(weights)} weights for {len(utility_parents)} parents."
    )
    assert len(increments) == len(utility_parents), (
        f"Number of increments must match number of parents. "
        f"Found {len(increments)} increments for {len(utility_parents)} parents."
    )

    utilities = {}

    parent_domains = {
        parent_name: model.variable(model.idFromName(parent_name)).labels()
        for parent_name in parent_names
    }

    for parent_state_combination in itertools.product(*parent_domains.values()):
        utility_value = 0

        for i, (parent_name, state) in enumerate(zip(parent_names, parent_state_combination)):
            num_states = len(parent_domains[parent_name])
            state_index = parent_domains[parent_name].index(state)

            incremental_value = (num_states - state_index - 1) * increments[i]
            weighted_value = weights[i] * incremental_value
            utility_value += weighted_value

        utilities[parent_state_combination] = utility_value

    for parent_state_combination, utility_value in utilities.items():
        state_dict = {parent_name: state for parent_name, state in zip(parent_names, parent_state_combination)}
        model.utility(utility_node_id)[state_dict] = utility_value




def update_decision_node_options(model1, model2, decision_node_name):
    """
    This function updates decision node options in model1 by looking options in model2 for the given decision node.
    """
    try:
        model1_decision_node_id = model1.idFromName(decision_node_name)
        model2_decision_node_id = model2.idFromName(decision_node_name)
    except gum.NotFound:
        print(f"Object not found: No element with the key '{decision_node_name}'")
        return None, None

    if not (model1.isDecisionNode(model1_decision_node_id) and model2.isDecisionNode(model2_decision_node_id)):
        print(f"The node '{decision_node_name}' is not a decision node in both models.")
        return None, None

    model1_options = model1.variable(decision_node_name).labels()
    model2_options = model2.variable(decision_node_name).labels()

    print(f"Options for decision node '{decision_node_name}' before updating: {model1_options}")

    missing_options = [option for option in model2_options if option not in model1_options]
    parents_of_decision_node = [parent for parent in model1.parents(model1_decision_node_id)]
    children_of_decision_node = [child for child in model1.children(model1_decision_node_id)]

    original_cpts = {}
    for child in children_of_decision_node:
        try:
            original_cpts[child] = model1.cpt(child).tolist()
        except gum.NotFound:
            print(f"CPT for child node '{model1.variable(child).name()}' not found. Skipping.")
            original_cpts[child] = None

    if missing_options:
        model1.erase(decision_node_name)

        all_options = list(model1_options) + missing_options
        updated_decision_var = gum.LabelizedVariable(decision_node_name, 'Decision Node', all_options)
        model1.addDecisionNode(updated_decision_var)

        for parent in parents_of_decision_node:
            model1.addArc(model1.variable(parent).name(), decision_node_name)

        for child in children_of_decision_node:
            model1.addArc(decision_node_name, model1.variable(child).name())
            if original_cpts[child] is not None:
                for idx, option in enumerate(model1_options):
                    model1.cpt(child)[{decision_node_name: option}] = original_cpts[child][idx]

        # Refresh child nodes after re-adding the decision node
        children_of_decision_node = [child for child in model1.children(model1_decision_node_id)]
        
        updated_model1_options = model1.variable(decision_node_name).labels()
        print(f"Updated options for decision node '{decision_node_name}': {updated_model1_options}")

    return model1, decision_node_name


def add_equal_probabilities_for_new_option(model):
    """
    This function calculates equal probabilities for a new state in the decision node to all its child nodes.
    """
    decision_node_id = next((node for node in model.nodes() if model.isDecisionNode(node)), None)

    if decision_node_id is None:
        print("No decision node found in the model.")
        return

    decision_node_name = model.variable(decision_node_id).name()
    all_decision_states = model.variable(decision_node_id).labels()
    new_state = all_decision_states[-1]  

    print(f"Decision Node: {decision_node_name}, States: {all_decision_states}")

    children = list(model.children(decision_node_id))

    for child_id in children:
        child_name = model.variable(child_id).name()
        child_states = model.variable(child_id).labels()
        num_child_states = len(child_states)

        print(f"Updating CPT for child '{child_name}' with {num_child_states} states...")

        try:
            cpt = model.cpt(child_id)
            parent_names = [model.variable(parent).name() for parent in model.parents(child_id)]

            for parent_combination in cpt.loopOnEvidence():
                if decision_node_name not in parent_combination:
                    continue

                if parent_combination[decision_node_name] == new_state:
                    continue

                equal_probabilities = [1.0 / num_child_states] * num_child_states
                print(f"Assigning probabilities {equal_probabilities} for combination {parent_combination}")
                cpt[parent_combination] = equal_probabilities

        except Exception as e:
            print(f"Error updating CPT for child '{child_name}': {e}")




def fill_model1_cpts_with_noisy_data(model1, model2, selected_node_name, noise_factor=0.1):
    """
    This function fills missing CPTs in model1 using noisy data derived from model2's CPTs.
    """
    if not model1.exists(selected_node_name) or not model2.exists(selected_node_name):
        print(f"Node '{selected_node_name}' does not exist in both models.")
        return model1

    model1_node_id = model1.idFromName(selected_node_name)
    model2_node_id = model2.idFromName(selected_node_name)
    
    model1_cpt = model1.cpt(model1_node_id)
    model2_cpt = model2.cpt(model2_node_id)

    if model1_cpt.shape != model2_cpt.shape:
        print(f"Shape mismatch for node '{selected_node_name}':")
        print(f"Model1 CPT shape: {model1_cpt.shape}")
        print(f"Model2 CPT shape: {model2_cpt.shape}")

        min_shape = min(model1_cpt.shape, model2_cpt.shape)
        noisy_cpt = add_noise_to_cpt(np.array(model2_cpt[:min_shape].tolist()), noise_factor)
    else:
        # If the shapes match, we can proceed with the noisy transfer
        noisy_cpt = add_noise_to_cpt(np.array(model2_cpt.tolist()), noise_factor)

    model1.cpt(model1_node_id)[:] = noisy_cpt

    return model1

def calculate_marginal_cpt_dynamic_with_noise(source_model, target_node, relevant_parents=None, noise_factor=0.0):
    """
    This function calculates the marginal CPT for the given chance node in the source model.
    """
    target_node_id = source_model.idFromName(target_node)
    all_parents = [source_model.variable(parent).name() for parent in source_model.parents(target_node_id)]
    print(f"Parents of '{target_node}' in source_model: {all_parents}")

    if relevant_parents is None:
        relevant_parents = all_parents
    elif isinstance(relevant_parents, str):
        relevant_parents = [relevant_parents]

    full_cpt = np.array(source_model.cpt(target_node_id).tolist())
    print(f"Full CPT shape: {full_cpt.shape}")

    retained_axes = [all_parents.index(parent) for parent in relevant_parents] + [len(all_parents)]
    marginal_axes = [i for i in range(len(full_cpt.shape)) if i not in retained_axes]

    print(f"Retained Axes: {retained_axes}")
    print(f"Marginalizing over axes: {marginal_axes}")

    marginalized_cpt = full_cpt.sum(axis=tuple(marginal_axes)) if marginal_axes else full_cpt
    print(f"Marginalized CPT shape (before alignment): {marginalized_cpt.shape}")

    expected_shape = [
        len(source_model.variable(source_model.idFromName(parent)).labels())
        for parent in relevant_parents
    ] + [len(source_model.variable(target_node_id).labels())]
    print(f"Expected CPT shape: {expected_shape}")

    current_shape = list(marginalized_cpt.shape)
    for i, (current_dim, expected_dim) in enumerate(zip(current_shape, expected_shape)):
        if current_dim < expected_dim:
            pad_width = [(0, 0)] * len(current_shape)
            pad_width[i] = (0, expected_dim - current_dim)
            marginalized_cpt = np.pad(
                marginalized_cpt, pad_width=pad_width, mode='constant', constant_values=0
            )
    print(f"Aligned Marginalized CPT shape: {marginalized_cpt.shape}")

    try:
        sums = marginalized_cpt.sum(axis=-1, keepdims=True)
        sums[sums == 0] = 1e-10  # Avoid division by zero
        normalized_cpt = marginalized_cpt / sums
        normalized_cpt = np.nan_to_num(normalized_cpt, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"Normalization error: {e}")
        normalized_cpt = np.nan_to_num(marginalized_cpt, nan=0.0, posinf=0.0, neginf=0.0)

    if noise_factor > 0:
        normalized_cpt = add_noise_to_cpt(normalized_cpt, noise_factor=noise_factor)

    print(f"Final Normalized CPT:\n{normalized_cpt}")
    return normalized_cpt


def add_marginal_cpt(patient_model, target_node, marginal_cpt, parent_states=None):
    """
    It transfers the given marginal CPT to a node in the patient model.
    """
    if parent_states:
        patient_model.cpt(target_node)[parent_states] = marginal_cpt
    else:
        patient_model.cpt(target_node)[:] = marginal_cpt
   

def transfer_marginal_cpt_multiple_parents(patient_model, target_node, parent_nodes, marginal_cpt):
    import itertools

    if isinstance(parent_nodes, str):
        parent_nodes = [parent_nodes]

    parent_states_dict = {
        parent: patient_model.variable(patient_model.idFromName(parent)).labels()
        for parent in parent_nodes
    }

    for axis_index, parent in enumerate(parent_nodes):
        expected_states = len(parent_states_dict[parent])
        current_dim = marginal_cpt.shape[axis_index]
        if current_dim != expected_states:
            if current_dim > expected_states:
                marginal_cpt = marginal_cpt.take(indices=range(expected_states), axis=axis_index)
            else:
                pad_width = [(0, 0)] * len(marginal_cpt.shape)
                pad_width[axis_index] = (0, expected_states - current_dim)
                marginal_cpt = np.pad(
                    marginal_cpt, pad_width, mode='constant', constant_values=0
                )

    cpt_parent_order = list(parent_states_dict.keys())

    reorder_indices = [cpt_parent_order.index(parent) for parent in parent_nodes]
    reorder_indices += [len(parent_nodes)]  

    if len(reorder_indices) != len(marginal_cpt.shape):
        raise ValueError(
            f"Mismatch between reorder indices and marginal CPT dimensions. "
            f"Reorder Indices: {reorder_indices}, CPT Shape: {marginal_cpt.shape}"
        )

    try:
        marginal_cpt_reordered = np.transpose(marginal_cpt, axes=reorder_indices)
    except ValueError as e:
        raise ValueError(
            f"Error transposing marginal CPT. Reorder Indices: {reorder_indices}, "
            f"Marginal CPT Shape: {marginal_cpt.shape}. Error: {e}"
        )


    state_combinations = list(itertools.product(*parent_states_dict.values()))

    for combination in state_combinations:
        parent_states = dict(zip(parent_nodes, combination))
        indices = tuple(parent_states_dict[parent].index(state) for parent, state in parent_states.items())

        prob = marginal_cpt_reordered[indices]

        prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)

        add_marginal_cpt(patient_model, target_node, prob, parent_states)

    return patient_model




def randomly_add_one_parent(model1, model2, node_name):
    """
    This function randomly adds one given chance node from model2 from model1.
    """
    print(f"\n--- Adding random parent and arc for '{node_name}' ---")
    
    if not model1.exists(node_name) or not model2.exists(node_name):
        print(f"Node '{node_name}' does not exist in both models.")
        return model1, None

    node_id_model2 = model2.idFromName(node_name)
    node_id_model1 = model1.idFromName(node_name)

    model2_parents = [model2.variable(parent).name() for parent in model2.parents(node_id_model2)]
    model1_parents = [model1.variable(parent).name() for parent in model1.parents(node_id_model1)]

    missing_parents = [parent for parent in model2_parents if parent not in model1_parents]

    if not missing_parents:
        print(f"No missing parents to add for '{node_name}'.")
        return model1, None

    selected_parent = random.choice(missing_parents)
    print(f"Selected parent to add: '{selected_parent}'")

    if not model1.exists(selected_parent):
        parent_id_model2 = model2.idFromName(selected_parent)
        parent_info = model2.variable(parent_id_model2)
        model1.add(parent_info)
        print(f"Parent node '{selected_parent}' added to model1.")

    if not model1.existsArc(model1.idFromName(selected_parent), node_id_model1):
        model1.addArc(model1.idFromName(selected_parent), node_id_model1)
        print(f"Arc added: {selected_parent} -> {node_name}")

    return model1, selected_parent

   
def add_noise_to_cpt(cpt_array, noise_factor=0.1):  
    """
    This function creates noise to a CPT and normalizes it to maintain probabilities.
    """
    
    noisy_cpt = cpt_array + noise_factor * np.random.randn(*cpt_array.shape)
    
    noisy_cpt = np.clip(noisy_cpt, 0, 1)
    
    sums = noisy_cpt.sum(axis=-1, keepdims=True)
    sums[sums == 0] = 1e-10  
    sums[np.isnan(sums)] = 1e-10     
  
    normalized_cpt = noisy_cpt / sums

    return normalized_cpt


def add_only_node_to_model1(model1, model2, node_name):
    """
    This function adds a node from model2 to model1 without transferring its parents or arcs.
    """
    if model1.exists(node_name):
        print(f"Node '{node_name}' already exists in model1.")

        node_id_model1 = model1.idFromName(node_name)
        node_id_model2 = model2.idFromName(node_name)

        states_model1 = model1.variable(node_id_model1).labels()
        states_model2 = model2.variable(node_id_model2).labels()

        if states_model1 != states_model2:
            print(f"States mismatch for '{node_name}'. Synchronizing states...")
            model1.variable(node_id_model1).setLabels(states_model2)
            print(f"States for '{node_name}' synchronized to: {states_model2}")

            marginal_cpt = calculate_marginal_cpt_dynamic_with_noise(model2, node_name, [], noise_factor=0.1)
            model1.cpt(node_name)[:] = marginal_cpt
            print(f"CPT for '{node_name}' updated after state synchronization.")
        else:
            print(f"States for '{node_name}' already match between model1 and model2. No action taken.")
        
        return model1, node_name  

    if not model2.exists(node_name):
        print(f"Node '{node_name}' does not exist in model2.")
        return model1, None

    node_id_model2 = model2.idFromName(node_name)

    if not model2.isChanceNode(node_id_model2):
        print(f"Node '{node_name}' is not a chance node in model2.")
        return model1, None
        
    node_info = model2.variable(node_id_model2)
    model1.add(node_info)
    print(f"Node '{node_name}' added to model1 (without parents or arcs).")

    return model1, node_name


   
def add_noise_to_increments(parameters, mu=0, sigma=5):
    """
    This function adds noise to increment values in parameter settings.
    """      
    noisy_parameters = parameters.copy()
    noise = np.random.normal(mu, sigma, size=noisy_parameters.shape[0])
    noisy_parameters[:, 2] = noisy_parameters[:, 2].astype(float) + noise
    return noisy_parameters

def delete_unmatched_arcs(model1, model2):
    """
    Ths function deletes arcs from unmatched chance nodes to the utility node in model1.
    """
    matched_parents = find_matched_parents(model2, model1)
    unmatched_parents = find_unmatched_parents(model1, model2)

    print("Matched Parents:", matched_parents)
    print("Unmatched Parents:", unmatched_parents)

    advisor_utility_node_id = next(node for node in model1.nodes() if model1.isUtilityNode(node))
    advisor_utility_node_name = model1.variable(advisor_utility_node_id).name()

    unmatched_chance_parents = [
        parent for parent in unmatched_parents if model1.isChanceNode(model1.idFromName(parent))
    ]

    print("Unmatched Chance Parents:", unmatched_chance_parents)

    for parent_name in unmatched_chance_parents:
        parent_id = model1.idFromName(parent_name)
        if (parent_id, advisor_utility_node_id) in model1.arcs():
            print(f"Deleting arc from '{parent_name}' to '{advisor_utility_node_name}'")
            model1.eraseArc(parent_id, advisor_utility_node_id)

    return model1

def find_matched_parents(model1, model2):
    model1_parents = get_utility_parents(model1)
    model2_parents = get_utility_parents(model2)

    matched_parents = set()
    for parent in model1_parents:
        if parent in model2_parents:
            matched_parents.add(parent)

    return matched_parents

def get_utility_parents(influence_diagram):
    utility_node_id = next((node for node in influence_diagram.nodes() if influence_diagram.isUtilityNode(node)), None)

    if utility_node_id is None:
        return []

    parents = []
    for parent_id in influence_diagram.parents(utility_node_id):
        parent_name = influence_diagram.variable(parent_id).name()
        parents.append(parent_name)

    return parents

def find_unmatched_parents(model1, model2):
    model1_parents = get_utility_parents(model1)
    model2_parents = get_utility_parents(model2)

    unmatched_parents = set(model1_parents) - set(model2_parents)

    return list(unmatched_parents)


###----------------FOR THE DONOR POOL------------####
# Lifelong Thyroid Replacement (2 states)
CPT1_Lifelong = [0.45, 0.55]
CPT2_Lifelong = [0.2, 0.8]
CPT3_Lifelong = [0.1, 0.9]
CPT4_Lifelong = [0.6, 0.4]
CPT5_Lifelong = [0.3, 0.7]

# Side Effects (2 states)
CPT1_SideEffects = [
    [[0.9, 0.1], [0.95, 0.05], [0.9, 0.1]],  # Parent state 1
    [[0.8, 0.2], [0.85, 0.15], [0.8, 0.2]]   # Parent state 2
]
CPT2_SideEffects = [
    [[0.9, 0.1], [0.95, 0.05], [0.9, 0.1]],  # Parent state 1
    [[0.8, 0.2], [0.85, 0.15], [0.8, 0.2]]   # Parent state 2
]

CPT3_SideEffects = [
    [[0.9, 0.1], [0.95, 0.05], [0.9, 0.1]],  # Parent state 1
    [[0.8, 0.2], [0.85, 0.15], [0.8, 0.2]]   # Parent state 2
]
CPT4_SideEffects = [
    [[0.9, 0.1], [0.95, 0.05], [0.9, 0.1]],  # Parent state 1
    [[0.8, 0.2], [0.85, 0.15], [0.8, 0.2]]   # Parent state 2
]

# Remission (2 states)
CPT1_Remission = [0.6, 0.4]
CPT2_Remission = [0.4, 0.6]
CPT3_Remission = [0.3, 0.7]
CPT4_Remission = [0.1, 0.9]
CPT5_Remission = [0.8, 0.2]

# Hypothyroidism (2 states)
CPT1_Hypothyroidism = [0.8, 0.2]
CPT2_Hypothyroidism = [0.9, 0.1]
CPT3_Hypothyroidism = [0.8, 0.2]
CPT4_Hypothyroidism = [0.9, 0.1]


# Cost (3 states)
CPT1_Cost = [0.2, 0.65, 0.05]
CPT2_Cost = [0.7, 0.2, 0.1]
CPT3_Cost = [0.6, 0.3, 0.1]
CPT4_Cost = [0.15, 0.75, 0.1]
CPT5_Cost = [0.1, 0.1, 0.8]
CPT6_Cost = [0.15, 0.25, 0.6]
 
pool = {
    "Lifelong": [CPT1_Lifelong, CPT2_Lifelong, CPT3_Lifelong, CPT4_Lifelong, CPT5_Lifelong],
    "SideEffects": [CPT1_SideEffects, CPT2_SideEffects,CPT3_SideEffects,CPT4_SideEffects],
    "Remission": [CPT1_Remission, CPT2_Remission, CPT3_Remission, CPT4_Remission, CPT5_Remission],
    "Hypothyroidism": [CPT1_Hypothyroidism, CPT2_Hypothyroidism,CPT3_Hypothyroidism,CPT4_Hypothyroidism],
    "Cost": [CPT1_Cost, CPT2_Cost, CPT3_Cost, CPT4_Cost, CPT5_Cost,CPT6_Cost],

}

selected_cpt_Lifelong_Thyroid_Replacement = random.choice(pool["Lifelong"])
selected_cpt_SideEffects = random.choice(pool["SideEffects"])
selected_cpt_Remission = random.choice(pool["Remission"])
selected_cpt_Hypothyroidism = random.choice(pool["Hypothyroidism"])
selected_cpt_Cost = random.choice(pool["Cost"])

def add_cpt_from_pool(patient_model, selected_cpt_Lifelong_Thyroid_Replacement, selected_cpt_SideEffects,selected_cpt_Remission,selected_cpt_Hypothyroidism,selected_cpt_Cost):
    """"
    Filling the missing CPTs from a donor pool.
    """
    patient_model.cpt('Lifelong_Thyroid_Replacement')[{'Treatment': 'Surgery'}] = selected_cpt_Lifelong_Thyroid_Replacement
    patient_model.cpt('SideEffects')[{'Treatment': 'Surgery'}] = selected_cpt_SideEffects
    patient_model.cpt('Remission')[{'Treatment': 'Surgery'}] = selected_cpt_Remission
    patient_model.cpt('Hypothyroidism')[{'Treatment': 'Surgery'}] = selected_cpt_Hypothyroidism
    patient_model.cpt('Cost')[{'Treatment': 'Surgery'}] = selected_cpt_Cost


import itertools

def add_equal_cpts_and_transfer(model, decision_node_name, new_state_name):
    """
    This function assigns equal probabilities to the new decision state and transfers them.
    """
    decision_node_id = model.idFromName(decision_node_name)
    children = model.children(decision_node_id)

    for child_id in children:
        child_name = model.variable(child_id).name()
        child_states = model.variable(child_id).labels()
        num_child_states = len(child_states)

        child_parents = model.parents(child_id)
        parent_names = [model.variable(parent_id).name() for parent_id in child_parents]
        parent_states = [model.variable(parent_id).labels() for parent_id in child_parents]

        try:
            for parent_state_combination in itertools.product(*parent_states):
                parent_state_dict = dict(zip(parent_names, parent_state_combination))
                parent_state_dict[decision_node_name] = new_state_name  

                equal_probabilities = [1.0 / num_child_states] * num_child_states
                model.cpt(child_name)[parent_state_dict] = equal_probabilities

            print(f"CPT updated for child '{child_name}' with new decision state '{new_state_name}'")
        except Exception as e:
            print(f"Error updating CPT for child '{child_name}': {e}")





def transfer_parent_node_parameter(model1_params, model2_params, parent_node_name):

    row_to_transfer = None
    for row in model2_params:
        if row[0] == parent_node_name:
            row_to_transfer = row
            break

    if row_to_transfer is None:
        raise ValueError(f"Parent node '{parent_node_name}' not found in model2 parameters.")

    for row in model1_params:
        if row[0] == parent_node_name:
            print(f"Parent node '{parent_node_name}' already exists in the parameters.")
            return model1_params  

    updated_model1_params = np.vstack([model1_params, row_to_transfer])

    return updated_model1_params

    
def calculate_baseline_eu(model):
        limid_solver = gum.ShaferShenoyLIMIDInference(model)
        limid_solver.makeInference()
        baseline_meu = limid_solver.MEU()
        return baseline_meu.get("mean", float("-inf")) if isinstance(baseline_meu, dict) else baseline_meu

def calculate_voi_with_marginal_probabilities(model, chance_node_name):
    baseline_eu = calculate_baseline_eu(model)
    
    chance_node_id = model.idFromName(chance_node_name)
    has_parents = len(model.parents(chance_node_id)) > 0
    chance_node_states = model.variable(chance_node_id).labels()

    marginal_probs = {}
    if has_parents:
        limid = gum.ShaferShenoyLIMIDInference(model)
        limid.makeInference()
        marginal_distribution = limid.posterior(chance_node_id)
        for i, prob in enumerate(marginal_distribution.tolist()):
            state_label = chance_node_states[i]
            marginal_probs[state_label] = prob
            
    else:
        cpt = model.cpt(chance_node_id)
        for i, prob in enumerate(cpt.tolist()):
            state_label = chance_node_states[i]
            marginal_probs[state_label] = prob
            
    expected_utilities = {}
    limid = gum.ShaferShenoyLIMIDInference(model)
    for state in chance_node_states:
        limid.setEvidence({chance_node_name: state})
        limid.makeInference()
        expected_utility = limid.MEU()
        expected_utilities[state] = float(expected_utility.get("mean", float('-inf')) if isinstance(expected_utility, dict) else expected_utility)
        limid.setEvidence({})  
    
    weighted_eu_with_perfect_info = sum(
        marginal_probs[state] * expected_utilities[state] for state in chance_node_states
    )
    
    voi = weighted_eu_with_perfect_info - baseline_eu
    
    return voi

##function for voi calculation 

def calculate_voi_for_relevant_chance_nodes(model):
    total_voi_by_node = {}

  
    decision_node_id = next(node for node in model.nodes() if model.isDecisionNode(node))
    decision_node_name = model.variable(decision_node_id).name()

    relevant_nodes = find_all_upstream_nodes(model, decision_node_id)

    root_nodes = find_root_nodes(model)
    relevant_nodes.update(root_nodes)

    for node_id in relevant_nodes:
        if model.isChanceNode(node_id):
            chance_node_name = model.variable(node_id).name()
            
            try:
                voi = calculate_voi_with_marginal_probabilities(model, chance_node_name)
                total_voi_by_node[chance_node_name] = voi
            except Exception as e:
                print(f"Error calculating VOI for {chance_node_name}: {e}")

    sorted_voi_by_node = dict(sorted(total_voi_by_node.items(), key=lambda item: item[1], reverse=True))

    print("\nTotal VOI for each relevant chance node (sorted by VOI):")
    for node, voi in sorted_voi_by_node.items():
        print(f"{node}: {voi}")
        
    return sorted_voi_by_node


def find_root_nodes(model):
    root_nodes = []
    for node_id in model.nodes():
        if model.isChanceNode(node_id) and not model.parents(node_id):
            root_nodes.append(node_id)
    return root_nodes


def find_all_upstream_nodes(model, node_id):
    upstream_nodes = set()
    to_visit = [node_id]

    while to_visit:
        current_node = to_visit.pop()
        for parent in model.parents(current_node):
            if parent not in upstream_nodes:
                upstream_nodes.add(parent)
                to_visit.append(parent)

    return upstream_nodes


def decision_alternative_transfer(model1, model2, decision_node):
    model1, decision_node = update_decision_node_options(model1, model2, decision_node)
    if model1 is None:
        print(f"Failed to update decision node options for '{decision_node}'. Terminating process.")
        return None, None   
    
    if random.random() < 0.5:
        add_equal_cpts_and_transfer(model1, decision_node,"Surgery")
        function_used = "add_equal_probabilities_for_new_option"    
    else:
        add_cpt_from_pool(model1, selected_cpt_Lifelong_Thyroid_Replacement, selected_cpt_SideEffects,
                          selected_cpt_Remission, selected_cpt_Hypothyroidism, selected_cpt_Cost)
        function_used = "add_cpt_from_pool"
        print({decision_node})
    
    return model1, function_used

def chance_node_transfer(model1, model2, chance_node,noise_factor=0.0):    
    model1, _ = add_only_node_to_model1(model1, model2, chance_node)
    model1, connected_node = randomly_add_one_parent(model1, model2, chance_node)
    marginal_cpt = calculate_marginal_cpt_dynamic_with_noise(model2, chance_node, connected_node)
    model1= transfer_marginal_cpt_multiple_parents(model1, chance_node, connected_node, marginal_cpt)
    model1 = fill_missing_cpts_with_noise(model1, model2, noise_factor=noise_factor)

    return model1



def fill_missing_cpts(model1, model2, noise_factor=0.05):
    """
    Identifies and fills missing or empty CPTs in model1 using CPTs from model2.
    Adds noise to the CPTs during the transfer.
    """
    print("Checking for missing CPTs in model1...")

    for node_id in model1.nodes():
        # Skip decision and utility nodes
        if model1.isDecisionNode(node_id) or model1.isUtilityNode(node_id):
            continue

        node_name = model1.variable(node_id).name()

        # Check if the CPT is empty (zero or uninitialized)
        cpt = model1.cpt(node_id)
        if cpt.sum() == 0 or np.isnan(cpt.sum()):
            if model2.exists(node_name):                
                model1 = fill_model1_cpts_with_noisy_data(model1, model2, node_name, noise_factor)
            else:
                print(f"Warning: Node '{node_name}' exists in model1 but not in model2. CPT cannot be filled.")

    
    return model1


def add_arc_to_utility_node(model,preference):
    utility_node_id = next((node for node in model.nodes() if model.isUtilityNode(node)), None)
    utility_node_name = model.variable(utility_node_id).name()

    decision_node_id = next((node for node in model.nodes() if model.isDecisionNode(node)), None)
    decision_node_name = model.variable(decision_node_id).name()


    model.addArc(preference,utility_node_name)
    model.addArc(decision_node_name,preference)
    return model


def add_preference_nodes_iteratively(model1, model2,parameters1, parameters2):

    model1_utility_node = next(node for node in model1.names() if model1.isUtilityNode(node))
    model2_utility_node = next(node for node in model2.names() if model2.isUtilityNode(node))
    
    model1_utility_parents = set(model1.parents(model1.idFromName(model1_utility_node)))
    model2_utility_parents = set(model2.parents(model2.idFromName(model2_utility_node)))

    preference_nodes = model2_utility_parents -  model1_utility_parents
    preference_nodes_names = [model2.variable(node).name() for node in preference_nodes]

    for node_name in preference_nodes_names:
        model1, _ = preference_transfer(model1, model2, parameters1, parameters2, node_name)

    return model1
    
def preference_transferORJ(model1, model2, parameters1, parameters2, parent_node_name):
    utility_node_name_model1 = next(
        node for node in model1.names() if model1.isUtilityNode(node)
    )
    utility_node_name_model2 = next(
        node for node in model2.names() if model2.isUtilityNode(node)
    )
    decision_node_name_model1 = next(
        node for node in model1.names() if model1.isDecisionNode(node)
    )
    decision_node_name_model2 = next(
        node for node in model2.names() if model2.isDecisionNode(node)
    )
    model1,_ = add_only_node_to_model1(model1, model2, parent_node_name)
    updated_parameters1 = transfer_parent_node_parameter(
        parameters1, parameters2, parent_node_name
    )
    
    noisy_parameters1 = add_noise_to_increments(updated_parameters1)
    model1= add_arc_to_utility_node(model1, parent_node_name)
    marginal_cpt = calculate_marginal_cpt_dynamic_with_noise(model2, parent_node_name, decision_node_name_model2)
    model1= transfer_marginal_cpt_multiple_parents(model1, parent_node_name, decision_node_name_model1, marginal_cpt)


    model1 = fill_missing_cpts_with_noise(model1, model2)
    calculate_utility_values(model1, utility_node_name_model1, noisy_parameters1)
    
    return model1, noisy_parameters1

def preference_transfer(model1, model2, parameters1, parameters2, parent_node_name):
   
    if parent_node_name in model1.names():
        print(f"Node '{parent_node_name}' already exists in model1.")
        if parent_node_name in model2.names():
            if model1.variable(model1.idFromName(parent_node_name)).labels() == model2.variable(model2.idFromName(parent_node_name)).labels():
                print(f"States for '{parent_node_name}' already match between model1 and model2. No action taken.")
                return model1, parameters1
        print(f"Node '{parent_node_name}' found in model1 but not in model2.")
        return model1, parameters1


    decision_node_name_model1 = next(
    node for node in model1.names() if model1.isDecisionNode(node)
    )
    decision_node_name_model2 = next(
        node for node in model2.names() if model2.isDecisionNode(node)
    )
    
    utility_node_name_model1 = next(node for node in model1.names() if model1.isUtilityNode(node))
    model1, _ = add_only_node_to_model1(model1, model2, parent_node_name)
    updated_parameters1 = transfer_parent_node_parameter(parameters1, parameters2, parent_node_name)

    noisy_parameters1 = add_noise_to_increments(updated_parameters1)
    model1 = add_arc_to_utility_node(model1, parent_node_name)

    marginal_cpt = calculate_marginal_cpt_dynamic_with_noise(model2, parent_node_name, decision_node_name_model2)
    model1 = transfer_marginal_cpt_multiple_parents(model1, parent_node_name, decision_node_name_model1, marginal_cpt)
    model1 = fill_missing_cpts_with_noise(model1, model2)
    calculate_utility_values(model1, utility_node_name_model1, noisy_parameters1)

    return model1, noisy_parameters1


def add_all_preference_nodes(model1, model2, parameters1, parameters2, log_file=None):

    log = []
    utility_node_model2 = next(
        node for node in model2.names() if model2.isUtilityNode(node)
    )
    utility_parents_model2 = [
        model2.variable(parent).name() for parent in model2.parents(model2.idFromName(utility_node_model2))
    ]

    print(f"Utility node parents in model2: {utility_parents_model2}")

    current_parameters = parameters1

    for parent_node_name in utility_parents_model2:
        print(f"Transferring preference node: {parent_node_name}")
        model1, updated_parameters = preference_transfer(
            model1, model2, current_parameters, parameters2, parent_node_name
        )
        state_utilities, max_eu, max_state=show_decision_utilities3(model1)
        
        limid = gum.ShaferShenoyLIMIDInference(model1)
        limid.setEvidence({})  # Clear all evidence
    
        log.append({
            "Operation": f"Preference node '{parent_node_name}' transferred",        
            "EU": max_eu,
            "State": max_state
        })


        current_parameters = updated_parameters

    if log_file:
        log_df = pd.DataFrame(log)
        log_df.to_excel(log_file, index=False)
    
    return model1, current_parameters, log




def check_missing_cpts(model):
    missing_cpts = []
    for node_id in model.nodes():
        if model.isChanceNode(node_id):   
            try:
                _ = model.cpt(node_id)
            except:
                missing_cpts.append(model.variable(node_id).name())
    return missing_cpts




def show_decision_utilities3(model):
    decision_node_id = next(node for node in model.nodes() if model.isDecisionNode(node))
    decision_node = model.variable(decision_node_id)
    limid = gum.ShaferShenoyLIMIDInference(model)

    state_utilities = {}
    max_expected_utility = float('-inf')
    max_utility_state = None

    decision_domain = decision_node.labels()

    for state in decision_domain:
        limid.setEvidence({decision_node.name(): state})
        limid.makeInference()
        meu = limid.MEU()

        
        expected_utility = meu.get('mean', float('-inf')) if isinstance(meu, dict) else meu
        state_utilities[state] = expected_utility

        if expected_utility > max_expected_utility:
            max_expected_utility = expected_utility
            max_utility_state = state

    return state_utilities, max_expected_utility, max_utility_state


def fill_missing_cpts_with_noise(model1, model2, noise_factor=0.1):

    for node_name in model1.names():
        node_id = model1.idFromName(node_name)
        
        if not model1.isChanceNode(node_id):
            continue
        
        try:
            cpt = model1.cpt(node_id)
            if np.allclose(cpt.tolist(), 0):   
                raise ValueError("CPT is uninitialized or invalid.")
        except Exception:
            try:
                parent_ids = model1.parents(node_id)
                parent_names = [model1.variable(pid).name() for pid in parent_ids]

                cpt = calculate_marginal_cpt_dynamic_with_noise(
                    model2, node_name, parent_names, noise_factor=noise_factor
                )

                model1.cpt(node_id)[:] = cpt
                print(f"Initialized CPT for '{node_name}' using parents: {parent_names}.")
            except Exception as e:
                print(f"Error initializing CPT for '{node_name}': {e}")

    return model1



def iterative_chance_node_transfer_with_logging(model1, model2, chance_node, noise_factor=0.1, log_file=None):
    """
    This function is for simulation data, it logs the operations and utility changes at each step.
    """
    import pandas as pd

    limid = gum.ShaferShenoyLIMIDInference(model1)
    limid.setEvidence({})
    
    log = [] 

    if model1.exists(chance_node):
        print(f"Node '{chance_node}' already exists in model1.")
    
        try:
            node_id_model1 = model1.idFromName(chance_node)
            node_id_model2 = model2.idFromName(chance_node)
            
            states_model1 = model1.variable(node_id_model1).labels()
            states_model2 = model2.variable(node_id_model2).labels()
        except Exception as e:
            print(f"Error accessing states for node '{chance_node}': {e}")
            return model1, [] 
        print(f"Options for chance node '{chance_node}' before updating: {states_model1}")
    
        if states_model1 != states_model2:
            print(f"States mismatch for '{chance_node}'. Synchronizing states...")
            
            parents = model1.parents(node_id_model1)
            children = model1.children(node_id_model1)
         
            model1.erase(node_id_model1)
            
             new_node = gum.LabelizedVariable(chance_node, f"{chance_node} (Synchronized)", len(states_model2))
            for i, label in enumerate(states_model2):
                new_node.changeLabel(i, label)  # Assign each label explicitly
            
             model1.addChanceNode(new_node)
            new_node_id = model1.idFromName(chance_node)
            
            print(f"Node '{chance_node}' states synchronized to: {states_model2}")
            
            for parent_id in parents:
                model1.addArc(parent_id, new_node_id)
            for child_id in children:
                model1.addArc(new_node_id, child_id)
            
            print(f"Re-established arcs for '{chance_node}' with parents {parents} and children {children}")
            
            if len(parents) > 0:
                marginal_cpt = calculate_marginal_cpt_dynamic_with_noise(model2, chance_node, [model2.variable(p).name() for p in parents])
                model1.cpt(chance_node)[:] = marginal_cpt
            else:
                model1.cpt(chance_node).fillWith(1.0 / len(states_model2))  # Uniform distribution if no parents

        marginal_cpt = calculate_marginal_cpt_dynamic_with_noise(model2, chance_node, [], noise_factor=noise_factor)
        model1.cpt(chance_node)[:] = marginal_cpt

        state_utilities, max_eu, max_state = show_decision_utilities3(model1)
        log.append({
            "Operation": f"Node '{chance_node}' synchronized and CPT updated",
            "Parent Node": None,
            "EU": max_eu,
            "State": max_state
        })
        return model1, log

    model1, _ = add_only_node_to_model1(model1, model2, chance_node)

    state_utilities, max_eu, max_state = show_decision_utilities3(model1)
    limid = gum.ShaferShenoyLIMIDInference(model1)
    limid.setEvidence({}) 

    log.append({
        "Operation": f"Node '{chance_node}' transferred",
        "Parent Node": None,
        "EU": max_eu,
        "State": max_state
    })

    chance_node_id = model2.idFromName(chance_node)
    parents_in_model2 = [model2.variable(parent).name() for parent in model2.parents(chance_node_id)]

    transferred_parents = []
    for _ in range(len(parents_in_model2)):
        model1, connected_node = randomly_add_one_parent(model1, model2, chance_node)
        transferred_parents.append(connected_node)
        model1 = fill_missing_cpts_with_noise(model1, model2, noise_factor=noise_factor)
        
        state_utilities, max_eu, max_state = show_decision_utilities3(model1)
        limid = gum.ShaferShenoyLIMIDInference(model1)
        limid.setEvidence({})       
        log.append({
            "Operation": f"Arc added to: '{connected_node}' ",
            "Parent Node": connected_node,
            "EU": max_eu,
            "State": max_state
        })

        marginal_cpt = calculate_marginal_cpt_dynamic_with_noise(model2, chance_node, transferred_parents, noise_factor=noise_factor)
       
        model1 = transfer_marginal_cpt_multiple_parents(model1, chance_node, transferred_parents, marginal_cpt)
        model1 = fill_missing_cpts_with_noise(model1, model2, noise_factor=noise_factor)

        state_utilities, max_eu, max_state = show_decision_utilities3(model1)
      
        limid = gum.ShaferShenoyLIMIDInference(model1)
        limid.setEvidence({})  
    
        log.append({
            "Operation": "Marginal CPT transferred",
            "Parent Node": ", ".join(transferred_parents),
            "EU": max_eu,
            "State": max_state
        })

    if has_empty_cpts(model1):
        model1 = fill_missing_cpts_with_noise(model1, model2, noise_factor=noise_factor)
    
        state_utilities, max_expected_utility, max_utility_state = show_decision_utilities3(model1)
        limid = gum.ShaferShenoyLIMIDInference(model1)
        limid.setEvidence({}) 
    
        log.append({
            "Operation": f"CPTs were transferred to '{chance_node}'",
            "Parent Node": None,
            "EU": max_expected_utility,
            "State": max_utility_state,
        })

    if log_file:
        log_df = pd.DataFrame(log)
        log_df.to_excel(log_file, index=False)

    return model1, log



def transfer_missing_arcs_and_update_cpts(model1, model2, chance_node, noise_factor=0.1):
    """
    This function transfer the arc between the given chance node and its children, if needed.
    """
    chance_node_id_model2 = model2.idFromName(chance_node)

    children_in_model2 = model2.children(chance_node_id_model2)
    children_names_in_model2 = [model2.variable(child).name() for child in children_in_model2]

    for child_name in children_names_in_model2:
        if child_name not in model1.names():
            continue

        if model1.isUtilityNode(model1.idFromName(child_name)):
            print(f"Skipping utility node '{child_name}' during arc transfer.")
            continue

        child_node_id_model1 = model1.idFromName(child_name)

        if not model1.existsArc(model1.idFromName(chance_node), child_node_id_model1):
            try:
                model1.addArc(model1.idFromName(chance_node), child_node_id_model1)
                print(f"Arc added: {chance_node} -> {child_name}")

                relevant_parents = [
                    model1.variable(parent).name()
                    for parent in model1.parents(child_node_id_model1)
                    if model1.exists(model1.variable(parent).name())
                ]

                marginal_cpt = calculate_marginal_cpt_dynamic_with_noise(
                    model2, child_name, relevant_parents, noise_factor=noise_factor
                )

                model1 = transfer_marginal_cpt_multiple_parents(
                    model1, child_name, relevant_parents, marginal_cpt
                )
                print(f"CPT updated for '{child_name}' after adding arc '{chance_node} -> {child_name}'")
            except Exception as e:
                print(f"Error updating CPT for '{child_name}': {e}")

    return model1


def iterative_chance_node_transfer_all(model1, model2, noise_factor=0.1, log_file=None, function_type=None):
    """
    This function is for simulation dataset, it transfers all chance nodes from model2 to model1, and logs each step in an Excel file.
    """
    import pandas as pd

    combined_log = pd.DataFrame()

    all_nodes = model2.names()

    chance_nodes = [
        node_name for node_name in all_nodes
        if model2.isChanceNode(model2.idFromName(node_name))
    ]

    print(f"Chance nodes in model2: {chance_nodes}")

    for chance_node in chance_nodes:
        print(f"Processing chance node: {chance_node}")

        try:
            model1, log = iterative_chance_node_transfer_with_logging(
                model1, model2, chance_node, noise_factor=noise_factor, log_file=None
            )

            for entry in log:
                entry["Function Type"] = function_type

            combined_log = pd.concat([combined_log, pd.DataFrame(log)], ignore_index=True)

        except Exception as e:
            print(f"Error processing node '{chance_node}': {e}")

    delete_unmatched_arcs(model2,model1)
    if log_file:
        combined_log.to_excel(log_file, index=False)

    return model1, combined_log


def has_empty_cpts(model):

    for node in model.names():
        if model.isChanceNode(node):
            cpt = model.cpt(node)
            if cpt.max() == 0 and cpt.min() == 0:
                return True
    return False

def print_all_cpts(model):
    for node_name in model.names():
        if model.isChanceNode(node_name):
            print(f"CPT of {node_name}:")
            print(model.cpt(node_name))
            print("\n-------------------------------\n")



def print_all_utilitiesX(model):
    for node_name in model.names():
        if model.isUtilityNode(node_name):
            print(f"Utility of {node_name}:")
            print(model.utility(node_name))
            print("\n----------------------\n")




def calculate_expected_utility(model):
    state_utilities, max_expected_utility, max_utility_state = show_decision_utilities3(model)
    return max_expected_utility, max_utility_state

def transfer_chance_nodes_with_filter(model1, model2, transfer_type, transfer_value, noise_factor=0.0, log_file=None):
    """
    This function transfer chance nodes from model2 to model1 based on 
    the inputs; transfer_type, transfer_value, noise_factor,
    and logs the expected utility before and after transfer.
    """

    all_nodes = model2.names()
    chance_nodes = [
        node_name for node_name in all_nodes
        if model2.isChanceNode(model2.idFromName(node_name))
    ]

    initial_eu_model1, initial_state_model1 = calculate_expected_utility(model1)
    initial_eu_model2, initial_state_model2 = calculate_expected_utility(model2)

    if transfer_type == "random":
        num_nodes = max(1, int(len(chance_nodes) * transfer_value / 100))  # Transfer % of nodes
        nodes_to_transfer = random.sample(chance_nodes, num_nodes)
    elif transfer_type == "voi":
        voi_sorted_nodes = list(calculate_voi_for_relevant_chance_nodes(model2).keys())
        num_nodes = max(1, int(len(voi_sorted_nodes) * transfer_value / 100))  # Transfer % of nodes by VOI
        nodes_to_transfer = voi_sorted_nodes[:num_nodes]
    else:
        raise ValueError("Invalid transfer_type. Use 'random' or 'voi'.")

    print(f"Nodes to transfer ({transfer_type}): {nodes_to_transfer}")

    combined_log = []
    for chance_node in nodes_to_transfer:
        try:
            model1, log = iterative_chance_node_transfer_with_logging(
                model1, model2, chance_node, noise_factor=noise_factor, log_file=None
            )

            final_eu_model1, final_state_model1 = calculate_expected_utility(model1)
            log_entry = {
                "Number_Transferred_Node": num_nodes,
                "Transferred_Node": chance_node,
                "Initial_EU_Patient": initial_eu_model1,
                "Initial_State_Patient": initial_state_model1,
                "Initial_EU_Doctor": initial_eu_model2,
                "Initial_State_Doctor": initial_state_model2,
                "Final_EU_Patient": final_eu_model1,
                "Final_State_Patient": final_state_model1,
                "Difference_EU_Patient": final_eu_model1 - initial_eu_model1,
            }
            combined_log.append(log_entry)

            initial_eu_model1, initial_state_model1 = final_eu_model1, final_state_model1

        except Exception as e:
            print(f"Error transferring node '{chance_node}': {e}")

    return model1, pd.DataFrame(combined_log)


def add_preference_nodes_with_percentage(model1, model2, parameters1, parameters2, percentage, log_file=None):
    """
    This function transfers a percentage of utility node parents from model2 to model1.
    """
    import pandas as pd
    import random

    log = []
    utility_node_model2 = next(
        node for node in model2.names() if model2.isUtilityNode(node)
    )
    utility_parents_model2 = [
        model2.variable(parent).name() for parent in model2.parents(model2.idFromName(utility_node_model2))
    ]

    print(f"Utility node parents in model2: {utility_parents_model2}")
    print("Nodes in model2:", model2.names()) 
    num_nodes_to_transfer = max(1, int(len(utility_parents_model2) * percentage / 100))
    parents_to_transfer = random.sample(utility_parents_model2, num_nodes_to_transfer)
    print(f"Transferring {num_nodes_to_transfer} nodes: {parents_to_transfer}")

    initial_eu_model1, initial_state_model1 = calculate_expected_utility(model1)

    current_parameters = parameters1

    for parent_node_name in parents_to_transfer:
        print(f"Transferring preference node: {parent_node_name}")
        try:
            if not model2.exists(model2.idFromName(parent_node_name)):
                print(f"Node '{parent_node_name}' does not exist in model2. Skipping.")
                continue

            model1, updated_parameters = preference_transfer(
                model1, model2, current_parameters, parameters2, parent_node_name
            )
            current_parameters = updated_parameters 
            final_eu_model1, final_state_model1 = calculate_expected_utility(model1)
            log.append({
                "Number_Transferred_Node": num_nodes_to_transfer,
                "Transferred_Node": parent_node_name,
                "Initial_EU_Doctor": initial_eu_model1,
                "Initial_State_Doctor": initial_state_model1,
                "Final_EU_Doctor": final_eu_model1,
                "Final_State_Doctor": final_state_model1,
                "Difference_EU_Doctor": final_eu_model1 - initial_eu_model1,
                "Operation": "Preference Node Transfer"
            })

            initial_eu_model1 = final_eu_model1
            initial_state_model1 = final_state_model1

        except Exception as e:
            print(f"Error transferring node '{parent_node_name}': {e}")

    if log_file:
        log_df = pd.DataFrame(log)
        log_df.to_excel(log_file, index=False)

    return model1, current_parameters, pd.DataFrame(log)






