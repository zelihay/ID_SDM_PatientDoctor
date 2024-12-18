import pyAgrum as gum
import pyAgrum.lib.notebook as gnb

def create_patient_model():
    import pyAgrum as gum

    # Step 1: Create a New Influence Diagram for the patient
    patient_model = gum.InfluenceDiagram()

    # Step 2: Add Pretreatment Variables (Chance Nodes)
    age = patient_model.add(gum.LabelizedVariable("Age", "Age of the patient", 3))  # 30-39, 40-49, 50+
    smoking = patient_model.add(gum.LabelizedVariable("Smoking", "Smoking", 2))  # No, Yes
    goiter_size = patient_model.add(gum.LabelizedVariable("GoiterSize", "Goiter Size", 2))  # Small, Large
    lifelong_thyroid_replacement = patient_model.add(gum.LabelizedVariable("Lifelong_Thyroid_Replacement", "Lifelong_Thyroid_Replacement", 2))  # No, Yes

    # Step 3: Add Decision Node (Treatment Choice)
    treatment_choice = patient_model.addDecisionNode(gum.LabelizedVariable("Treatment", "Patient's Treatment Choice", ["ATD", "RAI"]))  # ATD, RAI

    # Step 4: Add Posttreatment Variables (Chance Nodes)
    side_effects = patient_model.add(gum.LabelizedVariable("SideEffects", "Perceived Side Effects", 2))  # No, Yes
    remission = patient_model.add(gum.LabelizedVariable("Remission", "Perceived Remission Outcome", 2))  # No, Yes
    hypothyroidism = patient_model.add(gum.LabelizedVariable("Hypothyroidism", "Perceived Hypothyroidism Risk", 2))  # No, Yes
    cost = patient_model.add(gum.LabelizedVariable("Cost", "Perceived Cost", 3))  # Low, Medium, High

    # Step 5: Add Utility Node (Patient's Satisfaction)
    satisfaction = patient_model.addUtilityNode(gum.LabelizedVariable("Satisfaction", "Patient's Satisfaction", 1))

    # Step 6: Define Dependencies (Arcs)
    patient_model.addArc(age, side_effects)
    patient_model.addArc(smoking, side_effects)
    patient_model.addArc(goiter_size, treatment_choice)
    patient_model.addArc(smoking, treatment_choice)
    patient_model.addArc(treatment_choice, side_effects)
    patient_model.addArc(treatment_choice, remission)
    patient_model.addArc(treatment_choice, hypothyroidism)
    patient_model.addArc(treatment_choice, cost)
    patient_model.addArc(treatment_choice, lifelong_thyroid_replacement)
    patient_model.addArc(remission, satisfaction)
    patient_model.addArc(side_effects, satisfaction)
    patient_model.addArc(hypothyroidism, satisfaction)
    patient_model.addArc(cost, satisfaction)
    patient_model.addArc(lifelong_thyroid_replacement, satisfaction)

    # Step 7: Define CPTs for Each Node
    # CPT for Age
    patient_model.cpt(age)[{}] = [0.40, 0.35, 0.25]  # Probabilities for 30-39, 40-49, 50+

    # CPT for Smoking
    patient_model.cpt(smoking)[{}] = [0.65, 0.35]  # Probabilities for Non-Smoker (No) and Smoker (Yes)

    # CPT for Goiter Size
    patient_model.cpt(goiter_size)[{}] = [0.75, 0.25]  # Small 75%, Large 25%


    # CPT for SideEffects given Treatment, Age, and Smoking
    patient_model.cpt(side_effects)[{'Treatment': 1, 'Age': 0, 'Smoking': 0}] = [0.45, 0.55]
    # Treatment = RAI, Age = 30-39, Smoking = Yes
    patient_model.cpt(side_effects)[{'Treatment': 1, 'Age': 0, 'Smoking': 1}] = [0.30, 0.70]
    # Treatment = RAI, Age = 40-49, Smoking = No
    patient_model.cpt(side_effects)[{'Treatment': 1, 'Age': 1, 'Smoking': 0}] = [0.35, 0.65]
    # Treatment = RAI, Age = 40-49, Smoking = Yes
    patient_model.cpt(side_effects)[{'Treatment': 1, 'Age': 1, 'Smoking': 1}] = [0.20, 0.80]
    # Treatment = RAI, Age = 50+, Smoking = No
    patient_model.cpt(side_effects)[{'Treatment': 1, 'Age': 2, 'Smoking': 0}] = [0.25, 0.75]
    # Treatment = RAI, Age = 50+, Smoking = Yes
    patient_model.cpt(side_effects)[{'Treatment': 1, 'Age': 2, 'Smoking': 1}] = [0.10, 0.90]
    
    # Treatment = RAI, Age = 30-39, Smoking = No

    patient_model.cpt(side_effects)[{'Treatment': 0, 'Age': 0, 'Smoking': 0}] = [0.90, 0.10]
    # Treatment = ATD, Age = 30-39, Smoking = Yes
    patient_model.cpt(side_effects)[{'Treatment': 0, 'Age': 0, 'Smoking': 1}] = [0.80, 0.20]
    # Treatment = ATD, Age = 40-49, Smoking = No
    patient_model.cpt(side_effects)[{'Treatment': 0, 'Age': 1, 'Smoking': 0}] = [0.85, 0.15]
    # Treatment = ATD, Age = 40-49, Smoking = Yes
    patient_model.cpt(side_effects)[{'Treatment': 0, 'Age': 1, 'Smoking': 1}] = [0.75, 0.25]
    # Treatment = ATD, Age = 50+, Smoking = No
    patient_model.cpt(side_effects)[{'Treatment': 0, 'Age': 2, 'Smoking': 0}] = [0.80, 0.20]
    # Treatment = ATD, Age = 50+, Smoking = Yes
    patient_model.cpt(side_effects)[{'Treatment': 0, 'Age': 2, 'Smoking': 1}] = [0.70, 0.30]

    # CPT for Lifelong_Thyroid_Replacement given Treatment
    patient_model.cpt(lifelong_thyroid_replacement)[{'Treatment': 0}] = [0.70, 0.30]  # ATD
    patient_model.cpt(lifelong_thyroid_replacement)[{'Treatment': 1}] = [0.15, 0.85]  # RAI

    # CPT for Remission based on Treatment
    patient_model.cpt(remission)[{'Treatment': 1}] = [0.60, 0.40]  # 
    patient_model.cpt(remission)[{'Treatment': 0}] = [0.20, 0.80]  # 

    # CPT for Hypothyroidism based on Treatment and Goiter Size
    patient_model.cpt(hypothyroidism)[{'Treatment': 1, 'GoiterSize': 0}] = [0.15, 0.75]  # ATD, Small Goiter
    patient_model.cpt(hypothyroidism)[{'Treatment': 1, 'GoiterSize': 1}] = [0.75, 0.25]  # ATD, Large Goiter
    patient_model.cpt(hypothyroidism)[{'Treatment': 0, 'GoiterSize': 0}] = [0.90, 0.10]  # RAI, Small Goiter
    patient_model.cpt(hypothyroidism)[{'Treatment': 0, 'GoiterSize': 1}] = [0.85, 0.15]  # RAI, Large Goiter

    # CPT for Cost based on Treatment
    patient_model.cpt(cost)[{'Treatment': 1}] = [0.40, 0.50, 0.10]  # RAI: Low 40%, Medium 50%, High 10%
    patient_model.cpt(cost)[{'Treatment': 0}] = [0.20, 0.60, 0.20]  # ATD: Low 20%, Medium 60%, High 20%

    return patient_model
