

import pyAgrum as gum

  
#------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------DOCTOR MODEL------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
def create_doctor_model():
    import pyAgrum as gum
    
    # Create a new influence diagram model
    doctor_model = gum.InfluenceDiagram()
        
    age = doctor_model.add(gum.LabelizedVariable("Age", "Age of the patient", 3))  
    
    smoking = doctor_model.add(gum.LabelizedVariable("Smoking", "Smoking", 2))  # No, Yes
    goiter_size = doctor_model.add(gum.LabelizedVariable("GoiterSize", "Goiter Size", 2))  # Small, Large
    
    
    Eye_Disease_Status = doctor_model.add(gum.LabelizedVariable("Eye_Disease_Status", "Eye_Disease_Status", 3))  # low, Moderate, severe
    tyroid_eye_disease = doctor_model.add(gum.LabelizedVariable("Tyroid_Eye_Disease", "Tyroid_Eye_Disease", 2))  # No, Yes
    
   # Lifelong_Thyroid_Replacement = doctor_model.add(gum.LabelizedVariable("Lifelong_Thyroid_Replacement", "Lifelong_Thyroid_Replacement", 2))  # No, Yes
    
    Thyroid_Function_Status = doctor_model.add(gum.LabelizedVariable("Thyroid_Function_Status", "Thyroid_Function_Status", 3)) #Normal , Overactive, Underactive 
    
    Duration_of_Treatment = doctor_model.add(gum.LabelizedVariable("Duration_of_Treatment", "Duration_of_Treatment", 3)) #Short , Medium, Long 
    
    cardiovascular_risk = doctor_model.add(gum.LabelizedVariable("CardioRisk", "Cardiovascular Risk", 2))  # No, Yes
    General_Health_Status = doctor_model.add(gum.LabelizedVariable("General_Health_Status", "General Health Status", 3))  # Good, Average, Poor
    
    TSH_Level = doctor_model.add(gum.LabelizedVariable("TSH_Level", "General TSH_Level Status", 3)) 
    #cost = doctor_model.add(gum.LabelizedVariable("cost", "cost", 3))
    
    risk_of_relapse = doctor_model.add(gum.LabelizedVariable("RelapseRisk", "Risk of Relapse", 2))  # Low, High
    
    severity_hypothyroidism = doctor_model.add(gum.LabelizedVariable("Severity_Hypothyroidism", "Severity_Hypothyroidism", 2))  # No, Yes
    
    treatment = doctor_model.addDecisionNode(gum.LabelizedVariable("Treatment", "Chosen Treatment", ["ATD", "RAI", "Surgery"]))  # ATD, RAI, Surgery
    
    #remission = doctor_model.add(gum.LabelizedVariable("Remission", "Treatment Remission Outcome", 2))  # No, Yes
    #side_effects = doctor_model.add(gum.LabelizedVariable("SideEffects", "Severe Side Effects", 2))  # No, Yes
    #voice_change = doctor_model.add(gum.LabelizedVariable("VoiceChange", "Permanent Voice Change", 2))  # No, Yes
    #Hypothyroidism = doctor_model.add(gum.LabelizedVariable("Hypothyroidism", "Hypothyroidism", 2))  # No, Yes
    hypothyroidism = doctor_model.add(gum.LabelizedVariable("Hypothyroidism", "Hypothyroidism Risk", 2))  # No, Yes
    satisfaction = doctor_model.addUtilityNode(gum.LabelizedVariable("Satisfaction", "Patient Satisfaction", 1))  # Low, Medium, High
    
    doctor_model.addArc(age, General_Health_Status)
    doctor_model.addArc(age, goiter_size)
    
    doctor_model.addArc(General_Health_Status,cardiovascular_risk)
    doctor_model.addArc(smoking,General_Health_Status)
    doctor_model.addArc(smoking,cardiovascular_risk)
    doctor_model.addArc(age,cardiovascular_risk)
   # doctor_model.addArc(age,satisfaction)
    
    
    doctor_model.addArc(goiter_size, risk_of_relapse)
    doctor_model.addArc(goiter_size, severity_hypothyroidism)
    doctor_model.addArc(cardiovascular_risk, treatment)
    #doctor_model.addArc(goiter_size, remission)
    #doctor_model.addArc(goiter_size, treatment)
    #doctor_model.addArc(age,treatment)
    #doctor_model.addArc(smoking,treatment)
    #doctor_model.addArc(General_Health_Status, treatment)
    doctor_model.addArc(treatment,risk_of_relapse)
    
    
    doctor_model.addArc(severity_hypothyroidism, treatment)
    doctor_model.addArc(Duration_of_Treatment,TSH_Level )
    doctor_model.addArc(TSH_Level,Thyroid_Function_Status )
    
    
    #doctor_model.addArc(treatment, remission)
   # doctor_model.addArc(treatment, side_effects)
   # doctor_model.addArc(treatment, cost)
    #doctor_model.addArc(treatment, voice_change)
    #doctor_model.addArc(treatment, Hypothyroidism)
    doctor_model.addArc(smoking,tyroid_eye_disease)
    doctor_model.addArc(Eye_Disease_Status, tyroid_eye_disease)
    #doctor_model.addArc(treatment, Lifelong_Thyroid_Replacement)
    doctor_model.addArc(treatment, Thyroid_Function_Status)
    #doctor_model.addArc(Eye_Disease_Status, treatment)
    
    doctor_model.addArc(treatment, Duration_of_Treatment)
   # doctor_model.addArc(Duration_of_Treatment,remission )
    doctor_model.addArc(Duration_of_Treatment, Thyroid_Function_Status)
    
    #doctor_model.addArc(smoking,remission)
    #doctor_model.addArc(smoking,side_effects)
    
    #doctor_model.addArc(Eye_Disease_Status,side_effects)
    doctor_model.addArc(treatment, tyroid_eye_disease)
   # doctor_model.addArc(remission, satisfaction)
    #doctor_model.addArc(side_effects, satisfaction)
    #doctor_model.addArc(voice_change, satisfaction)
    #doctor_model.addArc(hypocalcemia, satisfaction)
    doctor_model.addArc(risk_of_relapse, satisfaction)
    doctor_model.addArc(Thyroid_Function_Status, satisfaction)
    doctor_model.addArc(tyroid_eye_disease, satisfaction)
    doctor_model.addArc(treatment, satisfaction)
    doctor_model.addArc(treatment, hypothyroidism)
    doctor_model.addArc(hypothyroidism, satisfaction)
    
    ## CPTS based on Table 1 | Advantages and disadvantages of treatments for Graves disease
    #Diagnosis and management of Graves disease: a global overview

    # CPT for Age Node
    doctor_model.cpt(age)[{}] = [0.50, 0.30, 0.20]  # Probabilities for 30-39, 40-49, 50+
    # CPT for Smoking Node
    doctor_model.cpt(smoking)[{}] = [0.65, 0.35]  # Probabilities for Non-Smoker (0) and Smoker (1)
    
    # CPT for Goiter Size Node based on Age
    doctor_model.cpt(goiter_size)[{'Age': 0}] = [0.80, 0.20]  # Age 30-39: Small (80%), Large (20%)
    doctor_model.cpt(goiter_size)[{'Age': 1}] = [0.70, 0.30]  # Age 40-49: Small (70%), Large (30%)
    doctor_model.cpt(goiter_size)[{'Age': 2}] = [0.60, 0.40]  # Age 50+: Small (60%), Large (40%)
    
    
    # CPT for General Health Status Node based on Age and Smoking
    # General Health Status states: Good (0), Average (1), Poor (2)
    
    # Age: 30-39 (0), Non-smoker (0)
    doctor_model.cpt(General_Health_Status)[{'Age': 0, 'Smoking': 0}] = [0.70, 0.20, 0.10]
    # Age: 30-39 (0), Smoker (1)
    doctor_model.cpt(General_Health_Status)[{'Age': 0, 'Smoking': 1}] = [0.50, 0.30, 0.20]
    
    # Age: 40-49 (1), Non-smoker (0)
    doctor_model.cpt(General_Health_Status)[{'Age': 1, 'Smoking': 0}] = [0.50, 0.30, 0.20]
    # Age: 40-49 (1), Smoker (1)
    doctor_model.cpt(General_Health_Status)[{'Age': 1, 'Smoking': 1}] = [0.30, 0.40, 0.30]
    
    # Age: 50+ (2), Non-smoker (0)
    doctor_model.cpt(General_Health_Status)[{'Age': 2, 'Smoking': 0}] = [0.30, 0.40, 0.30]
    # Age: 50+ (2), Smoker (1)
    doctor_model.cpt(General_Health_Status)[{'Age': 2, 'Smoking': 1}] = [0.10, 0.30, 0.60]
    
    # CPT for Cardiovascular Risk based on Age, Smoking, and General Health Status
    # Cardiovascular Risk states: No (0), Yes (1)
    
    # Age 30-39, Non-smoker, Good Health
    # CPT for Cardiovascular Risk based on Age, Smoking, and General Health Status
    # Cardiovascular Risk states: No (0), Yes (1)
    
    # Age: 30-39 (0), Non-smoker (0)
    doctor_model.cpt(cardiovascular_risk)[{'Age': 0, 'Smoking': 0, 'General_Health_Status': 0}] = [0.90, 0.10]  # Good Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 0, 'Smoking': 0, 'General_Health_Status': 1}] = [0.80, 0.20]  # Average Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 0, 'Smoking': 0, 'General_Health_Status': 2}] = [0.60, 0.40]  # Poor Health
    
    # Age: 30-39 (0), Smoker (1)
    doctor_model.cpt(cardiovascular_risk)[{'Age': 0, 'Smoking': 1, 'General_Health_Status': 0}] = [0.70, 0.30]  # Good Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 0, 'Smoking': 1, 'General_Health_Status': 1}] = [0.60, 0.40]  # Average Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 0, 'Smoking': 1, 'General_Health_Status': 2}] = [0.40, 0.60]  # Poor Health
    
    # Age: 40-49 (1), Non-smoker (0)
    doctor_model.cpt(cardiovascular_risk)[{'Age': 1, 'Smoking': 0, 'General_Health_Status': 0}] = [0.80, 0.20]  # Good Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 1, 'Smoking': 0, 'General_Health_Status': 1}] = [0.70, 0.30]  # Average Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 1, 'Smoking': 0, 'General_Health_Status': 2}] = [0.50, 0.50]  # Poor Health
    
    # Age: 40-49 (1), Smoker (1)
    doctor_model.cpt(cardiovascular_risk)[{'Age': 1, 'Smoking': 1, 'General_Health_Status': 0}] = [0.60, 0.40]  # Good Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 1, 'Smoking': 1, 'General_Health_Status': 1}] = [0.50, 0.50]  # Average Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 1, 'Smoking': 1, 'General_Health_Status': 2}] = [0.30, 0.70]  # Poor Health
    
    # Age: 50+ (2), Non-smoker (0)
    doctor_model.cpt(cardiovascular_risk)[{'Age': 2, 'Smoking': 0, 'General_Health_Status': 0}] = [0.70, 0.30]  # Good Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 2, 'Smoking': 0, 'General_Health_Status': 1}] = [0.60, 0.40]  # Average Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 2, 'Smoking': 0, 'General_Health_Status': 2}] = [0.40, 0.60]  # Poor Health
    
    # Age: 50+ (2), Smoker (1)
    doctor_model.cpt(cardiovascular_risk)[{'Age': 2, 'Smoking': 1, 'General_Health_Status': 0}] = [0.50, 0.50]  # Good Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 2, 'Smoking': 1, 'General_Health_Status': 1}] = [0.40, 0.60]  # Average Health
    doctor_model.cpt(cardiovascular_risk)[{'Age': 2, 'Smoking': 1, 'General_Health_Status': 2}] = [0.20, 0.80]  # Poor Health
    
    # CPT for Severity of Hypothyroidism based on Goiter Size
    # Goiter Size: Small (0), Large (1)
    # Severity of Hypothyroidism: No (0), Yes (1)
    
    # Small goiter
    doctor_model.cpt(severity_hypothyroidism)[{'GoiterSize': 0}] = [0.80, 0.20]  # 80% chance of No, 20% chance of Yes
    
    # Large goiter
    doctor_model.cpt(severity_hypothyroidism)[{'GoiterSize': 1}] = [0.40, 0.60]  # 40% chance of No, 60% chance of Yes
    
    # CPT for Relapse Risk based on Goiter Size and Treatment Choice
    # Relapse Risk states: Low (0), High (1)
    # Treatment Choice: ATD (0), RAI (1), Surgery (2)
    # Goiter Size: Small (0), Large (1)
    
    # ATD treatment
    #doctor_model.cpt(risk_of_relapse)[{'Treatment': 0, 'GoiterSize': 0}] = [0.50, 0.50]  # Small goiter, 50% chance of relapse
    #doctor_model.cpt(risk_of_relapse)[{'Treatment': 0, 'GoiterSize': 1}] = [0.30, 0.70]  # Large goiter, 70% chance of relapse

    doctor_model.cpt(risk_of_relapse)[{'Treatment': 0, 'GoiterSize': 0}] = [0.40, 0.60]  # Increase relapse risk likelihood
    doctor_model.cpt(risk_of_relapse)[{'Treatment': 0, 'GoiterSize': 1}] = [0.30, 0.70]

    # RAI treatment
    doctor_model.cpt(risk_of_relapse)[{'Treatment': 1, 'GoiterSize': 0}] = [0.80, 0.20]  # Small goiter, 20% chance of relapse
    doctor_model.cpt(risk_of_relapse)[{'Treatment': 1, 'GoiterSize': 1}] = [0.70, 0.30]  # Large goiter, 30% chance of relapse
    
    # Surgery treatment
    doctor_model.cpt(risk_of_relapse)[{'Treatment': 2, 'GoiterSize': 0}] = [0.95, 0.05]  # Small goiter, 5% chance of relapse
    doctor_model.cpt(risk_of_relapse)[{'Treatment': 2, 'GoiterSize': 1}] = [0.90, 0.10]  # Large goiter, 10% chance of relapse
    
    
    # CPT for Eye Disease Status with Three States: Low, Moderate, Severe
    # Eye Disease Status: Low (0), Moderate (1), Severe (2)
    
    doctor_model.cpt(Eye_Disease_Status)[{}] = [0.60, 0.25, 0.15]  # Low 60%, Moderate 25%, Severe 15%
    
    # CPT for Thyroid Eye Disease based on Smoking, Eye Disease Status, and Treatment Choice
    # Thyroid Eye Disease states: No (0), Yes (1)
    
    # ATD treatment, non-smoker, low initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 0, 'Smoking': 0, 'Eye_Disease_Status': 0}] = [0.60, 0.40]  # Low chance of TED with ATD and no smoking
    # ATD treatment, non-smoker, moderate initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 0, 'Smoking': 0, 'Eye_Disease_Status': 1}] = [0.80, 0.20]  # Moderate TED chance
        # ATD treatment, non-smoker, moderate initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 0, 'Smoking': 0, 'Eye_Disease_Status': 2}] = [0.70, 0.30]  # Moderate TED chance
        
    # ATD treatment, smoker, low initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 0, 'Smoking': 1, 'Eye_Disease_Status': 0}] = [0.70, 0.30]  # Higher chance of TED due to smoking
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 0, 'Smoking': 1, 'Eye_Disease_Status': 1}] = [0.40, 0.60]  # Higher chance of TED due to smoking   
    # ATD treatment, smoker, severe initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 0, 'Smoking': 1, 'Eye_Disease_Status': 2}] = [0.20, 0.80]  # Equal chance of persistent or worsening TED
    
    # RAI treatment, non-smoker, low initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 1, 'Smoking': 0, 'Eye_Disease_Status': 0}] = [0.80, 0.20]  # Moderate TED chance due to RAI
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 1, 'Smoking': 0, 'Eye_Disease_Status': 1}] = [0.70, 0.30]  # Moderate TED chance due to RAI
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 1, 'Smoking': 0, 'Eye_Disease_Status': 2}] = [0.60, 0.40]  # Moderate TED chance due to RAI
          
    # RAI treatment, smoker, low initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 1, 'Smoking': 1, 'Eye_Disease_Status': 0}] = [0.60, 0.40]  # Higher chance of TED due to RAI and smoking    
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 1, 'Smoking': 1, 'Eye_Disease_Status': 1}] = [0.30, 0.70]  # Higher chance of TED due to RAI and smoking    
    # RAI treatment, smoker, severe initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 1, 'Smoking': 1, 'Eye_Disease_Status': 2}] = [0.10, 0.90]  # Very high chance of persistent or worsening TED
    
    # Surgery treatment, non-smoker, moderate initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 2, 'Smoking': 0, 'Eye_Disease_Status': 0}] = [0.7, 0.3]  # Lower chance of TED due to definitive treatment
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 2, 'Smoking': 0, 'Eye_Disease_Status': 1}] = [0.6, 0.4]  # Lower chance of TED due to definitive treatment   
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 2, 'Smoking': 0, 'Eye_Disease_Status': 2}] = [0.5, 0.5]  # Lower chance of TED due to definitive treatment       
    # Surgery treatment, smoker, severe initial eye disease
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 2, 'Smoking': 1, 'Eye_Disease_Status': 0}] = [0.5, 0.5]  # Lower chance of TED due to definitive treatment
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 2, 'Smoking': 1, 'Eye_Disease_Status': 1}] = [0.2, 0.8]  # Lower chance of TED due to definitive treatment   
    doctor_model.cpt(tyroid_eye_disease)[{'Treatment': 2, 'Smoking': 1, 'Eye_Disease_Status': 2}] = [0.1, 0.9]  # Lower chance of TED due to definitive treatment     
    
    
    # CPT for Side Effects based on Eye Disease Status, Smoking, and Treatment Choice
    # Side Effects states: No (0), Yes (1)
    
    # ATD treatment, non-smoker, low initial eye disease
    #doctor_model.cpt(side_effects)[{'Treatment': 0, 'Smoking': 0, 'Eye_Disease_Status': 0}] = [0.95, 0.05]  # Low chance of side effects with ATD
    
    # ATD treatment, smoker, low initial eye disease
    #doctor_model.cpt(side_effects)[{'Treatment': 0, 'Smoking': 1, 'Eye_Disease_Status': 0}] = [0.90, 0.10]  # Slightly higher risk due to smoking
    
    # ATD treatment, smoker, severe initial eye disease
   # doctor_model.cpt(side_effects)[{'Treatment': 0, 'Smoking': 1, 'Eye_Disease_Status': 2}] = [0.85, 0.15]  # Higher chance due to smoking and severe eye disease
    
    # RAI treatment, non-smoker, low initial eye disease
  #  doctor_model.cpt(side_effects)[{'Treatment': 1, 'Smoking': 0, 'Eye_Disease_Status': 0}] = [0.85, 0.15]  # Moderate chance of side effects with RAI
    
    # RAI treatment, smoker, severe initial eye disease
   # doctor_model.cpt(side_effects)[{'Treatment': 1, 'Smoking': 1, 'Eye_Disease_Status': 2}] = [0.50, 0.50]  # High risk due to RAI, smoking, and severe eye disease
    
    # Surgery treatment, non-smoker, moderate initial eye disease
    #doctor_model.cpt(side_effects)[{'Treatment': 2, 'Smoking': 0, 'Eye_Disease_Status': 1}] = [0.80, 0.20]  # Moderate chance of surgical side effects
    
    # Surgery treatment, smoker, severe initial eye disease
  #  doctor_model.cpt(side_effects)[{'Treatment': 2, 'Smoking': 1, 'Eye_Disease_Status': 2}] = [0.60, 0.40]  # Higher chance due to smoking and severe eye disease
    
    # RAI treatment, non-smoker, moderate initial eye disease
   # doctor_model.cpt(side_effects)[{'Treatment': 1, 'Smoking': 0, 'Eye_Disease_Status': 1}] = [0.75, 0.25]  # Moderate risk of side effects
    
    
    # CPT for Duration of Treatment based on Treatment Choice
    # Duration of Treatment states: Short (0), Medium (1), Long (2)
    # Treatment Choice: ATD (0), RAI (1), Surgery (2)
    
    # ATD treatment: most patients require a long treatment duration
    doctor_model.cpt(Duration_of_Treatment)[{'Treatment': 0}] = [0.10, 0.20, 0.70]  # Short 10%, Medium 20%, Long 70%
    
    # RAI treatment: typically medium duration but can be short or require follow-up
    doctor_model.cpt(Duration_of_Treatment)[{'Treatment': 1}] = [0.30, 0.60, 0.10]  # Short 30%, Medium 60%, Long 10%
    
    # Surgery treatment: shorter active treatment phase but immediate and long-term follow-up
    doctor_model.cpt(Duration_of_Treatment)[{'Treatment': 2}] = [0.70, 0.20, 0.10]  # Short 70%, Medium 20%, Long 10%
    
    # CPT for Hypothyroidism based on Treatment Choice
    # Hypothyroidism states: No (0), Yes (1)
    # Treatment Choice: ATD (0), RAI (1), Surgery (2)
    
    # ATD treatment: lower likelihood of hypothyroidism
    #doctor_model.cpt(Hypothyroidism)[{'Treatment': 0}] = [0.70, 0.30]  # No 70%, Yes 30%
    
    # RAI treatment: high likelihood of hypothyroidism
   # doctor_model.cpt(Hypothyroidism)[{'Treatment': 1}] = [0.15, 0.85]  # No 15%, Yes 85%
    
    # Surgery treatment: nearly all patients develop hypothyroidism
   # doctor_model.cpt(Hypothyroidism)[{'Treatment': 2}] = [0.02, 0.98]  # No 2%, Yes 98%
    
    # CPT for Lifelong Thyroid Replacement based on Treatment Choice
    # Lifelong Thyroid Replacement: No (0), Yes (1)
    # Treatment Choice: ATD (0), RAI (1), Surgery (2)
    
    # ATD treatment: lower chance of needing lifelong thyroid replacement
    #doctor_model.cpt(Lifelong_Thyroid_Replacement)[{'Treatment': 0}] = [0.70, 0.30]  # No 70%, Yes 30%
    
    # RAI treatment: high chance of needing lifelong thyroid replacement
  #  doctor_model.cpt(Lifelong_Thyroid_Replacement)[{'Treatment': 1}] = [0.15, 0.85]  # No 15%, Yes 85%
    
    # Surgery treatment: almost all require lifelong replacement
  #  doctor_model.cpt(Lifelong_Thyroid_Replacement)[{'Treatment': 2}] = [0.02, 0.98]  # No 2%, Yes 98%
    
    
    # CPT for Voice Change based on Treatment Choice
    # Voice Change: No (0), Yes (1)
    # Treatment Choice: ATD (0), RAI (1), Surgery (2)
    
    # ATD treatment: very low risk of voice change
  #  doctor_model.cpt(voice_change)[{'Treatment': 0}] = [0.99, 0.01]  # No 99%, Yes 1%
    
    # RAI treatment: low risk of voice change
  #  doctor_model.cpt(voice_change)[{'Treatment': 1}] = [0.98, 0.02]  # No 98%, Yes 2%
    
    # Surgery treatment: higher risk of voice change
 #   doctor_model.cpt(voice_change)[{'Treatment': 2}] = [0.90, 0.10]  # No 90%, Yes 10%
    
    # CPT for Cost based on Treatment Choice
    # Cost: Low (0), Medium (1), High (2)
    # Treatment Choice: ATD (0), RAI (1), Surgery (2)
    
    # ATD treatment: typically lower cost but can be medium if extended
   # doctor_model.cpt(cost)[{'Treatment': 0}] = [0.60, 0.30, 0.10]  # Low 60%, Medium 30%, High 10%
    
    # RAI treatment: moderate cost due to initial treatment and follow-up
  #  doctor_model.cpt(cost)[{'Treatment': 1}] = [0.20, 0.60, 0.20]  # Low 20%, Medium 60%, High 20%
    
    # Surgery treatment: higher cost due to surgical fees and hospital stays
  #  doctor_model.cpt(cost)[{'Treatment': 2}] = [0.05, 0.20, 0.75]  # Low 5%, Medium 20%, High 75%
    
    # CPT for TSH Level based on Duration of Treatment
    # TSH Level states: Low (0), Normal (1), High (2)
    # Duration of Treatment: Short (0), Medium (1), Long (2)
    
    # Short duration: TSH is more likely to be abnormal (low or high)
    doctor_model.cpt(TSH_Level)[{'Duration_of_Treatment': 0}] = [0.40, 0.30, 0.30]  # Low 40%, Normal 30%, High 30%
    
    # Medium duration: TSH is more likely to be normal as the treatment takes effect
    doctor_model.cpt(TSH_Level)[{'Duration_of_Treatment': 1}] = [0.20, 0.60, 0.20]  # Low 20%, Normal 60%, High 20%
    
    # Long duration: TSH is most likely normalized, with some risk of overtreatment leading to high TSH
    doctor_model.cpt(TSH_Level)[{'Duration_of_Treatment': 2}] = [0.10, 0.70, 0.20]  # Low 10%, Normal 70%, High 20%
    
    # CPT for Remission based on Treatment Choice, Smoking, Goiter Size, and Duration of Treatment
    # Remission states: No (0), Yes (1)
    
    # ATD treatment (0)
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 0, 'GoiterSize': 0, 'Duration_of_Treatment': 0}] = [0.80, 0.20]  # Short duration, small goiter, non-smoker
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 0, 'GoiterSize': 0, 'Duration_of_Treatment': 1}] = [0.60, 0.40]  # Medium duration
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 0, 'GoiterSize': 0, 'Duration_of_Treatment': 2}] = [0.40, 0.60]  # Long duration
    
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 0, 'GoiterSize': 1, 'Duration_of_Treatment': 0}] = [0.90, 0.10]  # Short duration, large goiter
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 0, 'GoiterSize': 1, 'Duration_of_Treatment': 1}] = [0.75, 0.25]  # Medium duration
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 0, 'GoiterSize': 1, 'Duration_of_Treatment': 2}] = [0.60, 0.40]  # Long duration
    
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 1, 'GoiterSize': 0, 'Duration_of_Treatment': 0}] = [0.85, 0.15]  # Smoker, small goiter
   # doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 1, 'GoiterSize': 0, 'Duration_of_Treatment': 1}] = [0.70, 0.30]  # Medium duration
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 1, 'GoiterSize': 0, 'Duration_of_Treatment': 2}] = [0.50, 0.50]  # Long duration
    
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 1, 'GoiterSize': 1, 'Duration_of_Treatment': 0}] = [0.95, 0.05]  # Smoker, large goiter
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 1, 'GoiterSize': 1, 'Duration_of_Treatment': 1}] = [0.80, 0.20]  # Medium duration
    #doctor_model.cpt(remission)[{'Treatment': 0, 'Smoking': 1, 'GoiterSize': 1, 'Duration_of_Treatment': 2}] = [0.70, 0.30]  # Long duration
    
    # RAI treatment (1)
    #doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 0, 'GoiterSize': 0, 'Duration_of_Treatment': 0}] = [0.40, 0.60]  # Non-smoker, small goiter, short duration
    #doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 0, 'GoiterSize': 0, 'Duration_of_Treatment': 1}] = [0.30, 0.70]  # Medium duration
    #doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 0, 'GoiterSize': 0, 'Duration_of_Treatment': 2}] = [0.20, 0.80]  # Long duration
    
    #doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 0, 'GoiterSize': 1, 'Duration_of_Treatment': 0}] = [0.60, 0.40]  # Large goiter
    #doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 0, 'GoiterSize': 1, 'Duration_of_Treatment': 1}] = [0.50, 0.50]  # Medium duration
   # doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 0, 'GoiterSize': 1, 'Duration_of_Treatment': 2}] = [0.40, 0.60]  # Long duration
    
    #doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 1, 'GoiterSize': 0, 'Duration_of_Treatment': 0}] = [0.60, 0.40]  # Smoker, small goiter
    #doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 1, 'GoiterSize': 0, 'Duration_of_Treatment': 1}] = [0.50, 0.50]  # Medium duration
   # doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 1, 'GoiterSize': 0, 'Duration_of_Treatment': 2}] = [0.30, 0.70]  # Long duration
    
    #doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 1, 'GoiterSize': 1, 'Duration_of_Treatment': 0}] = [0.70, 0.30]  # Large goiter
  #  doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 1, 'GoiterSize': 1, 'Duration_of_Treatment': 1}] = [0.60, 0.40]  # Medium duration
  #  doctor_model.cpt(remission)[{'Treatment': 1, 'Smoking': 1, 'GoiterSize': 1, 'Duration_of_Treatment': 2}] = [0.50, 0.50]  # Long duration
    
    # Surgery treatment (2)
   # doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 0, 'GoiterSize': 0, 'Duration_of_Treatment': 0}] = [0.02, 0.98]  # Non-smoker, small goiter
  #  doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 0, 'GoiterSize': 0, 'Duration_of_Treatment': 1}] = [0.02, 0.98]  # Medium duration
  #  doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 0, 'GoiterSize': 0, 'Duration_of_Treatment': 2}] = [0.02, 0.98]  # Long duration
    
   # doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 0, 'GoiterSize': 1, 'Duration_of_Treatment': 0}] = [0.05, 0.95]  # Large goiter
   # doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 0, 'GoiterSize': 1, 'Duration_of_Treatment': 1}] = [0.04, 0.96]  # Medium duration
   # doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 0, 'GoiterSize': 1, 'Duration_of_Treatment': 2}] = [0.03, 0.97]  # Long duration
    
    #doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 1, 'GoiterSize': 0, 'Duration_of_Treatment': 0}] = [0.05, 0.95]  # Smoker, small goiter
   # doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 1, 'GoiterSize': 0, 'Duration_of_Treatment': 1}] = [0.04, 0.96]  # Medium duration
    #doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 1, 'GoiterSize': 0, 'Duration_of_Treatment': 2}] = [0.03, 0.97]  # Long duration
    
   # doctor_model.cpt(remission)[{'Treatment': 2, 'Smoking': 1, 'GoiterSize': 1, 'Duration_of_Treatment': 0}]
    
    # CPT for Thyroid Function Status based on TSH Level, Duration of Treatment, and Treatment Choice
    # Thyroid Function Status states: Overactive (0), Normal (1), Underactive (2)
    
    # ATD treatment
    doctor_model.cpt(Thyroid_Function_Status)[{'TSH_Level': 0, 'Duration_of_Treatment': 0, 'Treatment': 0}] = [0.30, 0.30, 0.40]  # Short duration, low TSH
    doctor_model.cpt(Thyroid_Function_Status)[{'TSH_Level': 1, 'Duration_of_Treatment': 1, 'Treatment': 0}] = [0.20, 0.50, 0.30]  # Medium duration, normal TSH
    doctor_model.cpt(Thyroid_Function_Status)[{'TSH_Level': 2, 'Duration_of_Treatment': 2, 'Treatment': 0}] = [0.05, 0.15, 0.80]  # Long duration, high TSH
    
    # RAI treatment
    doctor_model.cpt(Thyroid_Function_Status)[{'TSH_Level': 0, 'Duration_of_Treatment': 0, 'Treatment': 1}] = [0.50, 0.40, 0.10]  # Short duration, low TSH
    doctor_model.cpt(Thyroid_Function_Status)[{'TSH_Level': 1, 'Duration_of_Treatment': 1, 'Treatment': 1}] = [0.10, 0.60, 0.30]  # Medium duration, normal TSH
    doctor_model.cpt(Thyroid_Function_Status)[{'TSH_Level': 2, 'Duration_of_Treatment': 2, 'Treatment': 1}] = [0.05, 0.10, 0.85]  # Long duration, high TSH
    
    # Surgery treatment
    doctor_model.cpt(Thyroid_Function_Status)[{'TSH_Level': 0, 'Duration_of_Treatment': 0, 'Treatment': 2}] = [0.40, 0.50, 0.10]  # Short duration, low TSH
    doctor_model.cpt(Thyroid_Function_Status)[{'TSH_Level': 1, 'Duration_of_Treatment': 1, 'Treatment': 2}] = [0.05, 0.30, 0.65]  # Medium duration, normal TSH
    doctor_model.cpt(Thyroid_Function_Status)[{'TSH_Level': 2, 'Duration_of_Treatment': 2, 'Treatment': 2}] = [0.02, 0.08, 0.90]  # Long duration, high TSH
    
    doctor_model.cpt(hypothyroidism)[{'Treatment': 0}] = [0.15, 0.75]   
    doctor_model.cpt(hypothyroidism)[{'Treatment': 1}] = [0.75, 0.25]   
    doctor_model.cpt(hypothyroidism)[{'Treatment': 2}] = [0.90, 0.10]  
   
    # Return the influence diagram model
    return doctor_model





