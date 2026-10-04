import pandas as pd

def impute_cat_child_vars(
    input_data,
    parent_var,
    child_vars,
    impute_method 
):
    """
    Impute missing values for (non-binary) children of parent-child variables.
    """

    original_data = input_data.copy()
    parent_data = original_data.loc[original_data[parent_var] == 1, child_vars]

    ############################################################
    # Method 1: Fill missing child values based on the most
    # common combination among matching observations
    ############################################################
    if impute_method == "by_common_combination":

        # Calculate overall modes for parent = 1
        overall_modes = {}

        for child in child_vars:

            child_mode = parent_data[child].mode()
            overall_modes[child] = child_mode.iloc[0]

        # Loop through rows in original data where parent = 1
        for idx in original_data.index[original_data[parent_var] == 1]:

            # Find which children are missing
            missing_children = [child for child in child_vars
                                if pd.isna(original_data.loc[idx, child])]

            # Find which children are observed
            filled_children = [child for child in child_vars
                               if pd.notna(original_data.loc[idx, child])]

            # Skip if no missing children
            if len(missing_children) == 0:
                continue
        
            # Match using the children that are filled in
            matching_data = parent_data.copy()

            for child in filled_children:

                matching_data = matching_data.loc[matching_data[child] == original_data.loc[idx, child]]

            matching_complete = matching_data.dropna(subset = missing_children)

            # If matching observations exist, use the most common combination of the missing children
            if len(matching_complete) > 0:

                combination_counts = matching_complete[missing_children].value_counts()
                most_common_combination = combination_counts.index[0]

                if len(missing_children) == 1:
                    input_data.loc[idx, missing_children[0]] = most_common_combination
                else:
                    for child, value in zip(missing_children, most_common_combination):
                        input_data.loc[idx, child] = value

            # If no matching observations exist, use the overall modes for parent = 1
            else:

                for child in missing_children:
                    input_data.loc[idx, child] = overall_modes[child]

    ############################################################
    # Method 2: Fill each missing child using its
    # overall mode for parent = 1
    ############################################################
    elif impute_method == "by_overall_mode":

        for child in child_vars:

            child_mode = parent_data[child].mode()
            child_overall_mode = child_mode.iloc[0]

            input_data.loc[(original_data[parent_var] == 1) &
                           original_data[child].isna(),
                           child] = child_overall_mode

    ############################################################
    # Method 3: Do not fill missing child values
    ############################################################
    elif impute_method == "none":
        pass

    return input_data


parent_cat_child_vars = {
    "seizure": [
        "seizure_occur",
        "seizure_duration"
    ],

    "headache_at_eval": [
        "headache_severity",
        "headache_start"
    ],

    "vomiting": [
        "vomiting_ep_count",
        "vomiting_start",
        "vomiting_last_ep"
    ],

    "hematoma": [
        "hematoma_location",
        "hematoma_size"
    ]
}


parent_bin_child_vars = {
    "skull_fracture": [
        "skull_fracture_depressed"
    ],

    "basilar_skull_fracture": [
        "basilar_skull_fracture_hemo",
        "basilar_skull_fracture_oto",
        "basilar_skull_fracture_peri_ecc",
        "basilar_skull_fracture_retro_ecc",
        "basilar_skull_fracture_rhin"
    ],

    "trauma_above_clavical": [
        "trauma_above_clavical_face",
        "trauma_above_clavical_neck",
        "trauma_above_clavical_scalp_frontal",
        "trauma_above_clavical_scalp_occipital",
        "trauma_above_clavical_scalp_parietal",
        "trauma_above_clavical_scalp_temporal"
    ],

    "neuro_deficit": [
        "neuro_deficit_motor",
        "neuro_deficit_sensory",
        "neuro_deficit_cranial",
        "neuro_deficit_reflex",
        "neuro_deficit_other"
    ],

    "other_sub_injuries": [
        "other_sub_injuries_extremity",
        "other_sub_injuries_cut",
        "other_sub_injuries_spine",
        "other_sub_injuries_flank",
        "other_sub_injuries_abdominal",
        "other_sub_injuries_pelvis",
        "other_sub_injuries_other"
    ],

    "alt_mental_stat": [
        "alt_mental_stat_agitated",
        "alt_mental_stat_sleepy",
        "alt_mental_stat_slow",
        "alt_mental_stat_repetitive",
        "alt_mental_stat_other"
    ]
}


def rename_variables(input_data):
    """
    Rename columns to make them human-readable and snake case.
    """

    rename_dict = {
        "PatNum": "patient_number",
        "EmplType": "physician_position",
        "Certification": "physician_cert",
        "InjuryMech": "injury_mech",
        "High_impact_InjSev": "injury_mech_severity",
        "Amnesia_verb": "event_amnesia",
        "LOCSeparate": "loss_consciousness",
        "LocLen": "loss_consciousness_duration",
        "Seiz": "seizure",
        "SeizOccur": "seizure_occur",
        "SeizLen": "seizure_duration",
        "ActNorm": "acting_normal",
        "HA_verb": "headache_at_eval",
        "HASeverity": "headache_severity",
        "HAStart": "headache_start",
        "Vomit": "vomiting",
        "VomitNbr": "vomiting_ep_count",
        "VomitStart": "vomiting_start",
        "VomitLast": "vomiting_last_ep",
        "SFxPalp": "skull_fracture",
        "SFxPalpDepress": "skull_fracture_depressed",
        "SFxBas": "basilar_skull_fracture",
        "SFxBasHem": "basilar_skull_fracture_hemo",
        "SFxBasOto": "basilar_skull_fracture_oto",
        "SFxBasPer": "basilar_skull_fracture_peri_ecc",
        "SFxBasRet": "basilar_skull_fracture_retro_ecc",
        "SFxBasRhi": "basilar_skull_fracture_rhin",
        "Hema": "hematoma",
        "HemaLoc": "hematoma_location",
        "HemaSize": "hematoma_size",
        "Clav": "trauma_above_clavical",
        "ClavFace": "trauma_above_clavical_face",
        "ClavNeck": "trauma_above_clavical_neck",
        "ClavFro": "trauma_above_clavical_scalp_frontal",
        "ClavOcc": "trauma_above_clavical_scalp_occipital",
        "ClavPar": "trauma_above_clavical_scalp_parietal",
        "ClavTem": "trauma_above_clavical_scalp_temporal",
        "NeuroD": "neuro_deficit",
        "NeuroDMotor": "neuro_deficit_motor",
        "NeuroDSensory": "neuro_deficit_sensory",
        "NeuroDCranial": "neuro_deficit_cranial",
        "NeuroDReflex": "neuro_deficit_reflex",
        "NeuroDOth": "neuro_deficit_other",
        "OSI": "other_sub_injuries",
        "OSIExtremity": "other_sub_injuries_extremity",
        "OSICut": "other_sub_injuries_cut",
        "OSICspine": "other_sub_injuries_spine",
        "OSIFlank": "other_sub_injuries_flank",
        "OSIAbdomen": "other_sub_injuries_abdominal",
        "OSIPelvis": "other_sub_injuries_pelvis",
        "OSIOth": "other_sub_injuries_other",
        "GCSEye": "gcs_eye",
        "GCSVerbal": "gcs_verbal",
        "GCSMotor": "gcs_motor",
        "GCSTotal": "gcs_total",
        "GCSGroup": "gcs_group",
        "Dizzy": "dizziness",
        "Intubated": "intubated",
        "Paralyzed": "paralyzed",
        "Sedated": "sedated",
        "FontBulg": "bulging_fontanelle",
        "Drugs": "intoxication",
        "AMS": "alt_mental_stat",
        "AMSAgitated": "alt_mental_stat_agitated",
        "AMSSleep": "alt_mental_stat_sleepy",
        "AMSSlow": "alt_mental_stat_slow",
        "AMSRepeat": "alt_mental_stat_repetitive",
        "AMSOth": "alt_mental_stat_other",
        "CTForm1": "ct_planned",
        "IndAge": "ct_planned_bc_young_age",
        "IndAmnesia": "ct_planned_bc_amnesia",
        "IndAMS": "ct_planned_bc_mental_stat",
        "IndClinSFx": "ct_planned_bc_skull_frac",
        "IndHA": "ct_planned_bc_headache",
        "IndHema": "ct_planned_bc_hematoma",
        "IndLOC": "ct_planned_bc_loss_cons",
        "IndMech": "ct_planned_bc_mech",        
        "IndNeuroD": "ct_planned_bc_neuro_deficit",
        "IndRqstMD": "ct_planned_bc_md_rqst",
        "IndRqstParent": "ct_planned_bc_parent_rqst",
        "IndRqstTrauma": "ct_planned_bc_trauma_team_rqst",
        "IndSeiz": "ct_planned_bc_seizure",
        "IndVomit": "ct_planned_bc_vomiting",
        "IndXraySFx": "ct_planned_bc_xray",
        "IndOth": "ct_planned_bc_other",
        "CTSed": "ct_sedation",
        "CTSedAgitate": "ct_sedation_bc_agitation",
        "CTSedAge": "ct_sedation_bc_young",
        "CTSedRqst": "ct_sedation_bc_tech_rqst",
        "CTSedOth": "ct_sedation_bc_other",
        "AgeInMonth": "age_in_months",
        "AgeinYears": "age_in_years",
        "AgeTwoPlus": "age_less_two",
        "Gender": "gender",
        "Ethnicity": "ethnicity",
        "Race": "race",
        "Observed": "observed_in_ed",
        "EDDisposition": "ed_disposition",
        "CTDone": "ct_performed",
        "EDCT": "ct_performed_in_ed",
        "PosCT": "tbi_on_ct",
        "Finding1": "tbi_finding_cer_hemorrhage",
        "Finding2": "tbi_finding_cer_contusion",
        "Finding3": "tbi_finding_cer_edema",
        "Finding4": "tbi_finding_cer_hematoma",
        "Finding5": "tbi_finding_skull_diastasis",
        "Finding6": "tbi_finding_epi_hematoma",
        "Finding7": "tbi_finding_exax_hematoma",
        "Finding8": "tbi_finding_intra_hemorrhage",
        "Finding9": "tbi_finding_brain_shift",
        "Finding10": "tbi_finding_pneumocephalus",
        "Finding11": "tbi_finding_skull_frac",
        "Finding12": "tbi_finding_sub_hemorrhage",
        "Finding13": "tbi_finding_sub_hematoma",
        "Finding14": "tbi_finding_infraction",
        "Finding20": "tbi_finding_diffuse_axonal",
        "Finding21": "tbi_finding_herniation",
        "Finding22": "tbi_finding_shear",
        "Finding23": "tbi_finding_sinus_thrombosis",
        "DeathTBI": "tbi_death",
        "HospHead": "hospitalized_for_head",
        "HospHeadPosCT": "hospitalized_for_head_tbi",
        "Intub24Head": "intubated_greater_24hr",
        "Neurosurgery": "neurosurgery",
        "PosIntFinal": "ci_tbi"
   }

    output_data = input_data.rename(columns = rename_dict)

    return output_data


def set_variable_types(input_data):
    """
    Set proper variable types.
    """

    categorical_vars = [
        "physician_position",
        "physician_cert",
        "injury_mech",
        "injury_mech_severity",
        "event_amnesia",
        "loss_consciousness",
        "loss_consciousness_duration",
        "seizure",
        "seizure_occur",
        "seizure_duration",
        "acting_normal",
        "headache_at_eval",
        "headache_severity",
        "headache_start",
        "vomiting",
        "vomiting_ep_count",
        "vomiting_start",
        "vomiting_last_ep",
        "skull_fracture",
        "skull_fracture_depressed",
        "basilar_skull_fracture",
        "basilar_skull_fracture_hemo",
        "basilar_skull_fracture_oto",
        "basilar_skull_fracture_peri_ecc",
        "basilar_skull_fracture_retro_ecc",
        "basilar_skull_fracture_rhin",
        "hematoma",
        "hematoma_location",
        "hematoma_size",
        "trauma_above_clavical",
        "trauma_above_clavical_face",
        "trauma_above_clavical_neck",
        "trauma_above_clavical_scalp_frontal",
        "trauma_above_clavical_scalp_occipital",
        "trauma_above_clavical_scalp_parietal",
        "trauma_above_clavical_scalp_temporal",
        "neuro_deficit",
        "neuro_deficit_motor",
        "neuro_deficit_sensory",
        "neuro_deficit_cranial",
        "neuro_deficit_reflex",
        "neuro_deficit_other",
        "other_sub_injuries",
        "other_sub_injuries_extremity",
        "other_sub_injuries_cut",
        "other_sub_injuries_spine",
        "other_sub_injuries_flank",
        "other_sub_injuries_abdominal",
        "other_sub_injuries_pelvis",
        "other_sub_injuries_other",
        "dizziness",
        "intubated",
        "paralyzed",
        "sedated",
        "gcs_group",
        "bulging_fontanelle",
        "intoxication",
        "alt_mental_stat",
        "alt_mental_stat_agitated",
        "alt_mental_stat_sleepy",
        "alt_mental_stat_slow",
        "alt_mental_stat_repetitive",
        "alt_mental_stat_other",
        "ct_planned",
        "ct_planned_bc_young_age",
        "ct_planned_bc_amnesia",
        "ct_planned_bc_mental_stat",
        "ct_planned_bc_skull_frac",
        "ct_planned_bc_headache",
        "ct_planned_bc_hematoma",
        "ct_planned_bc_loss_cons",
        "ct_planned_bc_mech",        
        "ct_planned_bc_neuro_deficit",
        "ct_planned_bc_md_rqst",
        "ct_planned_bc_parent_rqst",
        "ct_planned_bc_trauma_team_rqst",
        "ct_planned_bc_seizure",
        "ct_planned_bc_vomiting",
        "ct_planned_bc_xray",
        "ct_planned_bc_other",
        "ct_sedation",
        "ct_sedation_bc_agitation",
        "ct_sedation_bc_young",
        "ct_sedation_bc_tech_rqst",
        "ct_sedation_bc_other",
        "age_less_two",
        "gender",
        "ethnicity",
        "race",
        "observed_in_ed",
        "ed_disposition",
        "ct_performed",
        "ct_performed_in_ed",
        "tbi_on_ct",
        "tbi_finding_cer_hemorrhage",
        "tbi_finding_cer_contusion",
        "tbi_finding_cer_edema",
        "tbi_finding_cer_hematoma",
        "tbi_finding_skull_diastasis",
        "tbi_finding_epi_hematoma",
        "tbi_finding_exax_hematoma",
        "tbi_finding_intra_hemorrhage",
        "tbi_finding_brain_shift",
        "tbi_finding_pneumocephalus",
        "tbi_finding_skull_frac",
        "tbi_finding_sub_hemorrhage",
        "tbi_finding_sub_hematoma",
        "tbi_finding_infraction",
        "tbi_finding_diffuse_axonal",
        "tbi_finding_herniation",
        "tbi_finding_shear",
        "tbi_finding_sinus_thrombosis",
        "tbi_death",
        "hospitalized_for_head",
        "hospitalized_for_head_tbi",
        "intubated_greater_24hr",
        "neurosurgery",
        "ci_tbi"
    ]

    integer_vars = [
        "gcs_eye",
        "gcs_verbal",
        "gcs_motor",
        "gcs_total",
        "age_in_months",
        "age_in_years"
    ]

    # Convert to (integer) categories
    for col in categorical_vars:
        input_data[col] = input_data[col].astype("Int64").astype("category")

    # Convert to integers
    input_data[integer_vars] = input_data[integer_vars].astype("Int64")

    return input_data


vars_91_92_to_nan = [
    "event_amnesia",
    "loss_consciousness_duration",
    "seizure_occur",
    "seizure_duration",
    "headache_at_eval",
    "headache_severity",
    "headache_start",
    "vomiting_ep_count",
    "vomiting_start",
    "vomiting_last_ep",
    "skull_fracture_depressed",
    "basilar_skull_fracture_hemo",
    "basilar_skull_fracture_oto",
    "basilar_skull_fracture_peri_ecc",
    "basilar_skull_fracture_retro_ecc",
    "basilar_skull_fracture_rhin",
    "hematoma_location",
    "hematoma_size",
    "trauma_above_clavical_face",
    "trauma_above_clavical_neck",
    "trauma_above_clavical_scalp_frontal",
    "trauma_above_clavical_scalp_occipital",
    "trauma_above_clavical_scalp_parietal",
    "trauma_above_clavical_scalp_temporal",
    "neuro_deficit_motor",
    "neuro_deficit_sensory",
    "neuro_deficit_cranial",
    "neuro_deficit_reflex",
    "neuro_deficit_other",
    "other_sub_injuries_extremity",
    "other_sub_injuries_cut",
    "other_sub_injuries_spine",
    "other_sub_injuries_flank",
    "other_sub_injuries_abdominal",
    "other_sub_injuries_pelvis",
    "other_sub_injuries_other",
    "alt_mental_stat_agitated",
    "alt_mental_stat_sleepy",
    "alt_mental_stat_slow",
    "alt_mental_stat_repetitive",
    "alt_mental_stat_other",
    "ct_planned_bc_young_age",
    "ct_planned_bc_amnesia",
    "ct_planned_bc_mental_stat",
    "ct_planned_bc_skull_frac",
    "ct_planned_bc_headache",
    "ct_planned_bc_hematoma",
    "ct_planned_bc_loss_cons",
    "ct_planned_bc_mech",        
    "ct_planned_bc_neuro_deficit",
    "ct_planned_bc_md_rqst",
    "ct_planned_bc_parent_rqst",
    "ct_planned_bc_trauma_team_rqst",
    "ct_planned_bc_seizure",
    "ct_planned_bc_vomiting",
    "ct_planned_bc_xray",
    "ct_planned_bc_other",
    "ct_sedation",
    "ct_sedation_bc_agitation",
    "ct_sedation_bc_young",
    "ct_sedation_bc_tech_rqst",
    "ct_sedation_bc_other",
    "ct_performed_in_ed",
    "tbi_on_ct",
    "tbi_finding_cer_hemorrhage",
    "tbi_finding_cer_contusion",
    "tbi_finding_cer_edema",
    "tbi_finding_cer_hematoma",
    "tbi_finding_skull_diastasis",
    "tbi_finding_epi_hematoma",
    "tbi_finding_exax_hematoma",
    "tbi_finding_intra_hemorrhage",
    "tbi_finding_brain_shift",
    "tbi_finding_pneumocephalus",
    "tbi_finding_skull_frac",
    "tbi_finding_sub_hemorrhage",
    "tbi_finding_sub_hematoma",
    "tbi_finding_infraction",
    "tbi_finding_diffuse_axonal",
    "tbi_finding_herniation",
    "tbi_finding_shear",
    "tbi_finding_sinus_thrombosis"
]