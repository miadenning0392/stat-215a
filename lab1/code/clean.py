#import libraries
import numpy as np
import pandas as pd

#general cleaning functions
def clean_gated_detail(detail_col, gate_col, value_map, real_answer_values, special_labels=None):
    """Clean a 'daughter' column that only has a real answer when its 'parent'
    column indicates the question applied to this patient."""
    if special_labels is None:
        special_labels = {}
    out = detail_col.replace(value_map)
    for gate_value, label in special_labels.items():
        out = out.mask(gate_col == gate_value, label)
    is_real_answer = gate_col.isin(real_answer_values)
    is_special = gate_col.isin(special_labels.keys())
    needs_nan = ~is_real_answer & ~is_special
    out = out.mask(needs_nan, np.nan)
    return out

def clean_gated_family(cleaned, df, gate_name, gate_map, daughters, relabel_gate=True):
    if relabel_gate:
        cleaned[gate_name] = df[gate_name].map(gate_map)

    for d in daughters:
        output_name = d.get("output_name", d["name"])
        cleaned[output_name] = clean_gated_detail(
            df[d["name"]], df[gate_name],
            value_map=d["value_map"],
            real_answer_values=d["real_answer_values"],
            special_labels=d.get("special_labels"),
        )
        if output_name != d["name"]:
            cleaned.drop(columns=d["name"], inplace=True)
    return cleaned


#check cleaning function
def verify_gated_cleaning(cleaned_col, detail_col, gate_col, real_answer_values, special_labels=None, name=""):
    """Check, not required for cleaning itself. Only run when verbose=True."""
    special_labels = special_labels or {}
    ok = True
    for gate_value, label in special_labels.items():
        expected = (gate_col == gate_value).sum()
        actual = (cleaned_col == label).sum()
        match = expected == actual
        ok = ok and match
        if not match:
            print(f"{name}: '{label}' expected {expected}, got {actual} -> MISMATCH")
    is_real_answer = gate_col.isin(real_answer_values)
    is_special = gate_col.isin(special_labels.keys())
    expected_nan = (~is_real_answer & ~is_special).sum() + detail_col[is_real_answer].isna().sum()
    actual_nan = cleaned_col.isna().sum()
    nan_match = expected_nan == actual_nan
    ok = ok and nan_match
    if not nan_match:
        print(f"{name}: NaN expected {expected_nan}, got {actual_nan} -> MISMATCH")
    if ok:
        print(f"{name}: all checks passed")
    return ok


#family-level cleaning function
def clean_data(df, verbose=False):
    cleaned = df.copy()

    #Headache family
    clean_gated_family(
        cleaned, df,
        gate_name="HA_verb",
        gate_map={0: "No", 1: "Yes", 91: "Pre-verbal/Non-verbal"},
        daughters=[
            {"name": "HASeverity",
             "value_map": {1: "Mild", 2: "Moderate", 3: "Severe"},
             "real_answer_values": [1], #HA_verb == 1 (Yes)
             "special_labels": {0: "No headache", 91: "Pre-verbal/Non-verbal"}},
            {"name": "HAStart",
             "value_map": {1: "Before head injury", 2: "Within 1 hr of event",
                           3: "1 - 4 hrs after event", 4: " > 4 hrs after event"},
             "real_answer_values": [1],
             "special_labels": {0: "No headache", 91: "Pre-verbal/Non-verbal"}},
        ],
    )

    #Loss of consciousness family
    clean_gated_family(
        cleaned, df,
        gate_name="LOCSeparate",
        gate_map={0: "No", 1: "Yes", 2: "Suspected"},
        daughters=[
            {"name": "LocLen",
             "value_map": {1: "< 5 sec", 2: "5 sec - < 1 min", 3: "1 -5 min", 4: "> 5 min"},
             "real_answer_values": [1, 2],  #LOCSeparate == 1 (Yes) or 2 (Suspected) both real answers
             "special_labels": {0: "No loss of consciousness"}},
        ],
    )
    #Seizures family
    clean_gated_family(
        cleaned, df,
        gate_name="Seiz",
        gate_map={0: "No", 1: "Yes"},
        daughters=[
            {"name": "SeizOccur",
             "value_map": {1: "Immediately on contact", 2: "Within 30 minutes of injury", 3: "> 30 minutes after injury"},
             "real_answer_values": [1],  #Seiz == 1 means (Yes)
             "special_labels": {0: "No seizure"}},
            {"name": "SeizLen",
             "value_map": {1: "< 1 min", 2: "1 - < 5 min", 3: "5 - 15 min", 4: "> 15 min"},
             "real_answer_values": [1],
             "special_labels": {0: "No seizure"}},
        ],
    )

    #Vomiting family
    clean_gated_family(
        cleaned, df,
        gate_name="Vomit",
        gate_map={0: "No", 1: "Yes"},
        daughters=[
            {"name": "VomitNbr",
             "value_map": {1: "Once", 2: "Twice", 3: "> 2 times"},
             "real_answer_values": [1],  #Vomit == 1 (Yes)
             "special_labels": {0: "No vomiting"}},
            {"name": "VomitStart",
             "value_map": {1: "Before head injury", 2: "Within 1 hr of event",
                           3: "1 - 4 hrs after event", 4: "> 4 hrs after event"},
             "real_answer_values": [1],
             "special_labels": {0: "No vomiting"}},
            {"name": "VomitLast",
             "value_map": {1: "< 1 hr before ED evaluation", 2: "1 - 4 hrs before ED evaluation",
                           3: "> 4 hrs before ED evaluation"},
             "real_answer_values": [1],
             "special_labels": {0: "No vomiting"}},
        ],
    )

    #Hematoma family
    clean_gated_family(
        cleaned, df,
        gate_name="Hema",
        gate_map={0: "No", 1: "Yes"},
        daughters=[
            {"name": "HemaLoc",
             "value_map": {1: "Frontal", 2: "Occipital", 3: "Parietal/Temporal"},
             "real_answer_values": [1],  #Hema == 1 (Yes)
             "special_labels": {0: "No hematoma"}},
            {"name": "HemaSize",
             "value_map": {1: "Small (<1 cm, barely palpable)", 2: "Medium (1-3 cm)", 3: "Large (>3 cm)"},
             "real_answer_values": [1],
             "special_labels": {0: "No hematoma"}},
        ],
    )

    #Basilar skull fracture family
    clean_gated_family(
        cleaned, df,
        gate_name="SFxBas",
        gate_map={0: "No", 1: "Yes"},
        daughters=[
            {"name": "SFxBasHem",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],  #SFxBas == 1 (Yes)
             "special_labels": {0: "No signs of basilar skull fracture"}},
            {"name": "SFxBasOto",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No signs of basilar skull fracture"}},
            {"name": "SFxBasPer",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No signs of basilar skull fracture"}},
            {"name": "SFxBasRet",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No signs of basilar skull fracture"}},
            {"name": "SFxBasRhi",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No signs of basilar skull fracture"}},
        ],
    )

    #Trauma above the clavicles family
    clean_gated_family(
        cleaned, df,
        gate_name="Clav",
        gate_map={0: "No", 1: "Yes"},
        daughters=[
            {"name": "ClavFace",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],  #Clav == 1 (Yes)
             "special_labels": {0: "No trauma above the clavicles"}},
            {"name": "ClavNeck",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No trauma above the clavicles"}},
            {"name": "ClavFro",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No trauma above the clavicles"}},
            {"name": "ClavOcc",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No trauma above the clavicles"}},
            {"name": "ClavPar",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No trauma above the clavicles"}},
            {"name": "ClavTem",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No trauma above the clavicles"}},
        ],
    )

    #Neurological deficit family
    clean_gated_family(
        cleaned, df,
        gate_name="NeuroD",
        gate_map={0: "No", 1: "Yes"},
        daughters=[
            {"name": "NeuroDMotor",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],  #NeuroD == 1 (Yes)
             "special_labels": {0: "No neurological deficit"}},
            {"name": "NeuroDSensory",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No neurological deficit"}},
            {"name": "NeuroDCranial",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No neurological deficit"}},
            {"name": "NeuroDReflex",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No neurological deficit"}},
            {"name": "NeuroDOth",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No neurological deficit"}},
        ],
    )

    #Other substantial injury family
    clean_gated_family(
        cleaned, df,
        gate_name="OSI",
        gate_map={0: "No", 1: "Yes"},
        daughters=[
            {"name": "OSIExtremity",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],  #OSI == 1 (Yes)
             "special_labels": {0: "No other substantial injury"}},
            {"name": "OSICut",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No other substantial injury"}},
            {"name": "OSICspine",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No other substantial injury"}},
            {"name": "OSIFlank",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No other substantial injury"}},
            {"name": "OSIAbdomen",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No other substantial injury"}},
            {"name": "OSIPelvis",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No other substantial injury"}},
            {"name": "OSIOth",
             "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1],
             "special_labels": {0: "No other substantial injury"}},
        ],
    )

    #CT indication family
    clean_gated_family(
        cleaned, df,
        gate_name="CTForm1",
        gate_map={0: "No", 1: "Yes"},
        daughters=[
            {"name": "IndAge", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndAmnesia", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndAMS", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndClinSFx", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndHA", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndHema", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndLOC", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndMech", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndNeuroD", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndRqstMD", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndRqstParent", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndRqstTrauma", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndSeiz", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndVomit", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndXraySFx", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "IndOth", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
            {"name": "CTSed", "value_map": {0: "No", 1: "Yes"},
             "real_answer_values": [1], "special_labels": {0: "CT not ordered/obtained"}},
        ],
    )

    #CT sedation reason family
    clean_gated_family(
    cleaned, df,
    gate_name="CTSed",
    gate_map={0: "No", 1: "Yes"}, 
    daughters=[
        {"name": "CTSedAgitate", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No sedation given"}},
        {"name": "CTSedAge", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No sedation given"}},
        {"name": "CTSedRqst", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No sedation given"}},
        {"name": "CTSedOth", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No sedation given"}},
    ],
    relabel_gate=False,
    )

    #Altered mental status family
    clean_gated_family(cleaned, df, gate_name="AMS", gate_map={0: "No", 1: "Yes"},
    daughters=[
        {"name": "AMSAgitated", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No signs of altered mental status"}},
        {"name": "AMSSleep", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No signs of altered mental status"}},
        {"name": "AMSSlow", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No signs of altered mental status"}},
        {"name": "AMSRepeat", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No signs of altered mental status"}},
        {"name": "AMSOth", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No signs of altered mental status"}},
        ],
    )

    #Palpable skull fracture family
    clean_gated_family(cleaned, df, gate_name="SFxPalp", gate_map={0: "No", 1: "Yes", 2: "Unclear exam"},
    daughters=[
        {"name": "SFxPalpDepress", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1],
         "special_labels": {0: "No palpable skull fracture", 2: "Unclear exam"}},
        ],
    )

    #CT done family
    clean_gated_family(cleaned, df, gate_name="CTDone", gate_map={0: "No", 1: "Yes"},
    daughters=[
        {"name": "EDCT", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "PosCT", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding1", "output_name": "CerebellarHemorrhage", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding2", "output_name": "CerebralContusion", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding3", "output_name": "CerebralEdema", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding4", "output_name": "CerebralHemorrhage", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding5", "output_name": "DiastasisSkull", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding6", "output_name": "EpiduralHematoma", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding7", "output_name": "Extra-axialHematoma", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding8", "output_name": "IntraventricularHemorrhage", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding9", "output_name": "MidlineShift", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding10", "output_name": "Pneumocephalus", "value_map": {0: "No", 1: "Yes"},"real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding11", "output_name": "SkullFracture", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding12", "output_name": "SubarachnoidHemorrhage", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding13", "output_name": "SubduralHematoma", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding14", "output_name": "TraumaticInfarction", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding20", "output_name": "DiffuseAxonalInjury", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding21", "output_name": "Herniation", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding22", "output_name": "ShearInjury", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
        {"name": "Finding23", "output_name": "SigmoidSinusThrombosis", "value_map": {0: "No", 1: "Yes"}, "real_answer_values": [1], "special_labels": {0: "No CT performed"}},
    ])

    #verify cleaning if verbose
    if verbose:
        verify_gated_cleaning(cleaned["HASeverity"], df["HASeverity"], df["HA_verb"],
                               real_answer_values=[1], special_labels={0: "No headache", 91: "Pre-verbal/Non-verbal"},
                               name="HASeverity")
        verify_gated_cleaning(cleaned["HAStart"], df["HAStart"], df["HA_verb"],
                               real_answer_values=[1], special_labels={0: "No headache", 91: "Pre-verbal/Non-verbal"},
                               name="HAStart")

        verify_gated_cleaning(cleaned["LocLen"], df["LocLen"], df["LOCSeparate"],
                               real_answer_values=[1, 2], special_labels={0: "No loss of consciousness"},
                               name="LocLen")

        verify_gated_cleaning(cleaned["SeizOccur"], df["SeizOccur"], df["Seiz"],
                               real_answer_values=[1], special_labels={0: "No seizure"},
                               name="SeizOccur")
        verify_gated_cleaning(cleaned["SeizLen"], df["SeizLen"], df["Seiz"],
                               real_answer_values=[1], special_labels={0: "No seizure"},
                               name="SeizLen")

        verify_gated_cleaning(cleaned["VomitNbr"], df["VomitNbr"], df["Vomit"],
                               real_answer_values=[1], special_labels={0: "No vomiting"},
                               name="VomitNbr")
        verify_gated_cleaning(cleaned["VomitStart"], df["VomitStart"], df["Vomit"],
                               real_answer_values=[1], special_labels={0: "No vomiting"},
                               name="VomitStart")
        verify_gated_cleaning(cleaned["VomitLast"], df["VomitLast"], df["Vomit"],
                               real_answer_values=[1], special_labels={0: "No vomiting"},
                               name="VomitLast")

        verify_gated_cleaning(cleaned["HemaLoc"], df["HemaLoc"], df["Hema"],
                               real_answer_values=[1], special_labels={0: "No hematoma"},
                               name="HemaLoc")
        verify_gated_cleaning(cleaned["HemaSize"], df["HemaSize"], df["Hema"],
                               real_answer_values=[1], special_labels={0: "No hematoma"},
                               name="HemaSize")
        
        verify_gated_cleaning(cleaned["SFxBasHem"], df["SFxBasHem"], df["SFxBas"],
                               real_answer_values=[1], special_labels={0: "No signs of basilar skull fracture"},
                               name="SFxBasHem")
        verify_gated_cleaning(cleaned["SFxBasOto"], df["SFxBasOto"], df["SFxBas"],
                               real_answer_values=[1], special_labels={0: "No signs of basilar skull fracture"},
                               name="SFxBasOto")
        verify_gated_cleaning(cleaned["SFxBasPer"], df["SFxBasPer"], df["SFxBas"],
                               real_answer_values=[1], special_labels={0: "No signs of basilar skull fracture"},
                               name="SFxBasPer")
        verify_gated_cleaning(cleaned["SFxBasRet"], df["SFxBasRet"], df["SFxBas"],
                               real_answer_values=[1], special_labels={0: "No signs of basilar skull fracture"},
                               name="SFxBasRet")
        verify_gated_cleaning(cleaned["SFxBasRhi"], df["SFxBasRhi"], df["SFxBas"],
                               real_answer_values=[1], special_labels={0: "No signs of basilar skull fracture"},
                               name="SFxBasRhi")
        
        verify_gated_cleaning(cleaned["ClavFace"], df["ClavFace"], df["Clav"],
                               real_answer_values=[1], special_labels={0: "No trauma above the clavicles"},
                               name="ClavFace")
        verify_gated_cleaning(cleaned["ClavNeck"], df["ClavNeck"], df["Clav"],
                               real_answer_values=[1], special_labels={0: "No trauma above the clavicles"},
                               name="ClavNeck")
        verify_gated_cleaning(cleaned["ClavFro"], df["ClavFro"], df["Clav"],
                               real_answer_values=[1], special_labels={0: "No trauma above the clavicles"},
                               name="ClavFro")
        verify_gated_cleaning(cleaned["ClavOcc"], df["ClavOcc"], df["Clav"],
                               real_answer_values=[1], special_labels={0: "No trauma above the clavicles"},
                               name="ClavOcc")
        verify_gated_cleaning(cleaned["ClavPar"], df["ClavPar"], df["Clav"],
                               real_answer_values=[1], special_labels={0: "No trauma above the clavicles"},
                               name="ClavPar")
        verify_gated_cleaning(cleaned["ClavTem"], df["ClavTem"], df["Clav"],
                               real_answer_values=[1], special_labels={0: "No trauma above the clavicles"},
                               name="ClavTem")

        verify_gated_cleaning(cleaned["OSIExtremity"], df["OSIExtremity"], df["OSI"],
                               real_answer_values=[1], special_labels={0: "No other substantial injury"},
                               name="OSIExtremity")
        verify_gated_cleaning(cleaned["OSICut"], df["OSICut"], df["OSI"],
                               real_answer_values=[1], special_labels={0: "No other substantial injury"},
                               name="OSICut")
        verify_gated_cleaning(cleaned["OSICspine"], df["OSICspine"], df["OSI"],
                               real_answer_values=[1], special_labels={0: "No other substantial injury"},
                               name="OSICspine")
        verify_gated_cleaning(cleaned["OSIFlank"], df["OSIFlank"], df["OSI"],
                               real_answer_values=[1], special_labels={0: "No other substantial injury"},
                               name="OSIFlank")
        verify_gated_cleaning(cleaned["OSIAbdomen"], df["OSIAbdomen"], df["OSI"],
                               real_answer_values=[1], special_labels={0: "No other substantial injury"},
                               name="OSIAbdomen")
        verify_gated_cleaning(cleaned["OSIPelvis"], df["OSIPelvis"], df["OSI"],
                               real_answer_values=[1], special_labels={0: "No other substantial injury"},
                               name="OSIPelvis")
        verify_gated_cleaning(cleaned["OSIOth"], df["OSIOth"], df["OSI"],
                               real_answer_values=[1], special_labels={0: "No other substantial injury"},
                               name="OSIOth")

        for ind_col in ["IndAge", "IndAmnesia", "IndAMS", "IndClinSFx", "IndHA", "IndHema",
                        "IndLOC", "IndMech", "IndNeuroD", "IndRqstMD", "IndRqstParent", "IndRqstTrauma",
                        "IndSeiz", "IndVomit", "IndXraySFx", "IndOth"]:
            verify_gated_cleaning(cleaned[ind_col], df[ind_col], df["CTForm1"],
                                   real_answer_values=[1], special_labels={0: "CT not ordered/obtained"},
                                   name=ind_col)

        for sed_col in ["CTSedAgitate", "CTSedAge", "CTSedRqst", "CTSedOth"]:
            verify_gated_cleaning(cleaned[sed_col], df[sed_col], df["CTSed"],
                                   real_answer_values=[1], special_labels={0: "No sedation given"},
                                   name=sed_col)

        for ams_col in ["AMSAgitated", "AMSSleep", "AMSSlow", "AMSRepeat", "AMSOth"]:
            verify_gated_cleaning(cleaned[ams_col], df[ams_col], df["AMS"],
                           real_answer_values=[1], special_labels={0: "No signs of altered mental status"},
                           name=ams_col)

        verify_gated_cleaning(cleaned["SFxPalpDepress"], df["SFxPalpDepress"], df["SFxPalp"],
                       real_answer_values=[1],
                       special_labels={0: "No palpable skull fracture", 2: "Unclear exam"},
                       name="SFxPalpDepress")
        
        verify_gated_cleaning(cleaned["NeuroDMotor"], df["NeuroDMotor"], df["NeuroD"],
                       real_answer_values=[1], special_labels={0: "No neurological deficit"},
                       name="NeuroDMotor")
        verify_gated_cleaning(cleaned["NeuroDSensory"], df["NeuroDSensory"], df["NeuroD"],
                            real_answer_values=[1], special_labels={0: "No neurological deficit"},
                            name="NeuroDSensory")
        verify_gated_cleaning(cleaned["NeuroDCranial"], df["NeuroDCranial"], df["NeuroD"],
                            real_answer_values=[1], special_labels={0: "No neurological deficit"},
                            name="NeuroDCranial")
        verify_gated_cleaning(cleaned["NeuroDReflex"], df["NeuroDReflex"], df["NeuroD"],
                            real_answer_values=[1], special_labels={0: "No neurological deficit"},
                            name="NeuroDReflex")
        verify_gated_cleaning(cleaned["NeuroDOth"], df["NeuroDOth"], df["NeuroD"],
                            real_answer_values=[1], special_labels={0: "No neurological deficit"},
                            name="NeuroDOth")

        verify_gated_cleaning(cleaned["CTSed"], df["CTSed"], df["CTForm1"],
                       real_answer_values=[1], special_labels={0: "CT not ordered/obtained"},
                       name="CTSed")

        for raw_name, output_name in [
            ("EDCT", "EDCT"),
            ("PosCT", "PosCT"),
            ("Finding1", "CerebellarHemorrhage"),
            ("Finding2", "CerebralContusion"),
        ]:
            verify_gated_cleaning(cleaned[output_name], df[raw_name], df["CTDone"],
                                real_answer_values=[1], special_labels={0: "No CT performed"},
                                name=output_name)

    #standalone cleaning for variables with no family structure
    cleaned["ActNorm"] = df["ActNorm"].map({0: "No", 1: "Yes"})
    cleaned["Dizzy"] = df["Dizzy"].map({0: "No", 1: "Yes"})
    cleaned["Intubated"] = df["Intubated"].map({0: "No", 1: "Yes"})
    cleaned["Paralyzed"] = df["Paralyzed"].map({0: "No", 1: "Yes"})
    cleaned["Sedated"] = df["Sedated"].map({0: "No", 1: "Yes"})
    cleaned["Amnesia_verb"] = df["Amnesia_verb"].map({0: "No", 1: "Yes", 91: "Pre-verbal/Non-verbal"})
    cleaned["Race"] = df["Race"].map({1: "White", 2: "Black", 3: "Asian", 4: "American Indian/Alaskan Native", 5: "Pacific Islander", 90: "Other"})
    cleaned["Gender"] = df["Gender"].map({1: "Male", 2: "Female"})
    cleaned["Ethnicity"] = df["Ethnicity"].map({1: "Hispanic", 2: "Non-Hispanic"})
    cleaned["AgeTwoPlus"] = df["AgeTwoPlus"].map({1: "< 2 years", 2: ">= 2 years"})
    cleaned["FontBulg"] = df["FontBulg"].map({0: "No/Closed", 1: "Yes"})
    cleaned["GCSGroup"] = df["GCSGroup"].map({1: "3 - 13", 2: "14 - 15"})
    cleaned["Drugs"] = df["Drugs"].map({0: "No", 1: "Yes"})
    cleaned["GCSEye"] = df["GCSEye"].map({1: "None", 2: "Pain", 3: "Verbal", 4: "Spontaneous"})
    cleaned["GCSVerbal"] = df["GCSVerbal"].map({
        1: "None", 2: "Incomprehensible sounds (moans)", 3: "Inappropriate words (cries to pain)",
        4: "Confused (irritable/cries)", 5: "Oriented (coos/babbles)"
    })
    cleaned["GCSMotor"] = df["GCSMotor"].map({
        1: "None", 2: "Abnormal extension posturing", 3: "Abnormal flexure posturing",
        4: "Withdraws to pain", 5: "Localizes pain (withdraws to touch)", 6: "Follow commands (spontaneous movement)"
    })
    cleaned["EmplType"] = df["EmplType"].map({1: "Nurse Practitioner", 2: "Physician Assistant", 3: "Resident", 4: "Fellow", 5: "Faculty"})
    cleaned["Certification"] = df["Certification"].map({1: "Emergency Medicine", 2: "Pediatrics", 3: "Pediatrics Emergency Medicine", 4: "Emergency Medicine and Pediatrics", 90: "Other"})
    cleaned["InjuryMech"] = df["InjuryMech"].map({
        1: "Occupant in MVC", 2: "Pedestrian struck by moving vehicle", 3: "Bike rider struck by automobile",
        4: "Bike collision or fall from bike while riding", 5: "Other wheeled transport crash",
        6: "Fall to ground from standing/walking/running", 7: "Walked or ran into stationary object",
        8: "Fall from an elevation", 9: "Fall down stairs", 10: "Sports", 11: "Assault",
        12: "Object struck head - accidental", 90: "Other mechanism"
    })
    cleaned["High_impact_InjSev"] = df["High_impact_InjSev"].map({1: "Low", 2: "Moderate", 3: "High"})
    cleaned["Observed"] = df["Observed"].map({0: "No", 1: "Yes"})
    cleaned["EDDisposition"] = df["EDDisposition"].map({1: "Home", 2: "OR", 3: "Admit - general inpatient", 4: "Admit short-stay (< 24 hr)/observation unit", 5: "ICU", 6: "Transferred to another hospital", 7: "AMA", 8: "Death in ED", 90: "Other"})
    cleaned["DeathTBI"] = df["DeathTBI"].map({0: "No", 1: "Yes"})
    cleaned["HospHead"] = df["HospHead"].map({0: "No", 1: "Yes"})
    cleaned["HospHeadPosCT"] = df["HospHeadPosCT"].map({0: "No", 1: "Yes"})
    cleaned["Intub24Head"] = df["Intub24Head"].map({0: "No", 1: "Yes"})
    cleaned["Neurosurgery"] = df["Neurosurgery"].map({0: "No", 1: "Yes"})
    cleaned["PosIntFinal"] = df["PosIntFinal"].map({0: "No", 1: "Yes"})

    return cleaned