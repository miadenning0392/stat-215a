import numpy as np

from clean_helpers import (
    impute_cat_child_vars,
    parent_cat_child_vars,
    parent_bin_child_vars,
    rename_variables,
    set_variable_types,
    vars_91_92_to_nan
)

def clean_data(
    raw_data, 
    impute_missing_cat_child_vars = "by_common_combination",
    impute_missing_bin_child_vars = True,
    impute_missing_loss_cons_dur = True,
    impute_missing_injury_mech_severity = "by_injury_mech",
    model_preprocessing = True
):        
    """
    Clean and preprocess the raw PECARN TBI dataset.

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Raw input dataframe.

    impute_missing_cat_child_vars: {"by_common_combination", "by_overall_mode", "none"}, default="by_common_combination"
        Strategy used to impute missing values for categorical children of parent-child variables.
        It is specifially for cases where children are not binary.

        - "by_common_combination":
            Fill missing child values based on the most common combination
            among matching observations 
        - "by_overall_mode":
            Fill each missing child using its overall mode for parent = 1
        - "none":
            Leave children missing

    impute_missing_bin_child_vars: bool, default=True
        If True, impute missing values for categorical children of parent-child variables.
        It is specifially for cases where children are binary.
    
    impute_missing_loss_cons_dur: bool, default=True
        If True, impute missing values for loss_consciousness_duration 
        based on the most common value for loss_consciousness = 1 (and 2).
 
    impute_missing_injury_mech_severity: {"by_injury_mech", "as_moderate", "none"}, default="by_injur_mech"
        Strategy used to impute missing values for injury_mech_severity.

        - "by_injury_mech":
            Fill missing values based on injury_mech 
        - "by_overall_mode":
            Fill missing values as moderate
        - "none":
            Leave values missing

    model_preprocessing: bool, default=True
        If True, impute intentionally missing values for children of parent-child variables.
        It is specifially for modelling, where complete cases are required.

    Returns
    -------
    pandas.DataFrame
        Cleaned and preprocessed dataframe.
    """

    clean_data = raw_data.copy()

    # Rename variables
    clean_data = rename_variables(clean_data)

    # Treat 91 and 92 as missing values for specific variables
    clean_data[vars_91_92_to_nan] = clean_data[vars_91_92_to_nan].replace([91, 92], np.nan)

    # Set variable types
    clean_data = set_variable_types(clean_data)

    ############################################################
    # Rename 90 values (which are "other" categories)
    ############################################################
    clean_data["physician_cert"] = clean_data["physician_cert"].cat.rename_categories({90: 5})
    clean_data["race"] = clean_data["race"].cat.rename_categories({90: 6})
    clean_data["ed_disposition"] = clean_data["ed_disposition"].cat.rename_categories({90: 9})
    clean_data["injury_mech"] = clean_data["injury_mech"].cat.rename_categories({90: 13})

    ############################################################
    # Impute missing values for children of parent-child variables
    # Note: specifially for cases where children are not binary 
    ############################################################
    for parent_var, child_vars in parent_cat_child_vars.items():

        clean_data = impute_cat_child_vars(input_data = clean_data,
                                           parent_var = parent_var,
                                           child_vars = child_vars,
                                           impute_method = impute_missing_cat_child_vars)

    ############################################################
    # Impute missing values for children of parent-child variables
    # Note: specifially for cases where children are binary 
    ############################################################
    if impute_missing_bin_child_vars:

        for parent_var, child_vars in parent_bin_child_vars.items():

            for child_var in child_vars:

                clean_data.loc[clean_data[parent_var].eq(1) &
                               clean_data[child_var].isna(),
                               child_var] = 0

    ############################################################
    # Fill in missing values for loss_consciousness_duration 
    # based on the most common value for loss_consciousness = 1 (and 2)
    # Note: theses are a parent-child pairing 
    ############################################################
    if impute_missing_loss_cons_dur:

        lc_dur_mode = (clean_data[clean_data["loss_consciousness"].isin([1, 2])]
                       .groupby("loss_consciousness", observed=True)["loss_consciousness_duration"]
                       .agg(lambda x: x.mode().iloc[0]))

        loc_missing_lc_dur = (clean_data["loss_consciousness"].isin([1, 2]) &
                              clean_data["loss_consciousness_duration"].isna())

        clean_data.loc[loc_missing_lc_dur, "loss_consciousness_duration"] = (
            clean_data.loc[loc_missing_lc_dur, "loss_consciousness"].map(lc_dur_mode)
        )

    ############################################################
    # Impute missing injury_mech_severity values
    ############################################################
    # Fill in missing values for injury_mech_severity based on injury_mech
    if impute_missing_injury_mech_severity == "by_injury_mech":

        clean_data.loc[clean_data["injury_mech_severity"].isna() &
                       clean_data["injury_mech"].isin([2, 3]),
                       "injury_mech_severity"] = 3

        clean_data.loc[clean_data["injury_mech_severity"].isna() &
                       clean_data["injury_mech"].isin([1, 4, 5, 8, 9, 10, 11, 12, 13]),
                       "injury_mech_severity"] = 2

        clean_data.loc[clean_data["injury_mech_severity"].isna() &
                       clean_data["injury_mech"].isin([6, 7]),
                       "injury_mech_severity"] = 1
    
    # Fill in all missing values for injury_mech_severity as moderate (2)
    elif impute_missing_injury_mech_severity == "as_moderate":

        clean_data.loc[clean_data["injury_mech_severity"].isna() &
                       clean_data["injury_mech"].notna(),
                       "injury_mech_severity"] = 2

    # Do not fill missing injury_mech_severity values
    elif impute_missing_injury_mech_severity == "none":
        pass

    ############################################################
    # GCS variables 
    ############################################################
    # Step 1: If exactly one component is missing and total is present,
    # calculate the missing component from total.
    # Constrain calculated values to valid GCS ranges.
    gcs_components = ["gcs_eye", "gcs_verbal", "gcs_motor"]

    gcs_ranges = {
        "gcs_eye": (1, 4),
        "gcs_verbal": (1, 5),
        "gcs_motor": (1, 6)
    }

    one_missing = (clean_data[gcs_components].isna().sum(axis=1) == 1) & clean_data["gcs_total"].notna()

    for col in gcs_components:

        mask = one_missing & clean_data[col].isna()
        other_cols = [x for x in gcs_components if x != col]
        min_val, max_val = gcs_ranges[col]

        clean_data.loc[mask, col] = ((clean_data.loc[mask, "gcs_total"] - clean_data.loc[mask, other_cols].sum(axis = 1))
                                     .clip(lower = min_val, upper = max_val))

    # Step 2: If all three components are present, 
    # recalculate total when it does not equal their sum
    all_components = clean_data[gcs_components].notna().all(axis = 1)

    mismatch = (all_components &
                (clean_data["gcs_total"].isna() |
                (clean_data["gcs_total"] != clean_data[gcs_components].sum(axis = 1))))

    clean_data.loc[mismatch, "gcs_total"] = clean_data.loc[mismatch, gcs_components].sum(axis = 1)

    ############################################################
    # ci_tbi and associated variables 
    ############################################################
    # When all relevant variables are known and none of 
    # them indicate ci_tbi = 1, fill missing ci_tbi with 0
    # (if any revelant variables are unknown, leave ci_tbi as NaN)
    ci_tbi_relevant_vars = [
        "neurosurgery",
        "intubated_greater_24hr",
        "tbi_death",
        "ed_disposition",
        "hospitalized_for_head_tbi"
    ]

    all_rel_vars_known = clean_data[ci_tbi_relevant_vars].notna().all(axis=1)

    no_ci_tbi_conditions = ~(clean_data["neurosurgery"].eq(1)
                             | clean_data["intubated_greater_24hr"].eq(1)
                             | clean_data["tbi_death"].eq(1)
                             | clean_data["ed_disposition"].eq(8)
                             | clean_data["hospitalized_for_head_tbi"].eq(1))

    loc_missing_ci_tbi = (clean_data["ci_tbi"].isna()
                          & all_rel_vars_known
                          & no_ci_tbi_conditions)

    clean_data.loc[loc_missing_ci_tbi, "ci_tbi"] = 0

    # If ci_tbi = 0, fill missing relevant binary variables with 0
    ci_tbi_binary_vars = [
        "neurosurgery",
        "intubated_greater_24hr",
        "tbi_death",
        "hospitalized_for_head_tbi"
    ]

    loc_ci_tbi_zero = clean_data["ci_tbi"].eq(0)

    clean_data.loc[loc_ci_tbi_zero, 
                   ci_tbi_binary_vars] = (clean_data.loc[loc_ci_tbi_zero, ci_tbi_binary_vars].fillna(0))

    ############################################################
    # Impute missing values for children of parent-child variables
    # Specifically for modelling, where complete cases are needed
    ############################################################
    if model_preprocessing:

        for parent_child_dict in [parent_cat_child_vars, parent_bin_child_vars]:

            for parent_var, child_vars in parent_child_dict.items():

                for child_var in child_vars:

                    # Add 0 as a category, if needed
                    if 0 not in clean_data[child_var].cat.categories:
                        clean_data[child_var] = clean_data[child_var].cat.add_categories([0])

                    clean_data.loc[clean_data[parent_var].isin([0, 2]) &
                                   clean_data[child_var].isna(),
                                   child_var] = 0

        # loss_consciousness handled seperately since parent can take on 0, 1, 2
        clean_data["loss_consciousness_duration"] = clean_data["loss_consciousness_duration"].cat.add_categories([0])

        clean_data.loc[clean_data["loss_consciousness"].eq(0) &
                       clean_data["loss_consciousness_duration"].isna(),
                       "loss_consciousness_duration"] = 0

    return clean_data