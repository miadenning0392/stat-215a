import numpy as np
import pandas as pd
from pathlib import Path

# load dataframe with pandas
# take in raw dataframe and possibly additional parameters and returns the cleaned data as a DataFrame

# TODO:
# We mainly keep the columns relevant to the variables that were used in the clinical decision rules created in the paper
# we do a random split because the time based information is not available in the dataset
# split the data further based on whether the patient is greater than or equal to 2 years old, but this can be done during data exploration and modeling later
# combine unclear and palpable skull fracture into one value


# configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "TBI PUD 10-08-2013.csv"

# PARAMS
TEST_SIZE = 0.2
RANDOM_STATE = 42

# columns to keep for analysis and human-readable names
COLUMNS_TO_KEEP = {
    "PatNum": "patient_id",
    "InjuryMech": "injury_mechanism",
    "High_impact_InjSev": "injury_mechanism_severity",
    "Amnesia_verb": "amnesia",
    "LOCSeparate": "loss_of_consciousness",
    "LocLen": "duration_of_loss_of_consciousness",
    "Seiz": "seizure",
    "SeizOccur": "seizure_occurred_time",
    "SeizLen": "seizure_duration",
    "ActNorm": "acting_normally",
    "HA_verb": "headache",
    "HASeverity": "headache_severity",
    "HAStart": "headache_start_time",
    "Vomit": "vomiting",
    "VomitNbr": "number_of_vomiting_episodes",
    "VomitStart": "vomiting_start_time",
    "VomitLast": "vomiting_last_time",
    "Dizzy": "dizziness",
    "GCSEye": "gcs_eye",
    "GCSVerbal": "gcs_verbal",
    "GCSMotor": "gcs_motor",
    "GCSTotal": "gcs_total",
    "AMS": "altered_mental_status",
    "AMSAgitated": "ams_agitated",
    "AMSSleep": "ams_sleep",
    "AMSSlow": "ams_slow",
    "AMSRepeat": "ams_repeat",
    "AMSOth": "ams_other",
    "SFxPalp": "palpable_skull_fracture",
    "SFxPalpDepress": "palpable_skull_fracture_depressed",
    "SFxBas": "basilar_skull_fracture",
    "SFxBasHem": "basilar_skull_fracture_hemotympanum",
    "SFxBasOto": "basilar_skull_fracture_otorrhea",
    "SFxBasPer": "basilar_skull_fracture_periorbital_ecchymosis",
    "SFxBasRet": "basilar_skull_fracture_retroauricular_ecchymosis",
    "SFxBasRhi": "basilar_skull_fracture_rhinorrhea",
    "Hema": "hematoma",
    "HemaLoc": "hematoma_location",
    "HemaSize": "hematoma_size",
    "AgeInMonth": "age_in_months",
    "Gender": "gender",
    "Ethnicity": "ethnicity",
    "Race": "race",
    "CTDone": "ct_done",
    "PosIntFinal": "clinically_important_tbi",
    "HospHeadPosCT": "ciTBI_1",
    "Intub24Head": "ciTBI_2",
    "Neurosurgery": "ciTBI_3",
    "DeathTBI": "ciTBI_4",
}

# columns where 92 represents "not applicable"
NOT_APPLICABLE_COLUMNS = [
    "duration_of_loss_of_consciousness",
    "seizure_occurred_time",
    "seizure_duration",
    "headache_start_time",
    "headache_severity",
    "vomiting_start_time",
    "vomiting_last_time",
    "number_of_vomiting_episodes",
    "ams_agitated",
    "ams_sleep",
    "ams_slow",
    "ams_repeat",
    "ams_other",
    "palpable_skull_fracture_depressed",
    "basilar_skull_fracture_hemotympanum",
    "basilar_skull_fracture_otorrhea",
    "basilar_skull_fracture_periorbital_ecchymosis",
    "basilar_skull_fracture_retroauricular_ecchymosis",
    "basilar_skull_fracture_rhinorrhea",
    "hematoma_location",
    "hematoma_size",
]

GCS_COLUMNS = ["gcs_eye", "gcs_verbal", "gcs_motor"]


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw TBI PUD dataset from a CSV file.

    Parameters
    ----------
    data_path:
        Path to the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    return pd.read_csv(data_path)


def select_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Subset the dataframe to the variables used in the study and rename them.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing the raw variables.

    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe with only the relevant columns retained
        and renamed to more human-readable names
    """
    df = df[list(COLUMNS_TO_KEEP.keys())].copy()
    df.rename(columns=COLUMNS_TO_KEEP, inplace=True)

    return df


def filter_eligible_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Keep observations with GCS >= 14 and a non-missing outcome."""
    df = df[df["gcs_total"] >= 14].copy()
    df.dropna(subset=["clinically_important_tbi"], inplace=True)

    return df


def clean_gcs(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate the total GCS score when all three component scores exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing eye, verbal, and motor GCS components.

    Returns
    -------
    pd.DataFrame
        The same dataframe with ``gcs_total`` updated to the sum of all three
        component scores when those values are all observed.
    """

    gcs_sum = df[GCS_COLUMNS].sum(axis=1, min_count=3)

    all_subcategories_present = df[GCS_COLUMNS].notna().all(axis=1)

    # for observations with all 3 subcategories present, use their sum as the correct total
    df.loc[all_subcategories_present, "gcs_total"] = gcs_sum[all_subcategories_present]

    return df


def clean_palpable_skull_fracture(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the unclear and palpable skull fracture categories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the skull fracture indicator variable.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe where the value ``2`` (unclear) is recoded to
        ``1`` (palpable) to match the definition used in the analysis.
    """

    df["palpable_skull_fracture"] = df["palpable_skull_fracture"].replace(
        {2: 1}
    )  # combine unclear and palpable skull fracture into one value as that is what the domain knowledge experts did.
    # better to error on the side of caution

    return df


def handle_logically_inconsistent_palpable_skull_fracture(
    df: pd.DataFrame,
    handling: str,
) -> pd.DataFrame:
    """Handle logically inconsistent palpable skull fracture observations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe including the palpable skull fracture fields.
    handling : str
        Strategy for handling inconsistent records. Supported values are
        ``"palpable_skull_fracture_depressed_missing"`` and
        ``"palpable_skull_fracture_missing"``.

    Returns
    -------
    pd.DataFrame
        The dataframe with the specified logical inconsistency handled.
    """

    # argument for handle_logically_inconsistent_palpable_skull_fracture can be "palpable_skull_fracture_depressed_missing" or "palpable_skull_fracture_missing""

    if handling == "palpable_skull_fracture_depressed_missing":

        # if palpable skull fracture is 1, then palpable skull fracture depressed should not be 92 (not applicable)

        df.loc[
            (df["palpable_skull_fracture"] == 1)
            & (df["palpable_skull_fracture_depressed"] == 92),
            "palpable_skull_fracture_depressed",
        ] = np.nan

    elif handling == "palpable_skull_fracture_missing":

        df.loc[
            (df["palpable_skull_fracture"] == 1)
            & (df["palpable_skull_fracture_depressed"] == 92),
            "palpable_skull_fracture",
        ] = np.nan

    return df


def recode_not_applicable_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all ``92`` values with missing values for selected fields.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose not-applicable entries should be converted to ``NaN``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the selected columns recoded so that ``92`` becomes
        missing.
    """
    df[NOT_APPLICABLE_COLUMNS] = df[NOT_APPLICABLE_COLUMNS].replace(
        {92: np.nan}
    )  # recode 92 (not applicable) to NaN

    return df


def clean_loss_of_consciousness(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the duration and missingness encoding for loss-of-consciousness.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the loss-of-consciousness indicator and duration.

    Returns
    -------
    pd.DataFrame
        The dataframe with duration values set to 0 or ``NaN``
    """
    # loss of consciousness cleaning

    # recode 92 (not applicable) to 0 only when loss_of_consciousness = No
    df.loc[
        (df["loss_of_consciousness"] == 0)
        & (df["duration_of_loss_of_consciousness"] == 92),
        "duration_of_loss_of_consciousness",
    ] = 0

    # recode all remaining 92s to Nan
    df.loc[
        df["duration_of_loss_of_consciousness"] == 92,
        "duration_of_loss_of_consciousness",
    ] = np.nan

    return df


def clean_seizure(df: pd.DataFrame) -> pd.DataFrame:
    """Clean seizure timing and duration variables

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing seizure status, timing, and duration fields.

    Returns
    -------
    pd.DataFrame
        DataFrame with seizure timing and duration values recoded
    """
    # seizure cleaning

    df.loc[
        (df["seizure"] == 0) & (df["seizure_occurred_time"] == 92),
        "seizure_occurred_time",
    ] = 4

    df.loc[
        df["seizure_occurred_time"] == 92,
        "seizure_occurred_time",
    ] = np.nan

    df.loc[
        (df["seizure"] == 0) & (df["seizure_duration"] == 92),
        "seizure_duration",
    ] = 0

    df.loc[
        df["seizure_duration"] == 92,
        "seizure_duration",
    ] = np.nan

    return df


def clean_headache(df: pd.DataFrame) -> pd.DataFrame:
    """Clean headache timing and severity

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing headache presence, start time, and severity.

    Returns
    -------
    pd.DataFrame
        DataFrame with headache timing and severity values recoded
    """
    # headache cleaning

    df.loc[
        (df["headache"] == 0) & (df["headache_start_time"] == 92),
        "headache_start_time",
    ] = 0

    df.loc[
        df["headache_start_time"] == 92,
        "headache_start_time",
    ] = np.nan

    df.loc[
        (df["headache"] == 0) & (df["headache_severity"] == 92),
        "headache_severity",
    ] = 0

    df.loc[
        df["headache_severity"] == 92,
        "headache_severity",
    ] = np.nan

    return df


def clean_vomiting(df: pd.DataFrame) -> pd.DataFrame:
    """Clean vomiting timing and frequency fields

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing vomiting presence, timing, and count variables.

    Returns
    -------
    pd.DataFrame
        A dataframe where ``92`` values have been recoded to ``0`` or ``NaN``
    """
    # vomiting cleaning

    df.loc[
        (df["vomiting"] == 0) & (df["vomiting_start_time"] == 92),
        "vomiting_start_time",
    ] = 0

    df.loc[
        df["vomiting_start_time"] == 92,
        "vomiting_start_time",
    ] = np.nan

    df.loc[
        (df["vomiting"] == 0) & (df["vomiting_last_time"] == 92),
        "vomiting_last_time",
    ] = 4

    df.loc[
        df["vomiting_last_time"] == 92,
        "vomiting_last_time",
    ] = np.nan

    df.loc[
        (df["vomiting"] == 0) & (df["number_of_vomiting_episodes"] == 92),
        "number_of_vomiting_episodes",
    ] = 0

    df.loc[
        df["number_of_vomiting_episodes"] == 92,
        "number_of_vomiting_episodes",
    ] = np.nan

    return df


def clean_altered_mental_status(df: pd.DataFrame) -> pd.DataFrame:
    """Clean altered-mental-status subcategory indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with altered mental status flag and subcategory responses.

    Returns
    -------
    pd.DataFrame
        DataFrame with each AMS subcategory recoded
    """
    # altered mental status cleaning

    df.loc[
        (df["altered_mental_status"] == 0) & (df["ams_agitated"] == 92),
        "ams_agitated",
    ] = 2

    df.loc[df["ams_agitated"] == 92, "ams_agitated"] = np.nan

    df.loc[
        (df["altered_mental_status"] == 0) & (df["ams_sleep"] == 92),
        "ams_sleep",
    ] = 2

    df.loc[df["ams_sleep"] == 92, "ams_sleep"] = np.nan

    df.loc[
        (df["altered_mental_status"] == 0) & (df["ams_slow"] == 92),
        "ams_slow",
    ] = 2

    df.loc[df["ams_slow"] == 92, "ams_slow"] = np.nan

    df.loc[
        (df["altered_mental_status"] == 0) & (df["ams_repeat"] == 92),
        "ams_repeat",
    ] = 2

    df.loc[df["ams_repeat"] == 92, "ams_repeat"] = np.nan

    df.loc[
        (df["altered_mental_status"] == 0) & (df["ams_other"] == 92),
        "ams_other",
    ] = 2

    df.loc[df["ams_other"] == 92, "ams_other"] = np.nan

    return df


def clean_skull_fracture_subcategories(df: pd.DataFrame) -> pd.DataFrame:
    """Clean palpable and basilar skull fracture subcategory responses.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing skull fracture indicators and subtype fields.

    Returns
    -------
    pd.DataFrame
        DataFrame with values recoded
    """

    df.loc[
        (df["palpable_skull_fracture"] == 0)
        & (df["palpable_skull_fracture_depressed"] == 92),
        "palpable_skull_fracture_depressed",
    ] = 2

    df.loc[
        df["palpable_skull_fracture_depressed"] == 92,
        "palpable_skull_fracture_depressed",
    ] = np.nan

    basilar_fracture_columns = [
        "basilar_skull_fracture_hemotympanum",
        "basilar_skull_fracture_otorrhea",
        "basilar_skull_fracture_periorbital_ecchymosis",
        "basilar_skull_fracture_retroauricular_ecchymosis",
        "basilar_skull_fracture_rhinorrhea",
    ]

    for column in basilar_fracture_columns:
        df.loc[
            (df["basilar_skull_fracture"] == 0) & (df[column] == 92),
            column,
        ] = 2

        df.loc[df[column] == 92, column] = np.nan

    return df


def clean_hematoma(df: pd.DataFrame) -> pd.DataFrame:
    """Clean hematoma site and size variables

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing hematoma presence, location, and size fields.

    Returns
    -------
    pd.DataFrame
        DataFrame with hematoma-related fields recoded
    """
    df.loc[
        (df["hematoma"] == 0) & (df["hematoma_location"] == 92),
        "hematoma_location",
    ] = 4

    df.loc[
        df["hematoma_location"] == 92,
        "hematoma_location",
    ] = np.nan

    df.loc[
        (df["hematoma"] == 0) & (df["hematoma_size"] == 92),
        "hematoma_size",
    ] = 0

    df.loc[df["hematoma_size"] == 92, "hematoma_size"] = np.nan

    return df


def clean_not_applicable_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all variable-specific cleaning rules for ``92`` values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose variable-specific not-applicable codes need cleaning.

    Returns
    -------
    pd.DataFrame
        DataFrame after the loss-of-consciousness, seizure, headache,
        vomiting, AMS, skull-fracture, and hematoma cleaning steps.
    """
    df = clean_loss_of_consciousness(df)
    df = clean_seizure(df)
    df = clean_headache(df)
    df = clean_vomiting(df)
    df = clean_altered_mental_status(df)
    df = clean_skull_fracture_subcategories(df)
    df = clean_hematoma(df)

    return df


def create_age_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Convert patient age from months to years.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame including the ``age_in_months`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new ``age`` column expressed in years.
    """
    df["age"] = df["age_in_months"] / 12  # convert age in months to age in years

    return df


DTYPES = {
    "patient_id": "Int64",
    "injury_mechanism": "category",  # not ordinal
    "injury_mechanism_severity": "Int64",  # an ordered category so we can encode it as integers
    "amnesia": "category",
    "loss_of_consciousness": "category",
    "duration_of_loss_of_consciousness": "Int64",
    "seizure": "boolean",
    "seizure_occurred_time": "Int64",  # ordinal category
    "seizure_duration": "Int64",
    "acting_normally": "boolean",
    "headache": "category",
    "headache_severity": "Int64",  # an ordered category
    "headache_start_time": "Int64",
    "vomiting": "boolean",
    "number_of_vomiting_episodes": "Int64",
    "vomiting_start_time": "Int64",
    "vomiting_last_time": "Int64",
    "dizziness": "boolean",
    "gcs_eye": "Int64",
    "gcs_verbal": "Int64",
    "gcs_motor": "Int64",
    "gcs_total": "Int64",
    "altered_mental_status": "boolean",
    "ams_agitated": "category",
    "ams_sleep": "category",
    "ams_slow": "category",
    "ams_repeat": "category",
    "ams_other": "category",
    "palpable_skull_fracture": "boolean",
    "palpable_skull_fracture_depressed": "category",
    "basilar_skull_fracture": "boolean",
    "basilar_skull_fracture_hemotympanum": "category",
    "basilar_skull_fracture_otorrhea": "category",
    "basilar_skull_fracture_periorbital_ecchymosis": "category",
    "basilar_skull_fracture_retroauricular_ecchymosis": "category",
    "basilar_skull_fracture_rhinorrhea": "category",
    "hematoma": "boolean",
    "hematoma_location": "category",
    "hematoma_size": "Int64",
    "age": "float64",
    "gender": "category",
    "ethnicity": "category",
    "race": "category",
    "ct_done": "boolean",
    "clinically_important_tbi": "boolean",
    "ciTBI_1": "boolean",
    "ciTBI_2": "boolean",
    "ciTBI_3": "boolean",
    "ciTBI_4": "boolean",
}


def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast cleaned columns to the pandas dtypes expected by the modeling code.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after the cleaning and age transformation steps.

    Returns
    -------
    pd.DataFrame
        DataFrame where each variable has the intended pandas dtype
    """
    return df.astype(DTYPES)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a random training/test split of the cleaned dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe to split.
    test_size : float, default TEST_SIZE
        Fraction of rows to assign to the test set.
    random_state : int, default RANDOM_STATE
        Seed used to make the split reproducible.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A pair ``(train_df, test_df)`` containing the training and test subsets.
    """
    # divide the dataset into train/test set

    test_df = df.sample(
        frac=test_size,
        random_state=random_state,
    )

    train_df = df.drop(test_df.index)

    return train_df, test_df


def clean_data(
    df: pd.DataFrame,
    recode_na_to_nan: bool = False,
    palpable_skull_fracture_handling: str = (
        "palpable_skull_fracture_depressed_missing"
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the raw TBI PUD dataset and create train/test datasets.

    Parameters
    ----------
    df:
        Raw TBI PUD dataframe.

    recode_na_to_nan:
        If True, recode all 92 (not applicable) to NaN in the selected
        not-applicable variables. If False, apply variable-specific
        cleaning rules before converting remaining 92 values to NaN.

    handle_logically_inconsistent_palpable_skull_fracture:
        Controls how logically inconsistent palpable skull fracture
        observations are handled. Valid options are:

        - ``"palpable_skull_fracture_depressed_missing"``:
          Set palpable skull fracture depressed to NaN.

        - ``"palpable_skull_fracture_missing"``:
          Set palpable skull fracture to NaN.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and test datasets.

    Notes
    -----
    The cleaning pipeline:

    1. Selects and renames variables relevant to the clinical decision rules.
    2. Restricts observations to GCS scores of 14 or 15.
    3. Removes observations with missing outcome values.
    4. Recalculates total GCS when all three GCS components are available.
    5. Combines unclear and palpable skull fracture into one value.
    6. Handles 92 (not applicable) values.
    7. Handles logically inconsistent palpable skull fracture observations.
    8. Converts age from months to years.
    9. Converts variables to their intended pandas data types.
    10. Randomly splits the data into training and test sets.
    """

    # only keep relevant columns
    df = select_and_rename_columns(df)

    df = filter_eligible_observations(df)

    # GCS cleaning
    df = clean_gcs(df)

    df = clean_palpable_skull_fracture(df)

    df = handle_logically_inconsistent_palpable_skull_fracture(
        df,
        palpable_skull_fracture_handling,
    )

    if recode_na_to_nan:
        df = recode_not_applicable_to_nan(df)
    else:
        df = clean_not_applicable_values(df)

    df = create_age_variable(df)

    df = convert_dtypes(df)

    # remove duplicates if any but ignore the patientid column when checking for duplicates since it is unique for each patient
    df.drop_duplicates(
        subset=[col for col in df.columns if col != "patientid"], inplace=True
    )

    train_df, test_df = split_train_test(df)

    return train_df, test_df


if __name__ == "__main__":
    raw_data = load_data(DATA_PATH)
    train_df, test_df = clean_data(raw_data)
