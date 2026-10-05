import pandas as pd


def remove_missing_all_questions_func(ling_data):
    # drop respondents missing all 67 questions
    q_cols = [c for c in ling_data.columns if c.startswith("Q")]
    n_skipped = (ling_data[q_cols] == 0).sum(axis=1)
    ling_data = ling_data[n_skipped < 67]
    return ling_data


def clean_ling_data(
    ling_data,
    remove_missing_lat_long: bool = True,
    remove_missing_state: bool = False,
    remove_missing_city: bool = False,
    remove_missing_all_questions: bool = False,
    remove_alaska_hawaii: bool = True,
):
    # Cleans the ling data according to parameters, defaults as listed, returns clean data

    ling_data = (
        ling_data[(~ling_data["lat"].isna()) | (~ling_data["long"].isna())] if remove_missing_lat_long else ling_data
    )

    ling_data = ling_data[(~ling_data["STATE"].isna())] if remove_missing_state else ling_data

    ling_data = ling_data[(~ling_data["CITY"].isna())] if remove_missing_city else ling_data

    ling_data = remove_missing_all_questions_func(ling_data) if remove_missing_all_questions else ling_data

    ling_data = ling_data[~ling_data["STATE"].isin(["AK", "HI"])] if remove_alaska_hawaii else ling_data

    return ling_data
