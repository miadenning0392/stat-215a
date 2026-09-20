import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix

def apply_updated_kuppermann(data_input):
    """
    Apply modified Kuppermann et al. 2009 clinical decision rule.
    """
    
    data_input = data_input.loc[data_input["gcs_total"] >= 14].copy()
    ct_yes = pd.Series(False, index = data_input.index)
    under_two = data_input["age_less_two"] == 1
    over_two = data_input["age_less_two"] == 2

    # < 2 years 
    branch1_under_two = ((data_input["gcs_total"] == 14) |
                         (data_input["alt_mental_stat"] == 1) |
                         (data_input["skull_fracture"] == 1))

    branch2_under_two = ((data_input["trauma_above_clavical_scalp_occipital"] == 1) |
                         (data_input["trauma_above_clavical_scalp_parietal"] == 1) |
                         (data_input["trauma_above_clavical_scalp_temporal"] == 1) |
                         (data_input["loss_consciousness_duration"].isin([2,3,4])) |
                         (data_input["injury_mech_severity"] == 3) |
                         (data_input["acting_normal"] == 0) |
                         (data_input["injury_mech"].isin([2,3])) |
                         (data_input["vomiting_ep_count"] == 3))

    rule_under_two = branch1_under_two | branch2_under_two

    # >= 2 years
    branch1_over_two = ((data_input["gcs_total"] == 14) |
                        (data_input["alt_mental_stat"] == 1) |
                        (data_input["basilar_skull_fracture"] == 1))

    branch2_over_two = ((data_input["loss_consciousness"] == 1) |
                        (data_input["injury_mech_severity"] == 3) |
                        (data_input["headache_severity"] == 3) |
                        (data_input["injury_mech"].isin([2,3])) |
                        (data_input["vomiting_start"] == 2))

    rule_over_two = branch1_over_two | branch2_over_two

    ct_yes.loc[under_two] = rule_under_two.loc[under_two]
    ct_yes.loc[over_two] = rule_over_two.loc[over_two]

    # Pull performance metrics
    y_obs = data_input["ci_tbi"].dropna().astype(int)
    y_pred = ct_yes.loc[y_obs.index].astype(int) 

    tn, fp, fn, tp = confusion_matrix(y_obs, y_pred, labels = [0, 1]).ravel()
    sens = tp/(tp + fn)
    spec = tn/(tn + fp)
    ct_rec_rate = y_pred.mean()

    return {
        "metrics": {
            "sensitivity": round(sens, 3),
            "specificity": round(spec, 3),
            "ct_rec_percent": round(ct_rec_rate*100, 1)
        }
    }


def fit_logistic_model(
    train_data_input,
    test_data_input,
    age_group
):
    """
    Fit logistic regression, depending on age group. 
    """

    train_data_input = train_data_input.copy()
    test_data_input = test_data_input.copy()
    
    # Create binary predictor variables
    for data in [train_data_input, test_data_input]:

        data["gcs_14"] = (data["gcs_total"] == 14).astype(int)

        data["scalp_hematoma"] = ((data["trauma_above_clavical_scalp_occipital"] == 1) |
                                  (data["trauma_above_clavical_scalp_parietal"] == 1) |
                                  (data["trauma_above_clavical_scalp_temporal"] == 1)).astype(int)

        data["loc_gt_5sec"] = (data["loss_consciousness_duration"].isin([2, 3, 4])).astype(int)
        data["severe_mech"] = (data["injury_mech_severity"] == 3).astype(int)
        data["struck_by_vehicle"] = (data["injury_mech"].isin([2, 3])).astype(int)
        data["vomiting_gt_2"] = (data["vomiting_ep_count"] == 3).astype(int)
        data["severe_headache"] = (data["headache_severity"] == 3).astype(int)
        data["vomiting_within_1hr"] = (data["vomiting_start"] == 2).astype(int)

    if age_group == 1:
        all_preds = [
            "gcs_14",
            "alt_mental_stat",
            "skull_fracture",
            "scalp_hematoma",
            "loc_gt_5sec",
            "severe_mech",
            "acting_normal",
            "struck_by_vehicle",
            "vomiting_gt_2"
        ]

    elif age_group == 2:
        all_preds = [
            "gcs_14",
            "alt_mental_stat",
            "basilar_skull_fracture",
            "loss_consciousness",
            "severe_mech",
            "severe_headache",
            "struck_by_vehicle",
            "vomiting_within_1hr"
        ]

    model_train = train_data_input.loc[(train_data_input["gcs_total"] >= 14) &
                                       (train_data_input["age_less_two"] == age_group),
                                       all_preds + ["ci_tbi"]].dropna().copy()

    model_test = test_data_input.loc[(test_data_input["gcs_total"] >= 14) &
                                     (test_data_input["age_less_two"] == age_group),
                                     all_preds + ["ci_tbi"]].dropna().copy()

    X_train = model_train[all_preds].astype(float)
    y_train = model_train["ci_tbi"].astype(int)

    X_test = model_test[all_preds].astype(float)
    y_test = model_test["ci_tbi"].astype(int)
        
    model = LogisticRegression(
            C = np.inf, # no penalization 
            max_iter = 1000,
            random_state = 22
    )

    model.fit(X_train, y_train)

    # Pull coefs (for later plotting)
    coefs = pd.DataFrame({"variable": X_train.columns, "coefficient": model.coef_[0]})

    # Choose probability cutoff for turning prediction probs into binary outcomes 
    # while achieving at least 90% sensitivity on the training set 
    train_prob = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, train_prob)
    valid = tpr >= 0.9 # target sensitivity 90%
    threshold = thresholds[valid].max()

    # Pull performance metrics for test data
    test_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sens = tp/(tp + fn)
    spec = tn/(tn + fp)
    ct_rec_rate = y_pred.mean()

    return {
        "metrics": {
            "sensitivity": round(sens, 3),
            "specificity": round(spec, 3),
            "ct_rec_percent": round(ct_rec_rate*100, 1)
        },
        "coefs": coefs
    }