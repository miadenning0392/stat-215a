import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from clean import clean_data
from models import apply_updated_kuppermann, fit_logistic_model

from eda import (
    create_ct_tbi_outcomes_fig,
    create_top_ct_indicators_fig,
    create_injury_mech_fig,
    create_injury_mech_severity_fig,
    create_vomiting_fig,
    create_vomiting_stability_fig,
    plot_logistic_coefs
)

raw_data = pd.read_csv("../data/TBI PUD 10-08-2013.csv")

# Clean
cleaned_data = clean_data(raw_data)

# Random split 70 train, 30 test
train_data, test_data = train_test_split(cleaned_data, test_size = 0.30, random_state = 22)

# Generate EDA figures from training set
fig1 = create_ct_tbi_outcomes_fig(train_data)
fig1.savefig("../figs/ct_tbi_outcomes.pdf", format = "pdf", bbox_inches = "tight")
plt.close(fig1)

fig2 = create_top_ct_indicators_fig(train_data)
fig2.savefig("../figs/top_ct_indicators.pdf", format = "pdf", bbox_inches = "tight")
plt.close(fig2)

fig3 = create_injury_mech_fig(train_data)
fig3.savefig("../figs/injury_mech.pdf", format = "pdf", bbox_inches = "tight")
plt.close(fig3)

fig4 = create_injury_mech_severity_fig(train_data)
fig4.savefig("../figs/injury_mech_severity.pdf", format = "pdf", bbox_inches = "tight")
plt.close(fig4)

fig5 = create_vomiting_fig(train_data)
fig5.savefig("../figs/vomiting.pdf", format = "pdf", bbox_inches = "tight")
plt.close(fig5)

# Run models
usual_kupp = apply_updated_kuppermann(cleaned_data)
usual_under_two = fit_logistic_model(train_data_input = train_data, test_data_input = test_data, age_group = 1)
usual_over_two = fit_logistic_model(train_data_input = train_data, test_data_input = test_data, age_group = 2)

print("\nModel 1: Updated Kuppermann:", usual_kupp["metrics"])
print("\nModel 2: Logistic Regression (under 2 years old):", usual_under_two["metrics"])
print("\nModel 3: Logistic Regression (over 2 years old):", usual_over_two["metrics"])


################################################################
# Stability check
################################################################
# Clean (with perturbation)
cleaned_data_stability = clean_data(raw_data, impute_missing_cat_child_vars = "by_overall_mode")

# Random split 70 train, 30 test
train_data_stability, test_data_stability = train_test_split(cleaned_data_stability, test_size = 0.30, random_state = 22)

# Generate figure to show change from perturbation
fig6 = create_vomiting_stability_fig(train_data, train_data_stability)
fig6.savefig("../figs/vomiting_stability.pdf", format = "pdf", bbox_inches = "tight")
plt.close(fig6)

# Re-run models under perturbation 
stab_kupp = apply_updated_kuppermann(cleaned_data_stability)

stab_under_two = fit_logistic_model(train_data_input = train_data_stability, 
                                    test_data_input = test_data_stability,
                                    age_group = 1)

stab_over_two = fit_logistic_model(train_data_input = train_data_stability,
                                   test_data_input = test_data_stability,
                                   age_group = 2)

print("\nModel 1.1: Updated Kuppermann, Stability Check:", stab_kupp["metrics"])
print("\nModel 2.2: Logistic Regression, Stability Check (under 2 years old):", stab_under_two["metrics"])
print("\nModel 3.2: Logistic Regression, Stability Check (over 2 years old):", stab_over_two["metrics"])

fig7 = plot_logistic_coefs(usual_under_two["coefs"], stab_under_two["coefs"],
                           "Change in Logistic Regression Coefficients (<2 years)")
fig7.savefig("../figs/log_reg_stability.pdf", format = "pdf", bbox_inches = "tight")
plt.close(fig7)