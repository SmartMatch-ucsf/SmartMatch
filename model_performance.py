import pandas as pd
import pickle
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, brier_score_loss, confusion_matrix
from sklearn.utils import resample
from sklearn.calibration import calibration_curve 
from statsmodels.stats.proportion import proportion_confint
from MSBOS_06_Analysis_Tools import stat_metrics
import pprint
 
model_name = 'old'
from hipac_ml_msbos.hipac_modeling_tools_old import (
        FeatureTransformer, train_valid_test_split)
 
# %% # Load the model and data
rae_folder = 'O:'
with open("paper_model.pkl", 'rb') as f:
    gbm, feature_transformer, rbc_threshold = pickle.load(f)
with open(f"{rae_folder}/Data/20230828/train_test_elective_only.pkl", 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f)
#with open(f"{rae_folder}/Data/silent_study/silent_study_data_20231001_20240315.pkl", 'rb') as f:
    # p_df = pickle.load(f)

p_df = pd.read_csv(f"{rae_folder}/Data/20240821/data_elective_only_485ae5e.csv")  

# %% filter out cases missing at certain tp
 
time_points = [
    'an_hour_before_last_score', 'an_hour_before_max_score']

#    'three_pm_day_before_last_score', 'three_pm_day_before_max_score',
 #   'seven_days_before_last_score', 'seven_days_before_max_score',
  #  'month_before_last_score', 'month_before_max_score']


# Identify rows with missing values in certain timepoints
exclude_rows = p_df[time_points].apply(lambda row: row.isna().any(), axis=1)

# Create a new DataFrame with only those rows
p_df = p_df[~exclude_rows]  
print(len(p_df))
 

# %% performance curve

 # set flag to plot datasets
test = 1
valid = 1

if test == 1:
    # Prepare the retrospective data test
    X_test_transformed = feature_transformer.transform(X_test)[feature_transformer.features]
    y_test['probs_test'] = gbm.predict_proba(X_test_transformed)[:, 1]
    y_test['target'] = y_test['periop_prbc_units_transfused'] > 0
    
    # Handle NaN values
    y_test = y_test.dropna(subset=['target', 'probs_test'])
    
    # Ensure target variable is binary
    y_test['target_binary'] = y_test['target'].astype(int)
    
if valid == 1:
    # Prepare the retrospective data valid
    X_valid_transformed = feature_transformer.transform(X_valid)[feature_transformer.features]
    y_valid['probs_test'] = gbm.predict_proba(X_valid_transformed)[:, 1]
    y_valid['target'] = y_valid['periop_prbc_units_transfused'] > 0
    y_valid = y_valid.dropna(subset=['target', 'probs_test'])
    y_valid['target_binary'] = y_valid['target'].astype(int)


# Use existing probabilities and outcome for prospective data
p_df['probs'] = p_df['an_hour_before_max_score']
p_df['target'] = p_df['outcome'].astype(bool)

p_df = p_df.dropna(subset=['target', 'probs'])

p_df['target_binary'] = p_df['target'].astype(int)
# Define the bootstrap_curve function
def bootstrap_curve(y_true, y_pred, curve_func, n_bootstraps=1000):
    y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
    y_pred = y_pred.values if isinstance(y_pred, pd.Series) else y_pred
    bootstrapped_curves = []
    for _ in range(n_bootstraps):
        indices = resample(range(len(y_true)), n_samples=len(y_true))
        y_true_boot, y_pred_boot = y_true[indices], y_pred[indices]
        curve = curve_func(y_true_boot, y_pred_boot)
        bootstrapped_curves.append(curve)
    return bootstrapped_curves

# Define the plot_curve_with_ci function
def plot_curve_with_ci(ax, x, y, bootstrapped_curves, label, color):
    ax.plot(x, y, label=label, color=color)
    interp_curves = []
    for curve in bootstrapped_curves:
        interp_x = np.linspace(0, 1, 100)
        interp_y = np.interp(interp_x, curve[0], curve[1])
        interp_curves.append(interp_y)
    y_lower = np.percentile(interp_curves, 2.5, axis=0)
    y_upper = np.percentile(interp_curves, 97.5, axis=0)
    ax.fill_between(interp_x, y_lower, y_upper, alpha=0.2, color=color)

 
# Create a single row of plots with shared y-axis
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
fig.suptitle('Prediction Model Performance Plots', fontweight='bold',fontsize=25)
colors = {
    'retro-valid': '#5D3FD3',  # Darker purple
    'retro-test': '#9A4ED4',   # Lighter purple
    'prospective': 'green'     # Green
}
# Set larger font sizes
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize':18,
    'legend.fontsize': 18
})  

dlist = [(p_df, colors['prospective'], 'Prospective')]
if test:
    dlist.append((y_test, colors['retro-test'], 'Retro-test'))
if valid:
    dlist.append((y_valid, colors['retro-valid'], 'Retro-valid'))


def add_cutpoint_dot(ax, fpr, tpr, cutpoint_index, label):
   ax.plot(fpr[cutpoint_index], tpr[cutpoint_index], 'o', markersize=10, label=label, color='black')
# ROC curves
for i, (data, color, label) in enumerate(dlist):
    # Calculate ROC curve and AUC
   probs_column = 'probs_test' if 'probs_test' in data else 'probs'
   fpr, tpr, thresholds = roc_curve(data['target_binary'], data[probs_column])
   roc_auc = auc(fpr, tpr)
   roc_curves = bootstrap_curve(data['target_binary'], data[probs_column], roc_curve)
   
   # Plot ROC curve with confidence interval
   plot_curve_with_ci(ax1, fpr, tpr, roc_curves, f'{label} ROC (AUC = {roc_auc:.2f})', color)
   
   # Calculate Brier score
  #brier_score = brier_score_loss(data['target_binary'], data[probs_column])
   
   # Annotate the AUC and Brier score on the plot
   # Adjust y position to avoid overlapping annotations
   annotation_y_position = 0.2 + i * 0.1
   ax1.text(0.5, annotation_y_position, f'{label} AUC: {roc_auc:.2f}', 
            color=color, fontsize=18, fontweight='bold', transform=ax1.transAxes)

   # Add the cutpoint dot using rbc_threshold
   cutpoint_index = np.argmin(np.abs(thresholds - rbc_threshold))
   ax1.plot(fpr[cutpoint_index], tpr[cutpoint_index], 'o', markersize=10, color=color)
   
ax1.set_title('ROC Curves',fontweight='bold')
ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax1.set_xlabel('1-Specificity', fontsize=18,fontweight='bold')
ax1.set_ylabel('Sensitivity', fontsize=18,fontweight='bold')
ax1.text(-0.1, 1.05, 'a)', transform=ax1.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')
#ax1.text(0.6, 0.2, f'Retro-test AUC: {auc_retro:.2f}', fontsize=18,fontweight='bold', color=colors['retrospective'])
#ax1.text(0.6, 0.1, f'Prosp AUC: {auc_prosp:.2f}', fontsize=18, fontweight='bold',color=colors['prospective'])

# Precision-Recall curves
for i, (data, color, label) in enumerate(dlist):
    # Calculate Precision-Recall curve and PR AUC
    probs_column = 'probs_test' if 'probs_test' in data else 'probs'
    precision, recall, thresholds = precision_recall_curve(data['target_binary'], data[probs_column])
    pr_auc = auc(recall, precision)
    pr_curves = bootstrap_curve(data['target_binary'], data[probs_column], precision_recall_curve)
    
    # Plot Precision-Recall curve with confidence interval
    plot_curve_with_ci(ax2, recall, precision, pr_curves, f'{label} Precision-Recall (PR AUC = {pr_auc:.2f})', color)
    
    # Annotate the PR AUC on the plot
    # Adjust y position to avoid overlapping annotations
    annotation_y_position = 0.8 + i * 0.1
    ax2.text(0.3, annotation_y_position, f'{label} PR-AUC: {pr_auc:.2f}', fontsize=18, fontweight='bold', color=color)

    # Add the cutpoint dot using rbc_threshold
    cutpoint_index = np.argmin(np.abs(thresholds - rbc_threshold))
    ax2.plot(recall[cutpoint_index], precision[cutpoint_index], 'o', markersize=10, color=color)

# Set plot title and labels
ax2.set_title('Precision-Recall Curves', fontweight='bold')
ax2.set_xlabel('Sensitivity', fontsize=18, fontweight='bold')
ax2.set_ylabel('Positive Predictive Value', fontsize=18, fontweight='bold')
ax2.text(-0.1, 1.05, 'b)', transform=ax2.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')

# Calibration curves
ax3.set_title('Calibration Plots', fontweight='bold')
ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

for i, (data, color, label) in enumerate(dlist):
    probs_column = 'probs_test' if 'probs_test' in data else 'probs'
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(data['target_binary'], data[probs_column], n_bins=10)
    cal_curves = bootstrap_curve(data['target_binary'], data[probs_column], 
                                 lambda y, p: calibration_curve(y, p, n_bins=10)[0])
    
    # Calculate confidence intervals
    prob_true_lower = np.percentile(cal_curves, 2.5, axis=0)
    prob_true_upper = np.percentile(cal_curves, 97.5, axis=0)
    
    # Calculate Brier score
    brier_score = brier_score_loss(data['target_binary'], data[probs_column])
    
    # Plot calibration curve
    ax3.plot(prob_pred, prob_true, "s-", label=f"{label} Calibration (Brier = {brier_score:.4f})", color=color)
    ax3.fill_between(prob_pred, prob_true_lower, prob_true_upper, alpha=0.2, color=color)
    
    # Annotate the Brier score on the plot
    annotation_y_position = 0.9 - i * 0.1
    ax3.text(0.05, annotation_y_position, f'{label} Brier: {brier_score:.4f}', fontsize=18, fontweight='bold', color=color)

    # Add the cutpoint dot using rbc_threshold
    cutpoint_index = np.argmin(np.abs(prob_pred - rbc_threshold))
    ax3.plot(prob_pred[cutpoint_index], prob_true[cutpoint_index], 'o', markersize=10, color=color)

# Set axis labels
ax3.set_xlabel('Average Predicted Probability', fontsize=18, fontweight='bold')
ax3.set_ylabel('Fraction of Positives', fontsize=18, fontweight='bold')

ax3.text(-0.1, 1.05, 'c)', transform=ax3.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')

# Combine the legend into one
line_legend = [
    plt.Line2D([0], [0], color=colors['retro-valid'], lw=2, linestyle='solid', label='Retro-valid'),
    plt.Line2D([0], [0], color=colors['retro-test'], lw=2, linestyle='solid', label='Retro-test'),
    plt.Line2D([0], [0], color=colors['prospective'], lw=2, linestyle='solid', label='Prospective'),
   ]

# Add the legend to the figure
fig.legend(handles=line_legend, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)

fig.legend(handles=line_legend, title="Data", loc='upper left', fontsize=18)

# Adjust the layout
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

# %% performance metrics
# re-transform data 
with open("paper_model.pkl", 'rb') as f:
    gbm, feature_transformer, rbc_threshold = pickle.load(f)
with open(f"{rae_folder}/Data/20230828/train_test_elective_only.pkl", 'rb') as f:
    X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f) 
 
# valid 
y_valid.reset_index(drop=True, inplace=True)
X_valid.reset_index(drop=True, inplace=True)
X_valid_original = X_valid.copy()
X_valid = feature_transformer.transform(X_valid)[feature_transformer.features]
y_valid['prediction_prob'] = gbm.predict_proba(X_valid)[:, 1]
y_valid['target'] = y_valid['periop_prbc_units_transfused'].fillna(0) > 0
y_valid['prediction'] = y_valid['prediction_prob'] > rbc_threshold

# test
y_test.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
X_test_original = X_test.copy()
X_test = feature_transformer.transform(X_test)[feature_transformer.features]
y_test['prediction_prob'] = gbm.predict_proba(X_test)[:, 1]
y_test['target'] = y_test['periop_prbc_units_transfused'].fillna(0) > 0
y_test['prediction'] = y_test['prediction_prob'] > rbc_threshold

# prosp
y_prosp = p_df.copy()
y_prosp['prediction_prob'] = p_df['an_hour_before_max_score']
y_prosp['target'] = p_df['outcome'].fillna(0) > 0  
y_prosp['prediction'] = y_prosp['prediction_prob'] > rbc_threshold
# Define the bootstrap_curve function

# Print performan metrics with the desired sensitivity and single dataset generate performance curve (no CI)
pprint.PrettyPrinter(width=20).pprint((stat_metrics(y_test,'Retrospective Test PRBC Singular Thresholds', f'Model 0628 {model_name}',None,  y_test['prediction'])))
pprint.PrettyPrinter(width=20).pprint((stat_metrics(y_prosp,'Prospective Test PRBC Singular Thresholds', f'Model 0628 {model_name}',None,  y_prosp['prediction'])))


  
