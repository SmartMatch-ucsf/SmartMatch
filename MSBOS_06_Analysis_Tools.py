import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score
from sklearn.calibration import calibration_curve
from pathlib import Path
from sklearn.metrics import confusion_matrix
from statsmodels.stats.proportion import proportion_confint 
import pprint


#%%

def stat_metrics(y_valid, test_name='', model_name='',set_sensitivity=None,  predictions=None):
    """
    Calculates performance metrics for classification models, 
        including sensitivity (&95%CI), specificity, PPV (&95%CI), NPV, FN, FP, 
        percentage of cases with transfusion recommendations predicted by the model, and sample size.
    Optionally adjusts the prediction threshold based on a target sensitivity.
    
    Parameters:
    - y_valid: DataFrame with true labels ('target') and predicted probabilities ('prediction_prob').
    - test_name: Optional name of the test.
    - model_name: Optional name of the model.
    - set_sensitivity: Optional target sensitivity to adjust the prediction threshold. 
        Default probability threshold to binarize the prediction is 0.5
    - predictions: Optional precomputed binary predictions.
    
    Returns:
    - Dictionary containing the metrics
    """

    fpr, tpr, _ = roc_curve(y_valid['target'], y_valid['prediction_prob'])
    roc_auc = auc(fpr, tpr)
    if set_sensitivity!=None:
        precision, recall, thresholds = precision_recall_curve(
            y_valid['target'], y_valid['prediction_prob'])

    
        i = np.argmin(np.abs(recall - set_sensitivity))

        print(recall[i], precision[i], thresholds[i])
        threshold = thresholds[i]

        predictions = (y_valid['prediction_prob'] >= threshold).astype(int)
        

    # Compute confusion matrix
    cm = confusion_matrix(y_valid['target'], predictions, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    # Calculate sensitivity, specificity, PPV, NPV, and 1-Sensitivity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    one_minus_sens = 1 - sensitivity
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    pcnt_recommended = predictions.mean()

    # sensitivity CI
    sen_l, sen_u = proportion_confint(
        tp, tp + fn, alpha=0.05, method='wilson'
    )
    ppv_l, ppv_u = proportion_confint(
        tp, tp + fp, alpha=0.05, method='wilson'
    )
    generate_results(y_valid['target'], y_valid['prediction_prob'], predictions, test_name, model_name, len(y_valid['target']))



    # Create a dictionary with the desired output values
    if set_sensitivity!=None:
        
        return {
            'Sample Size': len(y_valid),
            
            'threshold': threshold,
            'Sensitivity & CI': '{:.1f}%, [{:.0f}, {:.0f}]'.format(
                    sensitivity * 100, sen_l * 100, sen_u * 100
                ),
            'Sensitivity': sensitivity,
            'sensitivity_CI': '[%.2f, %.2f]' % (sen_l, sen_u),
            'Specificity': specificity,
            'FP': fp,
            'FN': fn,
            '1-Sensitivity': one_minus_sens,
            'PPV & CI': '{:.1f}%, [{:.0f}, {:.0f}]'.format(
                    ppv * 100, ppv_l * 100, ppv_u * 100
                ),
            'NPV': npv,
            'pcnt_recommended': pcnt_recommended }
    else: 
        return{
        'Sample Size': len(y_valid),
        
        'Sensitivity & CI': '{:.1f}%, [{:.0f}, {:.0f}]'.format(
                sensitivity * 100, sen_l * 100, sen_u * 100
            ),
        'Sensitivity': sensitivity,
        'sensitivity_CI': '[%.2f, %.2f]' % (sen_l, sen_u),
        'Specificity': specificity,
        'FP': fp,
        'FN': fn,
        '1-Sensitivity': one_minus_sens,
        'PPV & CI': '{:.1f}%, [{:.0f}, {:.0f}]'.format(
                ppv * 100, ppv_l * 100, ppv_u * 100
            ),
        'NPV': npv,
        'pcnt_recommended': pcnt_recommended
    }



def generate_results(y_true, y_score, predictions, test_name, model_name, sample_size,folder_path=None):
    """
    Generates and plots ROC, Precision-Recall, and Calibration curves for a classification model, along with accuracy and AUC scores. 
    The plots are optionally saved to a specified folder.
    
    Parameters:
    - y_true: True labels of the dataset.
    - y_score: Predicted probabilities from the model.
    - predictions: Binary predictions from the model.
    - test_name: Name of the test (included in the plot title).
    - model_name: Name of the model (included in the plot title).
    - sample_size: The number of samples in the dataset.
    - folder_path: Optional path to save the figures (default: 'N:/Results/Figures').
    
    Returns:
    - Displays the plots for ROC, Precision-Recall, and Calibration curves. Optionally saved.
    """

    accuracy = accuracy_score(y_true, predictions)
    if folder_path is None:
        pathfigures = Path('N:/Results/Figures')
    else:
        pathfigures = folder_path
    
    # Set the overall aesthetics
    plt.style.use('seaborn-v0_8-white')  # Use a style with a white background

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=15, fontweight='bold')
    plt.title('Receiver Operating Characteristic', fontsize=18, fontweight='bold')
    plt.legend(loc=4, fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, label='PR-AUC = %0.2f' % pr_auc, linewidth=2)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=15, fontweight='bold')
    plt.ylabel('Precision', fontsize=15, fontweight='bold')
    plt.title('Precision-Recall Curve', fontsize=18, fontweight='bold')
    plt.legend(loc='lower left', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=5)
    plt.subplot(1, 3, 3)
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', label='Calibration curve', markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=15, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=15, fontweight='bold')
    plt.title('Calibration Curve', fontsize=18, fontweight='bold')
    plt.legend(loc='lower right',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.suptitle(model_name + ' ' + test_name + '\n' + ' N = ' + str(sample_size) + " & Accuracy: %.2f%%" % (accuracy * 100.0), fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f_name = model_name + '_sz_' + str(sample_size) + "_" + test_name + '.png'
    try:
        plt.savefig(pathfigures / f_name)
        plt.show()
    except:
        print('figure not saved due to error in folder_path')
 


