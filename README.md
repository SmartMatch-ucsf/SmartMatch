# AI Blood SmartMatch  
Code repository for the AI Blood project.

## Scripts 

### 1. build_hipac_model.py  
- Split the retrospective dataset into training, testing, and validation sets.  
- Perform feature transformation and train a default XGB classifier on the retrospective training set.  
- Identify the probability threshold that achieves MSBOS-matched sensitivity (0.71 in this paper).  
- Save the split data (untransformed), trained model, feature transformer, and threshold to pickle files.

### 2. model_performance.py  
- Plot model performance curves (ROC, Precision-Recall, and Calibration).  
- Calculate and output model performance metrics: sensitivity, specificity, positive predictive value, negative predictive value, accuracy, and other statistics related to the performance curves.

-- 

## Utilities:

### MSBOS_06_Analysis_Tools.py  
- Functions for generating performance curves and calculating performance metrics (sensitivity, specificity, PPV, NPV, etc.).

### hipac_ml_msbos module  
- Functions for splitting datasets and performing feature transformations.
