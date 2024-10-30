import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
rae_folder = 'O:' 
model_name = 'old'
from hipac_ml_msbos.hipac_modeling_tools_old import (
        FeatureTransformer, train_valid_test_split) 


data_df = pd.read_csv(f'{rae_folder}/Data/20230828/data_elective_only.csv') 
data_df['scheduled_for_dttm'] = (
    pd.to_datetime(data_df['scheduled_for_dttm'], utc=True)
    .dt.tz_convert('US/Pacific')
)
data_df = data_df.loc[
    (data_df['scheduled_for_dttm']<'2023-01-01')
    & (data_df['scheduled_for_dttm']>'2015-12-31')
]
 

outcomes_only = [
    'periop_platelets_units_transfused',
    'periop_prbc_units_transfused',
    'periop_ffp_units_transfused',
    'periop_cryoprecipitate_units_transfused'
]

X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(
    data_df,  target=outcomes_only,
    valid_size=0.1, test_size=0.1,
    method='sorted', sort_by_col='scheduled_for_dttm'
)

#saving the pickle files prior to calling transformer
#with open(f"{rae_folder}/Data/20230828/train_test_elective_only.pkl", 'wb') as f:
#    pickle.dump([X_train, y_train, X_valid, y_valid, X_test, y_test], f)

 
 
feature_transformer = FeatureTransformer()
X_train = feature_transformer.fit_transform(
        pd.concat([X_train, y_train], axis=1))[feature_transformer.features]
X_valid = feature_transformer.transform(X_valid)[feature_transformer.features]
X_test = feature_transformer.transform(X_test)[feature_transformer.features]


y_train = y_train.fillna(0)
y_valid = y_valid.fillna(0)
y_test = y_test.fillna(0)
y_train = y_train['periop_prbc_units_transfused'] > 0
y_valid = y_valid['periop_prbc_units_transfused'] > 0
y_test = y_test['periop_prbc_units_transfused'] > 0
 

# %%
# call default model
gbm = XGBClassifier()
gbm.fit(X_train, y_train)

# set threshold
precision, recall, threshold = precision_recall_curve(
    y_valid, gbm.predict_proba(X_valid)[:, 1]
)
# threshold is set @ sensitivity 71% - msbos recommendation's performance on retro valid
i = np.argmin(np.abs(recall - .71))  
print(precision[i], threshold[i])

# save model, feature transformer, and threshold
with open("paper_model.pkl", 'wb') as f:
    pickle.dump([gbm, feature_transformer, threshold[i]], f)

