
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureTransformer:

    def _get_picis_code_with_transfusion(self, data_df):
        """
        Extracts unique PICIS codes from the dataset and calculates the average transfusion rates for each code. 
        Identifies codes with historical transfusions and sets up a one-hot encoder for these PICIS codes.
        
        Parameters:
        - data_df: DataFrame containing patient and transfusion data.
        
        Returns:
        - Updates the transformer object with a list of PICIS codes that have transfusion history 
            and initializes a one-hot encoder for those codes.
        """

        transfusion_columns = [
            'periop_prbc_units_transfused',
            'periop_platelets_units_transfused',
            'periop_ffp_units_transfused',
            'periop_cryoprecipitate_units_transfused'
        ]

        df_transfusion = (
            data_df.assign(
                picis_codes=data_df.picis_codes.str.split(";")
            ).explode('picis_codes')
            .reset_index(drop=True)
        )
        df_tx_picis_means = (
            df_transfusion.groupby('picis_codes', as_index=False)
            [transfusion_columns].mean()
            .round(decimals=3)
        )
        df_tx_picis_count = (
            df_transfusion.groupby(df_transfusion.picis_codes)
            .size()
            .reset_index(name='count')
        )
        # Results in the picis code counts with transfusion rate per picis code
        df_tx_picis_means = pd.merge(
            df_tx_picis_count, df_tx_picis_means,
            how='left', on=['picis_codes']
        )
        df_tx_picis_means['hist_transfused'] = (
            df_tx_picis_means[transfusion_columns].sum(1) > 0
        )

        self.df_PICIS_txn = (
            df_tx_picis_means.loc[df_tx_picis_means['hist_transfused']]
            ['picis_codes'].tolist()
        )

        self.picis_one_hot_encoder = OneHotEncoder(
            categories=[self.df_PICIS_txn + ['PI0000'] + ['NA0000']],
            sparse=False,
            handle_unknown='infrequent_if_exist'
        )

        self.picis_one_hot_encoder.fit(df_transfusion[['picis_codes']])
        self.picis_one_hot_feature_names = list(
            self.picis_one_hot_encoder.get_feature_names_out()
        )

    def _create_picis_features(self, data_df):
        """
        Creates one-hot encoded PICIS code features for the dataset. 
        If a PICIS code does not have a transfusion history, 
        it is assigned a default value ('PI0000' or 'NA0000').
        
        Parameters:
        - data_df: DataFrame containing PICIS codes and episode IDs.
        
        Returns:
        - A DataFrame with the new PICIS one-hot encoded features, 
            merged with the original dataset based on episode ID.
        """

        X = data_df[['an_episode_id', 'picis_codes']]
        df_picis_codes = (
            X.assign(picis_codes=X.picis_codes.str.split(";"))
            .explode('picis_codes')
            .reset_index(drop=True)
        )

        # Create dummy variable if not in list of transfused codes
        df_picis_codes.loc[
                ~df_picis_codes.picis_codes.isin(self.df_PICIS_txn),
                'picis_codes'
            ] = 'PI0000'

        # Create dummy variable of NA pices codes
        df_picis_codes['picis_codes'] = (
            df_picis_codes['picis_codes'].fillna('NA0000')
        )

        df = pd.DataFrame(
            (
                self.picis_one_hot_encoder.transform(
                    df_picis_codes[['picis_codes']]
                )
                .astype(int)
            ),
            columns=self.picis_one_hot_feature_names
        )
        df_picis_codes = pd.concat(
            [df_picis_codes[['an_episode_id']], df], axis=1
        )

        return (
            data_df.merge(
                (
                    df_picis_codes.groupby('an_episode_id', as_index=False)
                    [self.picis_one_hot_feature_names].sum()
                ),
                on='an_episode_id',
                how='left'
            )
        )

    def fit_transform(self, data_df):
        """
        Fits and transforms the data using the one-hot encoder and standard scaler for categorical and continuous features, 
        respectively. Also, it creates additional PICIS-related features. 
        
        Parameters:
        - data_df: Input DataFrame with all necessary features.
        
        Returns:
        - A transformed DataFrame with encoded and scaled features, ready for model training.
        """

        # one-hot encoding
        self.categorical_features = [
            'sched_surgical_service',
            'sched_prim_surgeon_provid_1',
            'prepare_asa',
            'prepare_asa_e',
            'sched_proc_max_complexity',
            'sched_surgical_dept_campus',
            'prior_dept_location',
        ]
        self.continuous_features = [
            # dem_var
            'age',
            # bmi_var
            'weight_kg', 'height_cm', 'bmi',
            # case_cont
            'sched_est_case_length', 'sched_surgeon_cnt',
            'sched_proc_cnt', 'sched_proc_diag_cnt',
            # day_cont
            'enc_los_to_surg', 'hist_prior_transf_platelets_days',
            'hist_prior_transf_prbc_days',
            'hist_prior_transf_ffp_days',
            'hist_prior_transf_cryoprecipitate_days',
            # 'hist_prior_transf_platelets_days_toanesstart',
            # 'hist_prior_transf_prbc_days_toanesstart',
            # 'hist_prior_transf_ffp_days_toanesstart',
            # 'hist_prior_transf_cryoprecipitate_days_toanesstart',
            # labs_var
            'preop_base_excess_abg',
            'preop_base_excess_vbg', 'preop_bicarbonate_abg',
            'preop_bicarbonate_vbg', 'preop_bun', 'preop_chloride',
            'preop_chloride_abg', 'preop_chloride_vbg', 'preop_creatinine',
            'preop_hematocrit', 'preop_hematocrit_from_hb_abg',
            'preop_hematocrit_from_hb_vbg', 'preop_hemoglobin',
            'preop_hemoglobin_abg', 'preop_hemoglobin_vbg',
            'preop_lymphocyte_cnt', 'preop_neutrophil_cnt', 'preop_pco2_abg',
            'preop_pco2_vbg', 'preop_platelets', 'preop_ph_abg',
            'preop_ph_vbg', 'preop_potassium', 'preop_potassium_abg',
            'preop_potassium_vbg', 'preop_rbc', 'preop_sodium',
            'preop_sodium_abg', 'preop_sodium_vbg', 'preop_wbc',
            'msbos_cnt', 'msbos_wb_cnt', 'msbos_rbc_cnt',
        ]
        self.binary_features = [
            'sched_addon_yn', 'sched_emergency_case',
            'sched_or_equip_cell_saver_yn', 'sched_neuromonitoring_yn',
            'prepare_visit_yn', 'sched_bypass_yn',
            'hist_transf_1week_yn', 'hist_transf_1day_yn',
            'hist_prior_transf_yn',
            'language_interpreter_needed_yn', 'language_english_yn',
            'arrival_ed_yn', 'icu_admit_prior_24hr_yn', 'prior_dept_inpt_yn',
            'msbos_ts', 'msbos_tc', 'msbos_cryo', 'msbos_prbc', 'msbos_ffp',
            'msbos_platelets', 'msbos_bppp', 'msbos_wholeblood',
            'race_ethnicity_asian',
            'race_ethnicity_black',
            'race_ethnicity_latinx',
            'race_ethnicity_multi',
            'race_ethnicity_native_am_alaska',
            'race_ethnicity_hi_pac_islander',
            'race_ethnicity_other',
            'race_ethnicity_swana',
            'race_ethnicity_unknown',
            'race_ethnicity_white',
            'home_meds_anticoag_warfarin_yn',
            'home_meds_anticoag_heparin_sq_yn',
            'home_meds_anticoag_heparin_iv_yn',
            'home_meds_anticoag_fondaparinux_yn',
            'home_meds_anticoag_exoxaparin_yn',
            'home_meds_anticoag_argatroban_yn',
            'home_meds_anticoag_bivalirudin_yn',
            'home_meds_anticoag_lepirudin_yn',
            'home_meds_anticoag_dabigatran_yn',
            'home_meds_anticoag_clopidogrel_yn',
            'home_meds_anticoag_prasugrel_yn',
            'home_meds_anticoag_ticlodipine_yn',
            'home_meds_anticoag_abxicimab_yn',
            'home_meds_anticoag_eptifibatide_yn',
            'home_meds_anticoag_tirofiban_yn',
            'home_meds_anticoag_alteplase_yn',
            'home_meds_anticoag_apixaban_yn',
            'sex_female',
            'sex_male',
            'sex_nonbinary',
            'sex_unknown',
            'gender_identity_female',
            'gender_identity_male',
            'gender_identity_transgenderfemale',
            'gender_identity_transgendermale',
            'gender_identity_nonbinary',
            'gender_identity_other',
            'gender_identity_unknown',
        ]

        # one hot encoder
        self.one_hot_encoder = OneHotEncoder(
            sparse=False, handle_unknown='infrequent_if_exist'
        )
        self.one_hot_encoder.fit(data_df[self.categorical_features])
        self.one_hot_feature_names = list(
            self.one_hot_encoder.get_feature_names_out()
        )

        # %%Features to Normalize using Standard Scalar
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(data_df[self.continuous_features])

        self._get_picis_code_with_transfusion(data_df)

        # final list of features
        self.features = (
            self.continuous_features
            + self.one_hot_feature_names
            + self.picis_one_hot_feature_names
            + self.binary_features
        )

        return self.transform(data_df)

    def transform(self, data_df):
        """
        Transforms the input data by applying the already fitted one-hot encoders, 
        standard scalers, and PICIS code transformations to create a final feature set for the model.
        
        Parameters:
        - data_df: DataFrame with original features.
        
        Returns:
        - A transformed DataFrame with the continuous, one-hot encoded, and PICIS features.
        """


        data_df[self.continuous_features] = self.standard_scaler.transform(
                data_df[self.continuous_features]
            )

        df = pd.DataFrame(
            (
                self.one_hot_encoder.transform(
                    data_df[self.categorical_features]
                )
                .astype(int)
            ),
            columns=self.one_hot_feature_names
        )
        df['an_episode_id'] = data_df['an_episode_id']

        data_df = data_df.merge(df, how='left', on=['an_episode_id'])

        data_df = self._create_picis_features(data_df)

        return data_df


def train_valid_test_split(
        df, target, train_size=0.8, valid_size=0.1,
        test_size=0.1, method='random', sort_by_col = None, random_state=None):
    '''
    For a given input dataframe this prepares X_train, y_train, X_valid,
    y_valid, X_test, y_test for final model development

    Parameters:
    -----------
    df: 'dataframe', input dataframe
    target: 'str' , target variable
    train_size: 'float', proportion of train dataset
    valid_size: 'float', proportion of valid dataset
    test_size: 'float', proportion of test dataset
    method: 'str', default 'random'.
    2 methods available ['random', 'sorted']. in sorted dataframe is sorted by
    the input column and then splitting is done
    sort_by_col : 'str', defaul None. Required when method = 'sorted'
    random_state : random_state for train_test_split


    Output:
    -------
    X_train, y_train, X_valid, y_valid, X_test, y_test

    '''

    total = train_size + valid_size + test_size
    if total>1:
        raise Exception(
            " Total of train_size + valid_size + test_size should be 1"
        )
    else:

        if method=='random':
            df_train, df_rem = train_test_split(df, train_size=train_size, random_state=random_state)
            test_prop = test_size/(test_size+valid_size)
            df_valid, df_test = train_test_split(df_rem, test_size=test_prop, random_state=random_state)

            X_train, y_train = (
                df_train.drop(columns=target).copy(), df_train[target].copy()
            )
            X_valid, y_valid = (
                df_valid.drop(columns=target).copy(), df_valid[target].copy()
            )
            X_test, y_test = (
                df_test.drop(columns=target).copy(), df_test[target].copy()
            )

        if method == 'sorted':
            train_index = int(len(df)*train_size)
            valid_index = int(len(df)*valid_size)

            df.sort_values(by = sort_by_col, ascending=True, inplace=True)
            df_train = df[0:train_index]
            df_rem = df[train_index:]
            df_valid = df[train_index:train_index+valid_index]
            df_test = df[train_index+valid_index:]

            X_train, y_train = (
                df_train.drop(columns=target).copy(), df_train[target].copy()
            )
            X_valid, y_valid = (
                df_valid.drop(columns=target).copy(), df_valid[target].copy()
            )
            X_test, y_test = (
                df_test.drop(columns=target).copy(), df_test[target].copy()
            )


        return X_train, y_train, X_valid, y_valid, X_test, y_test