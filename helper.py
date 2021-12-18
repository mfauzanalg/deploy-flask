import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def drop_cols_func(data):
  drop_cols = ['count', 'country', 'state', 'churn_label', 'lat_long', 
                'customerid', 'city', 'zip_code', 
                'latitude', 'longitude', 'churn_score', 'cltv', 'churn_reason']

  data.drop(drop_cols, axis=1, inplace=True)
  return data

def replace_yes_no(data):
  bool_list = ['senior_citizen', 'partner', 'dependents', 'phone_service', 
                 'multiple_lines', 'internet_service', 'online_security', 
                 'online_backup', 'device_protection', 'tech_support', 
                 'streaming_tv', 'streaming_movies', 'paperless_billing']

  for col in bool_list:
    data.replace({col : { 'Yes' : 1, 'No' : 0, 'No phone service' : 0, 
                              'No internet service' : 0}}, inplace=True)
    
  return data

def impute_total_charges(data):
  i=0
  for label, row in data.iterrows():
      if not isinstance(row['total_charges'], (int, float)): 
          i+=1
          data.loc[label, 'total_charges'] = (
              row['tenure_months']*row['monthly_charges'])

  data['total_charges'] = data['total_charges'].astype(float)
  print('Total imputed rows:', i)
  return data

def ohe_categorical(data):
  # create one-hot encoded dummy variables for categoricals
  categoricals = ['gender', 'internet_service', 'contract', 'payment_method']
  tc_ohe = pd.get_dummies(data[categoricals], drop_first=False, dtype=int)
  tc_ohe.rename(
      columns={'gender_Male' : 'gender_male',
                'gender_Female' : 'gender_female', 
                'internet_service_0' : 'internet_svc_none' , 
                'internet_service_DSL' : 'internet_svc_dsl', 
                'internet_service_Fiber optic' : 'internet_svc_fiber', 
                'contract_Month-to-month' : 'contract_mtm', 
                'contract_One year' : 'contract_1yr', 
                'contract_Two year' : 'contract_2yr', 
                'payment_method_Bank transfer (automatic)' : 'pmt_meth_bank_trx_auto', 
                'payment_method_Credit card (automatic)' : 'pmt_meth_cc_auto', 
                'payment_method_Electronic check' : 'pmt_meth_echeck', 
                'payment_method_Mailed check' : 'pmt_meth_mail_check'
              }, inplace = True)

  # concatenate OHE with original df, and drop original category columns
  tc2 = pd.concat([data, tc_ohe], axis=1)
  tc2.drop(categoricals, axis=1, inplace=True)

  from copy import deepcopy

  data = deepcopy(tc2)
  return data

def map_products(data):
    prod_cols = ['p1_phone_only', 'p2_phone_multi', 'p3_dsl_only', 'p4_dsl_plus', 
                 'p5_dsl_bund_core', 'p6_dsl_bund_plus', 'p7_fib_bund_core', 
                 'p8_fib_bund_plus']
    for newcol in prod_cols:
        data[newcol]= 0
        
        data.loc[(data['phone_service']==1) & (data['multiple_lines']==0) & 
         (data['internet_svc_dsl']==0) & (data['internet_svc_fiber']==0) & 
         (data['tech_support']==0) & (data['online_backup']==0) & 
         (data['online_security']==0) & (data['device_protection']==0), 
         'p1_phone_only'] = 1

        data.loc[(data['phone_service']==1) & (data['multiple_lines']==1) & 
         (data['internet_svc_dsl']==0) & (data['internet_svc_fiber']==0) & 
         (data['tech_support']==0) & (data['online_backup']==0) &
         (data['online_security']==0) & (data['device_protection']==0), 
         'p2_phone_multi'] = 1

        data.loc[(data['phone_service']==0) & (data['multiple_lines']==0) & 
         (data['internet_svc_dsl']==1) & (data['internet_svc_fiber']==0) & 
         (data['tech_support']==0) & (data['online_backup']==0) &
         (data['online_security']==0) & (data['device_protection']==0), 
         'p3_dsl_only'] = 1

        data.loc[(data['phone_service']==0) & (data['multiple_lines']==0) & 
         (data['internet_svc_dsl']==1) & (data['internet_svc_fiber']==0) & 
         ((data['tech_support']==1) | (data['online_backup']==1) |
         (data['online_security']==1) | (data['device_protection']==1)), 
         'p4_dsl_plus'] = 1

        data.loc[(data['phone_service']==1) & (data['multiple_lines']==0) & 
         (data['internet_svc_dsl']==1) & (data['internet_svc_fiber']==0) & 
         (data['tech_support']==0) & (data['online_backup']==0) &
         (data['online_security']==0) & (data['device_protection']==0), 
         'p5_dsl_bund_core'] = 1

        data.loc[(data['phone_service']==1) & (data['internet_svc_dsl']==1) 
         & (data['internet_svc_fiber']==0) & ((data['multiple_lines']==1) | 
         (data['tech_support']==1) | (data['online_backup']==1) |
         (data['online_security']==1) | (data['device_protection']==1)), 
         'p6_dsl_bund_plus'] = 1

        data.loc[(data['phone_service']==1) & (data['multiple_lines']==0) & 
         (data['internet_svc_dsl']==0) & (data['internet_svc_fiber']==1) & 
         (data['tech_support']==0) & (data['online_backup']==0) &
         (data['online_security']==0) & (data['device_protection']==0), 
         'p7_fib_bund_core'] = 1

        data.loc[(data['phone_service']==1) & (data['internet_svc_dsl']==0) & 
         (data['internet_svc_fiber']==1) & ((data['multiple_lines']==1) | 
         (data['tech_support']==1) | (data['online_backup']==1) |
         (data['online_security']==1) | (data['device_protection']==1)), 
         'p8_fib_bund_plus'] = 1

    return data

# function creates a few calculated metrics as features (not all made final model)
def create_features(data):
    # create column for lifetime average monthly charge
    data['charge_trend_index'] = np.where(data.tenure_months!=0, 
                                               round(data.monthly_charges / 
                                               (data.total_charges / 
                                                data.tenure_months), 2), 
                                               data.monthly_charges)
    
    # create feature 0/1 whether customer streams content via internet svc
    data['streams'] = data[['streaming_movies', 'streaming_tv']].sum(axis=1)
    
    # create feature 0/1 whether payment method is an automated method or not
    data['pmt_meth_auto'] = data[['pmt_meth_cc_auto', 
                                        'pmt_meth_bank_trx_auto']].sum(axis=1)

    # create feature counting number of add-on services
    data['svc_add_ons'] = data[['online_security', 'online_backup', 
                                      'device_protection', 'tech_support', 
                                      'multiple_lines']].sum(axis=1)
    
    # create charges index ratio of actual charges to standard charges
    data['charge_cust_index'] = round(data['monthly_charges'] / 
                                      (data['phone_service']*18.26 + 
                                       data['internet_svc_dsl']*27.88 + 
                                       data['internet_svc_fiber']*58.79 + 
                                       data['multiple_lines']*7.51 + 
                                       data['online_security']*3.34 + 
                                       data['online_backup']*5.75 + 
                                       data['device_protection']*9.57 +
                                       data['tech_support']*5.86),2
                                     )
    
    # bin tenure into 1st year versus other longer loyalty
    data['tenure_1yr'] = data.tenure_months.apply(lambda x: 1 if x<12 else 0)
    data['tenure_2-4yr'] = data.tenure_months.apply(lambda x: 1 if x>12 and 
                                                    x<48 else 0)
    data['tenure_4+yr'] = data.tenure_months.apply(lambda x: 1 if x>48 else 0)

    return data

def feature_drops(data):
  feat_drops = ['senior_citizen', 'gender_female', 'gender_male', 
              'online_security', 'online_backup',  'device_protection', 
              'multiple_lines', 'phone_service', 'tech_support', 'internet_svc_dsl', 
              'internet_svc_fiber', 'p1_phone_only', 'internet_svc_none', 'svc_add_ons', 
              'streaming_movies', 'streaming_tv', 'total_charges', 'charge_trend_index', 
              'monthly_charges', 'pmt_meth_mail_check', 'pmt_meth_cc_auto', 
              'tenure_1yr', 'tenure_2-4yr', 'tenure_4+yr',
              'pmt_meth_bank_trx_auto', 'pmt_meth_echeck', 'contract_mtm']

  data.drop(feat_drops, axis=1, inplace=True)
  return data

def data_preprocess(data):
  data1 = drop_cols_func(data)
  data1 = replace_yes_no(data1)
  data1 = impute_total_charges(data1)
  data1 = ohe_categorical(data1)
  data1 = map_products(data1)
  data1 = create_features(data1)
  data1 = feature_drops(data1)
  return data1

def add_missing_columns(data):
  if 'senior_citizen' not in data:
    data['senior_citizen'] = 0
  if 'partner' not in data:
    data['partner'] = 0
  if 'dependents' not in data:
    data['dependents'] = 0
  if 'tenure_months' not in data:
    data['tenure_months'] = 0
  if 'phone_service' not in data:
    data['phone_service'] = 0
  if 'multiple_lines' not in data:
    data['multiple_lines'] = 0
  if 'online_security' not in data:
    data['online_security'] = 0
  if 'online_backup' not in data:
    data['online_backup'] = 0
  if 'device_protection' not in data:
    data['device_protection'] = 0
  if 'tech_support' not in data:
    data['tech_support'] = 0
  if 'streaming_tv' not in data:
    data['streaming_tv'] = 0
  if 'streaming_movies' not in data:
    data['streaming_movies'] = 0
  if 'paperless_billing' not in data:
    data['paperless_billing'] = 0
  if 'monthly_charges' not in data:
    data['monthly_charges'] = 0
  if 'total_charges' not in data:
    data['total_charges'] = 0
  if 'churn_value' not in data:
    data['churn_value'] = 0
  if 'gender_female' not in data:
    data['gender_female'] = 0
  if 'gender_male' not in data:
    data['gender_male'] = 0
  if 'internet_svc_none' not in data:
    data['internet_svc_none'] = 0
  if 'internet_svc_dsl' not in data:
    data['internet_svc_dsl'] = 0
  if 'internet_svc_fiber' not in data:
    data['internet_svc_fiber'] = 0
  if 'contract_mtm' not in data:
    data['contract_mtm'] = 0
  if 'contract_1yr' not in data:
    data['contract_1yr'] = 0
  if 'contract_2yr' not in data:
    data['contract_2yr'] = 0
  if 'pmt_meth_bank_trx_auto' not in data:
    data['pmt_meth_bank_trx_auto'] = 0
  if 'pmt_meth_cc_auto' not in data:
    data['pmt_meth_cc_auto'] = 0
  if 'pmt_meth_echeck' not in data:
    data['pmt_meth_echeck'] = 0
  if 'pmt_meth_mail_check' not in data:
    data['pmt_meth_mail_check'] = 0
  return data

def data_input_preprocess(data):
  print(1)
  data1 = replace_yes_no(data)
  print(2)
  data1 = impute_total_charges(data1)
  print(3)
  data1 = ohe_categorical(data1)
  print(4)
  data1 = add_missing_columns(data1)
  print(5)
  data1 = map_products(data1)
  print(6)
  data1 = create_features(data1)
  print(7)
  data1 = feature_drops(data1)
  return data1
