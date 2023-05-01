from features import *
import argparse, time, os, sys, pickle
import pandas as pd
import numpy as np
import xgboost
# USING STORED FEATURES
FINAL_COLS = ['gufi', 'timestamp', 'airport', 'minutes_until_pushback']

MODEL_PATH = '/models/v5_model_{}_{}.json'
LABEL_PATH = 'data/raw/{}'
SAVE_PATH = 'data/processed/submission_{}_{}.csv'


# WE SHOULD JOINT SEQUENTIAL not load the data
# WE use pd.dummies in this submission
if 'cuontr' not in os.getcwd():
  prediction_pd = pd.read_csv("/data/raw/submission_format.csv", parse_dates=["timestamp"])

def get_perf(airport_name, lr):
    model = xgboost.XGBRegressor(
        predictor='cpu_predictor',
        colsample_bytree=0.9,
        gamma=0,
        learning_rate=lr,
        max_depth=3,
        min_child_weight=3,
        n_estimators=30000,
        reg_alpha=0.75,
        reg_lambda=0.45,
        subsample=0.6,
        eval_metric ='mae',
        objective="reg:absoluteerror",
        seed=42)

    np.random.seed(0)
    sub_file_name = SAVE_PATH.format(airport_name, lr)
    model_file_name = MODEL_PATH.format(airport_name, lr)
    submission_pd = prediction_pd[prediction_pd['airport'] == airport_name]
    # EXTRACT FEATURES FOR TRAINING DATA
    features = []
    cat_features = []

    submission_pd, feat_dict = get_diff_depart_vs_ts(submission_pd, airport_name=airport_name)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']
    submission_pd, feat_dict = get_weather_info(submission_pd, airport_name=airport_name)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']
    submission_pd, feat_dict = get_basic_feats(submission_pd)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']
    submission_pd, feat_dict = get_meta_features(submission_pd, airport_name=airport_name)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']
    submission_pd, feat_dict = get_dep_rw(submission_pd, airport_name=airport_name)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']
    submission_pd, feat_dict = get_num_dep_flights(submission_pd, airport_name=airport_name)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']

    values = {v: -1 for v in features if v not in cat_features}
    for v in cat_features:
        submission_pd[v] = submission_pd[v].astype('string')
        submission_pd[v].fillna('NA', inplace=True)
        submission_pd[v] = submission_pd[v].astype('category')
    submission_pd = submission_pd.fillna(values)
    submission_pd = pd.get_dummies(submission_pd, columns =cat_features)
    new_features = [x for x in features if x not in cat_features]
    # TODO we should save list of features 
    # new_features = [x for x in train_pd.columns.tolist() if x not in label_pd.columns.tolist()] +[ x for x in features if x not in cat_features]
    # for x in new_features:
    #     if x not in submission_pd.columns.tolist():
    #         submission_pd[x] = 0



    model = xgboost.XGBRegressor()
    model.load_model(model_file_name)

    y_pred = model.predict(submission_pd[new_features])
    submission_pd['minutes_until_pushback'] = y_pred
    submission_pd['minutes_until_pushback'] = submission_pd['minutes_until_pushback'].apply(lambda x: max(int(x), 1))

    submission_pd.to_csv(sub_file_name, index=False)

def main():
   starttime = time.time()
   parser = argparse.ArgumentParser(description='Test')
   parser.add_argument('--airport_name', default='KATL', type=str)
   parser.add_argument('--lr', default=1, type=float)
   args = parser.parse_args()
   get_perf(args.airport_name, args.lr)
   print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":
    main()


