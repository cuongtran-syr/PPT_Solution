from features import *
from qa_features import  *
import argparse, time, os, sys, pickle
import pandas as pd
import numpy as np
import xgboost
# USING STORED FEATURES
FINAL_COLS = ['gufi', 'timestamp', 'airport', 'minutes_until_pushback']

MODEL_PATH = '/models/v5_model_{}_{}.json'
LABEL_PATH = 'data/raw/{}'

# WE SHOULD JOINT SEQUENTIAL not load the data
# WE use pd.dummies in this submission
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
    model_file_name = MODEL_PATH.format(airport_name, lr)
    pushback = pd.read_csv(LABEL_PATH.format(airport_name) + 'train_labels_{}.csv.bz2'.format(airport_name),
                           parse_dates=["timestamp"])
    # EXTRACT FEATURES FOR TRAINING DATA
    features = []
    cat_features = []
    label_pd, feat_dict = get_diff_depart_vs_ts(pushback, airport_name=airport_name)  # extract_training data
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']
    label_pd, feat_dict = get_weather_info(label_pd, airport_name=airport_name)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']
    label_pd, feat_dict = get_basic_feats(label_pd)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']
    label_pd, meta_feat_dict = get_meta_features(label_pd, airport_name=airport_name)
    features += meta_feat_dict['numeric'] + meta_feat_dict['cat']
    cat_features += meta_feat_dict['cat']
    label_pd, rw_feat_dict = get_dep_rw(label_pd, airport_name=airport_name)
    features += rw_feat_dict['numeric'] + rw_feat_dict['cat']
    cat_features += rw_feat_dict['cat']
    label_pd, num_dep_feat_dict = get_num_dep_flights(label_pd, airport_name=airport_name)
    features +=num_dep_feat_dict['numeric'] + num_dep_feat_dict['cat']
    cat_features += num_dep_feat_dict['cat']
    label_pd['airline'] = label_pd['gufi'].apply(get_airline)
    label_pd, feat_dict = get_departure_mfs_count_fts(label_pd, airport_name)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']

    ############
    label_pd, feat_dict = get_arrival_mfs_count_fts(label_pd, airport_name)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']
    label_pd, feat_dict = get_number_active_runways(label_pd, airport_name)
    features += feat_dict['numeric'] + feat_dict['cat']
    cat_features += feat_dict['cat']

    values = {v: -1 for v in features if v not in cat_features}
    for v in cat_features:
        label_pd[v] = label_pd[v].astype('string')
        label_pd[v].fillna('NA', inplace=True)
        label_pd[v] = label_pd[v].astype('category')
    label_pd = label_pd.fillna(values)

    label_pd['is_train'] = label_pd['timestamp'].apply(lambda x: np.random.rand()>=0.3).astype(int)
    # SPLIT TRAIN/TEST
    train_pd = pd.get_dummies( label_pd[label_pd['is_train'] == 1], columns = cat_features)
    test_pd = pd.get_dummies( label_pd[label_pd['is_train'] == 0], columns = cat_features)
    new_features = [x for x in features if x not in cat_features]
    new_features = [x for x in train_pd.columns.tolist() if x not in label_pd.columns.tolist()] +[ x for x in features if x not in cat_features]

    X_train = train_pd[new_features]
    X_test = test_pd[new_features]
    y_train = train_pd['minutes_until_pushback'].values
    y_test = test_pd['minutes_until_pushback'].values

    model.fit(X_train, y_train, verbose=500, early_stopping_rounds=10,
              eval_set=[(X_test, y_test)])
    model.save_model(model_file_name)

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


