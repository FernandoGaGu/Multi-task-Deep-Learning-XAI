# FFN-based model fitted to model changes in brain metabolism of selected ROIs from tabular information including mean 
# SUVR values. No clinical variables were considered.
#
# Author: Fernando García-Gutiérrez
# Email: fegarc05@ucm.es
#
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as sk_metrics
from scipy.stats import spearmanr

# inner dependencies
import src.loading as loading

# Gojo dependencies
from gojo import util
from gojo import core
from gojo import plotting
from gojo import deepl



# ============== Configuration parameters

# key used to export the models
DATA_LOADING_KEY = 'tabular_data_selection_nostd_v1'

# key to add to the exported model
MODEL_KEY = 'ffn_simple_nonclinical'

# cross-validation configuration
CROSS_VALIDATION_FOLDS = 10
CROSS_VALIDATION_REPEATS = 1

# random state for replicability
RANDOM_SEED = 2000

# number of jobs used for model evaluation
N_JOBS = 10

# metrics used to evaluate binary regression problems
REGRESSION_METRICS = core.getDefaultMetrics('regression')

# parameter indicating whether to save the model predictions on the training data
SAVE_TRAIN_PREDS = True 

# parameter indicating whether to save the trained models for each fold
SAVE_MODELS = True

# parameter indicating whether to save the trained transformations for each fold
SAVE_TRANSFORMS = True

# directory where the CV report will be saved
REPORT_OUTPUT_PATH = os.path.join('..', 'result', 'sel_roi_models_v5')

if __name__ == '__main__':
    
    # model definition
    model = core.TorchSKInterface(
        model=deepl.models.MultiTaskFFN(
            in_feats=90,
            emb_feats=75,
            layer_dims=[250, 175, 100],
            layer_dropout=[0.15, 0.1, 0.05],
            layer_activation='ReLU',
            batchnorm=True,
            n_reg_task=15,
            n_clf_task=0,
            multt_layer_dims=[25],
            multt_layer_activation='ReLU'),
        iter_fn=deepl.iterSupervisedEpoch,
        loss_function=torch.nn.HuberLoss(delta=2.0, reduction='mean'),
        n_epochs=500,
        train_split=0.85,
        optimizer_class=torch.optim.AdamW,
        dataset_class=deepl.loading.TorchDataset,
        dataloader_class=torch.utils.data.DataLoader,
        optimizer_kw=dict(
            lr=0.000075     
        ),
        train_dataloader_kw=dict(
            batch_size=32,
            shuffle=True,
            drop_last=True
        ),
        valid_dataloader_kw=dict(
            batch_size=256,
        ),
        callbacks=[
            deepl.callback.EarlyStopping(it_without_improve=15, track='count')
        ],
        train_split_stratify=False,
        metrics=REGRESSION_METRICS,
        seed=RANDOM_SEED,    
        device='cuda',
        verbose=1
    )

    # transforms definition
    transforms = [
        core.transform.SKLearnTransformWrapper(StandardScaler)]

    # load the input data
    X, y, indices_id = loading.getDataset(DATA_LOADING_KEY)

    # remove clinical and sociodemographic variables
    X = X.drop(columns=['AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4']).copy()

    # cross-validation iterator
    cv_iter = util.InstanceLevelKFoldSplitter(
        n_splits=CROSS_VALIDATION_FOLDS,
        instance_id=indices_id,
        n_repeats=CROSS_VALIDATION_REPEATS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    # perform cross validation
    cv_report = core.evalCrossVal(
        X=X,
        y=y,
        model=model,
        cv=cv_iter,
        transforms=transforms,
        n_jobs=N_JOBS,
        save_train_preds=SAVE_TRAIN_PREDS,
        save_transforms=SAVE_TRANSFORMS,
        save_models=SAVE_MODELS
    )

    # extract report statistics
    models = list(cv_report.getTrainedModels().values())
    test_preds = cv_report.getTestPredictions()
    train_preds = cv_report.getTrainPredictions()

    # calculate ROI statistics
    roi_stats_test = []
    for i in range(test_preds.shape[1] // 2):
        # get ROI name
        roi = y.columns[i].replace('mean_', '').replace('_cerebellumVermisSUVR_DELTA', '')
        
        # get test predictions associated with the target ROI
        predictions = pd.DataFrame(
            {'pred': test_preds['pred_labels_%d' % i].values, 
             'true': test_preds['true_labels_%d' % i].values,
             'n_fold': test_preds.reset_index()['n_fold'].values
            })

        
        # compute statistics for each fold
        roi_metrics_ = []
        for _, sub_df in predictions.groupby('n_fold'):
            roi_metrics_.append(core.getScores(
                y_pred=sub_df['pred'].values, 
                y_true=sub_df['true'].values, 
                metrics=REGRESSION_METRICS))
        
        roi_metrics_df_ = pd.DataFrame(roi_metrics_)
        roi_metrics_df_['explained_variance'] = roi_metrics_df_['explained_variance'] * 100
        roi_metrics_mean = roi_metrics_df_.mean()
        roi_metrics_std = roi_metrics_df_.std()
        
        roi_metrics = pd.concat([
            pd.DataFrame(roi_metrics_mean, columns=['mean']),
            pd.DataFrame(roi_metrics_std, columns=['std'])], axis=1)
        roi_metrics['roi'] = roi
        roi_metrics = roi_metrics.reset_index().rename(columns={'index': 'metric'})
        roi_metrics = roi_metrics.set_index(['metric', 'roi'])
        
        roi_stats_test.append(roi_metrics)
        
    roi_stats_test_df = pd.concat(roi_stats_test, axis=0).sort_index()

    # save ROI statistics in the report
    cv_report.addMetadata(roi_stats_test=roi_stats_test_df)

    # save the report
    util.io.serialize(
        cv_report,
        path=os.path.join(
            REPORT_OUTPUT_PATH, '%s-%s-%s.joblib' % (DATA_LOADING_KEY, 'optimized_model_v5', MODEL_KEY)),
        time_prefix=True, backend='joblib_gzip'
    )



