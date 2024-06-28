# Convolutional neural network based on three-dimensional input data using a basic 4-layer architecture with Instance 
# normalization. This model does not incorporate other clinical or sociodemographic information.
#
# Overview:
#
#   - Input data: 3D volumenes only
#   - Normalization: Instance normalization
#   - Objective: predict ROIs metabolism variations
#   - Uses clinical and demographic information: False
#   - Laslayer: independent across ROIs
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
from sklearn import metrics as sk_metrics
from nilearn import image

# inner dependencies
import src.loading as loading

# Gojo dependencies
from gojo import util
from gojo import core
from gojo import plotting
from gojo import deepl


# ============== Configuration parameters

DATA_LOADING_KEY = '3Dimages_data_selection_v1'

# key to add to the exported model
MODEL_KEY = 'cnn_a2_simple_nonclinical'

# number of folds for the hyperparameter optimization
CROSS_VALIDATION_FOLDS = 10
CROSS_VALIDATION_REPEATS = 1

# random state for replicability
RANDOM_SEED = 2000

# number of jobs used for model evaluation
N_JOBS = 1

# metrics used to evaluate binary regression problems
REGRESSION_METRICS = core.getDefaultMetrics('regression')

# parameter indicating whether to save the model predictions on the training data
SAVE_TRAIN_PREDS = False 

# parameter indicating whether to save the trained models for each fold
SAVE_MODELS = True

# parameter indicating whether to save the trained transformations for each fold
SAVE_TRANSFORMS = False

# directory where the CV report will be saved
REPORT_OUTPUT_PATH = os.path.join('..', 'result', 'sel_roi_models_v5')

def loadImage(file: str) -> torch.Tensor:
    """ Subrutine used to load an image from a given file """
    nii_img = image.load_img(file)
    nii_img_data = np.array(nii_img.get_fdata()).astype(np.float32)
    nii_img_data = np.expand_dims(nii_img_data, axis=0)  # add channel
    
    return torch.from_numpy(nii_img_data)


if __name__ == '__main__':
    
    # model definition
    model = core.TorchSKInterface(
        model=deepl.models.MultiTaskFFNv2(
            feature_extractor=torch.nn.Sequential(
                torch.nn.Conv3d(1, 16, kernel_size=1, stride=1),
                torch.nn.InstanceNorm3d(16),
                torch.nn.ReLU(),

                torch.nn.Conv3d(16, 32, kernel_size=3, stride=1),
                torch.nn.InstanceNorm3d(32),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2),

                torch.nn.Conv3d(32, 48, kernel_size=4, stride=2),
                torch.nn.InstanceNorm3d(48),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2),

                torch.nn.Conv3d(48, 48, kernel_size=3, stride=1),
                torch.nn.InstanceNorm3d(48),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d(2),

                torch.nn.Flatten(),

                torch.nn.Linear(576, 128),
                torch.nn.Dropout(0.2),
                torch.nn.ReLU(),
                
                torch.nn.Linear(128, 64),
                torch.nn.Dropout(0.2),
                torch.nn.ReLU(),
            ),
            multitask_projection=torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(64, 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20, 1)
                ) for _ in range(15)
            ])
        ),
        iter_fn=deepl.iterSupervisedEpoch,
        iter_fn_kw=dict(
            clear_cuda_cache=True
        ),
        loss_function=torch.nn.HuberLoss(delta=2.0, reduction='mean'),
        n_epochs=50,
        train_split=0.85,
        optimizer_class=torch.optim.AdamW,
        dataset_class=deepl.loading.StreamTorchDataset,
        dataloader_class=torch.utils.data.DataLoader,
        optimizer_kw=dict(
            lr=0.0001
        ),
        train_dataset_kw=dict(
            loading_fn=loadImage
        ),
        valid_dataset_kw=dict(
            loading_fn=loadImage
        ),
        train_dataloader_kw=dict(
            batch_size=20,
            shuffle=True,
            drop_last=True,
        ),
        valid_dataloader_kw=dict(
            batch_size=64,
        ),
        callbacks=[
           deepl.callback.EarlyStopping(it_without_improve=15, track='count')
        ],
        train_split_stratify=False,
        metrics=REGRESSION_METRICS,
        seed=RANDOM_SEED, 
        batch_size=20,
        device='cuda',
        verbose=1
    )

    print(util.tools.getNumModelParams(model.model))
    
    # transforms definition
    transforms = None

    # load the input data
    X, y, indices_id, var_args = loading.getDataset(DATA_LOADING_KEY)

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
        transforms=None,
        n_jobs=N_JOBS,
        save_train_preds=SAVE_TRAIN_PREDS,
        save_transforms=False,
        save_models=SAVE_MODELS
    )

    # extract report statistics
    models = list(cv_report.getTrainedModels().values())
    test_preds = cv_report.getTestPredictions()

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

    cv_report.addMetadata(roi_stats_test=roi_stats_test_df)

    # save the report
    util.io.serialize(
        cv_report,
        path=os.path.join(
            REPORT_OUTPUT_PATH, '%s-%s-%s.joblib' % (DATA_LOADING_KEY, 'optimized_model_v5', MODEL_KEY)),
        time_prefix=True, backend='joblib_gzip'
    )



