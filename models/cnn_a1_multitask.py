# Convolutional neural network based on three-dimensional input data using a basic 4-layer architecture with Instance 
# normalization. This model does not incorporate other clinical or sociodemographic information, and the modelization
# is carried out using a multi-task loss function.
# This model corresponds to the baseline model with balanced objective functions.
#
# Overview:
#
#   - Input data: 3D volumenes only
#   - Normalization: Instance normalization
#   - Objective: predict ROIs metabolism variations
#   - Uses clinical and demographic information: False
#   - Laslayer: multi-task layer
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
from scipy.special import softmax
from nilearn import image

# inner dependencies
import src.loading as loading
from src.models import multitaskLossNaN_v2

# Gojo dependencies
from gojo import util
from gojo import core
from gojo import plotting
from gojo import deepl


# ============== Configuration parameters

DATA_LOADING_KEY = '3Dimages_data_selection_all_targets_v1'

# key to add to the exported model
MODEL_KEY = 'cnn_a1_multitask_balanced_nonclinical'

# number of folds for the hyperparameter optimization
CROSS_VALIDATION_FOLDS = 10
CROSS_VALIDATION_REPEATS = 1

# random state for replicability
RANDOM_SEED = 2000

# number of jobs used for model evaluation
N_JOBS = 1

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


def lossFunction(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """ Wrapper for the loss function """
    return multitaskLossNaN_v2(
        y_hat=y_hat,
        y_true=y_true,
        bce_weight=1.0,
        ce_weight=2.0,
        reg_weight=0.5,
        bce_class_weight=4.0,
        reg_delta=2.0
    )


def bceLoss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Metric to monitor the evolution of the loss component associated with pMCI/sMCI prediction. """
    # ---- bce_weight = 2.0
    # ---- bce_class_weight = 4.0
    
    # select BCE-associated entries
    y_true_ = y_true[:, 15]
    y_pred_ = y_pred[:, 15]
    
    # calculate classification loss for pMCI/sMCI
    with torch.no_grad():
        bce_loss = deepl.loss.weightedBCEwithNaNs(
            y_hat=torch.from_numpy(y_pred_), 
            y_true=torch.from_numpy(y_true_), 
            weight=4.0    # ---- bce_class_weight
        )
        
    return bce_loss.item() * 1.0   # ---- bce_weight

    
def mcLoss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Metric to monitor the evolution of the loss component associated with CN/MCI/Dem prediction. """
    # ---- ce_weight = 1.0
    
    # select BCE-associated entries
    y_true_ = y_true[:, 16:]
    y_pred_ = y_pred[:, 16:]
    
    # calculate classification loss for CN/MCI/Dem
    with torch.no_grad():
        mc_loss = torch.nn.functional.cross_entropy(
            input=torch.from_numpy(y_pred_), 
            target=torch.from_numpy(y_true_), 
            reduction='mean'
        )
        
    return mc_loss.item() * 2.0   # ---- ce_weight


def regLoss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Metric to monitor the evolution of the loss component associated with metabolism variation prediction. """
    # ---- reg_weight = 0.2
    # ---- reg_delta = 2.0
    
    # select metabolism variation associated entries
    y_true_ = y_true[:, :15]
    y_pred_ = y_pred[:, :15]
    
    # calculate classification loss for metabolism variations
    with torch.no_grad():
        reg_loss = deepl.loss.huberLossWithNaNs(
            y_hat=torch.from_numpy(y_pred_), 
            y_true=torch.from_numpy(y_true_), 
            delta=2.0    # ---- reg_delta
        )
        
    return reg_loss.item() * 0.5   # ---- reg_weight


def calculateAvgMetrics(df: pd.DataFrame, ref_label: int, problem: str):
    """ Subrutine used to calculate one-vs-rest metrics for multiclass classification problems. """
    metrics_ = []
    for _, sub_df in df.groupby('n_fold'):
        metrics_.append(core.getScores(
            y_pred=sub_df['pred'].values == ref_label,
            y_true=sub_df['true'].values == ref_label,
            metrics=core.getDefaultMetrics('binary_classification')
        ))
    metrics_ = pd.DataFrame(metrics_)
    metrics_mean = metrics_.mean()
    metrics_std = metrics_.std()
    
    stat_metrics = pd.concat([
        pd.DataFrame(metrics_mean, columns=['mean']),
        pd.DataFrame(metrics_std, columns=['std'])
    ], axis=1)
    
    stat_metrics['problem'] = problem
    stat_metrics = stat_metrics.reset_index().rename(columns={'index': 'metric'})
    stat_metrics = stat_metrics.set_index(['problem', 'metric'])

    return stat_metrics


if __name__ == '__main__':
    
    # model definition
    model = core.TorchSKInterface(
        model=deepl.models.MultiTaskFFNv2(
            feature_extractor=torch.nn.Sequential(
                    torch.nn.Conv3d(1, 8, kernel_size=1, stride=1),
                    torch.nn.InstanceNorm3d(8),
                    torch.nn.ReLU(),

                    torch.nn.Conv3d(8, 16, kernel_size=3, stride=1),
                    torch.nn.InstanceNorm3d(16),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool3d(2),

                    torch.nn.Conv3d(16, 32, kernel_size=3, stride=2),
                    torch.nn.InstanceNorm3d(32),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool3d(2),

                    torch.nn.Conv3d(32, 32, kernel_size=3, stride=1),
                    torch.nn.InstanceNorm3d(32),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool3d(2),

                    torch.nn.Flatten(),

                    torch.nn.Linear(384, 64),
                    torch.nn.Dropout(0.1),
                    torch.nn.ReLU(),
                ),
            multitask_projection=torch.nn.ModuleList(
                [       # regression tasks
                    torch.nn.Sequential(
                        torch.nn.Linear(64, 20),
                        torch.nn.ReLU(),
                        torch.nn.Linear(20, 1)
                    ) for _ in range(15)
                ] + [   # binary classification task
                    torch.nn.Sequential(
                        torch.nn.Linear(64, 20),
                        torch.nn.ReLU(),
                        torch.nn.Linear(20, 1),
                        torch.nn.Sigmoid()
                    )     
            ] + [  # multiclass classification task
                    torch.nn.Sequential(
                        torch.nn.Linear(64, 20),
                        torch.nn.ReLU(),
                        torch.nn.Linear(20, 3)
                    )        
            ])
        ),
        iter_fn=deepl.iterSupervisedEpoch,
        iter_fn_kw=dict(
            clear_cuda_cache=True
        ),
        loss_function=lossFunction,
        n_epochs=50,
        train_split=0.85,
        optimizer_class=torch.optim.AdamW,
        dataset_class=deepl.loading.StreamTorchDataset,
        dataloader_class=torch.utils.data.DataLoader,
        optimizer_kw=dict(
            lr=0.00005
        ),
        train_dataset_kw=dict(
            loading_fn=loadImage
        ),
        valid_dataset_kw=dict(
            loading_fn=loadImage
        ),
        train_dataloader_kw=dict(
            batch_size=48,
            shuffle=True,
            drop_last=True,
        ),
        valid_dataloader_kw=dict(
            batch_size=128,
        ),
        callbacks=[
           deepl.callback.EarlyStopping(it_without_improve=15, track='count')
        ],
        train_split_stratify=False,
        metrics=[
            core.Metric(
                name='bce_loss',
                function=bceLoss),
            core.Metric(
                name='ce_loss',
                function=mcLoss),
            core.Metric(
                name='huber_loss',
                function=regLoss)
        ],
        seed=RANDOM_SEED, 
        batch_size=64,
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
    
    # calculate ROI statistics for the regression problems
    roi_stats_test = []

    # separate regression predictions
    y_hat_reg_test = test_preds[['pred_labels_%d' % i for i in range(15)]]
    y_true_reg_test = test_preds[['true_labels_%d' % i for i in range(15)]]

    for i in range(y_hat_reg_test.shape[1]):
        # get ROI name
        roi = y.columns[i].replace('mean_', '').replace('_cerebellumVermisSUVR_DELTA', '')

        # get test predictions associated with the target ROI
        predictions = pd.DataFrame(
            {'pred': y_hat_reg_test['pred_labels_%d' % i].values, 
             'true': y_true_reg_test['true_labels_%d' % i].values,
             'n_fold': y_hat_reg_test.reset_index()['n_fold'].values
            })

        # remove missing values
        predictions = predictions.dropna()

        # compute statistics for each fold
        roi_metrics_ = []
        for _, sub_df in predictions.groupby('n_fold'):
            roi_metrics_.append(core.getScores(
                y_pred=sub_df['pred'].values, 
                y_true=sub_df['true'].values, 
                metrics=core.getDefaultMetrics('regression')))

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

    # calculate pMCI/sMCI classification performance
    pmci_stats_test = []

    # separate binary classification predictions
    y_hat_pmci_test = test_preds['pred_labels_%d' % 15]
    y_true_pmci_test = test_preds['true_labels_%d' % 15]

    predictions = pd.DataFrame({
        'pred': y_hat_pmci_test.values,
        'true': y_true_pmci_test.values,
        'n_fold': y_hat_pmci_test.reset_index()['n_fold'].values
    })

    # remove entries with missing information
    predictions = predictions.dropna()

    pmci_metrics_ = []
    for _, sub_df in predictions.groupby('n_fold'):
        pmci_metrics_.append(core.getScores(
            y_pred=sub_df['pred'].values > 0.5,
            y_true=sub_df['true'].values,
            metrics=core.getDefaultMetrics('binary_classification')
        ))

    pmci_metrics = pd.DataFrame(pmci_metrics_)

    pmci_metrics_mean = pmci_metrics.mean()
    pmci_metrics_std = pmci_metrics.std()

    pmci_metrics = pd.concat([
        pd.DataFrame(pmci_metrics_mean, columns=['mean']),
        pd.DataFrame(pmci_metrics_std, columns=['std'])
    ], axis=1)

    pmci_metrics = pmci_metrics.reset_index().rename(columns={'index': 'metric'})
    pmci_metrics = pmci_metrics.set_index('metric')
    
    # calculate HC/MCI/Dementia classification metrics in one-vs-rest fashion
    mc_stats_test = []

    # separate multiclass classification predictions
    y_hat_mc_test = test_preds[['pred_labels_%d' % i for i in range(16, 19)]]
    y_true_mc_test = test_preds[['true_labels_%d' % i for i in range(16, 19)]]

    # normalize from logits to probabilities
    y_hat_mc_test = pd.DataFrame(
        softmax(y_hat_mc_test, axis=1), 
        columns=y_hat_mc_test.columns,
        index=y_hat_mc_test.index)

    # integer coding of the predictions
    y_hat_mc_test = pd.DataFrame(
        y_hat_mc_test.values.argmax(axis=1),
        columns=['pred'],
        index=y_hat_mc_test.index
    )

    y_true_mc_test = pd.DataFrame(
        y_true_mc_test.values.argmax(axis=1),
        columns=['true'],
        index=y_true_mc_test.index
    )

    # get test predictions associated with the target ROI
    predictions = pd.DataFrame(
        {'pred': y_hat_mc_test['pred'].values,
         'true': y_true_mc_test['true'].values,
         'n_fold': y_true_mc_test.reset_index()['n_fold'].values
        })

    hc_vs_rest  = calculateAvgMetrics(df=predictions, ref_label=0, problem='HCvsRest')
    mci_vs_rest = calculateAvgMetrics(df=predictions, ref_label=1, problem='MCIvsRest')
    dem_vs_rest = calculateAvgMetrics(df=predictions, ref_label=2, problem='DemvsRest')
    hc_vs_dem = calculateAvgMetrics(
        df=predictions.loc[predictions['true'].isin([0, 2])], ref_label=2, problem='HCvsDem')

    multiclass_metrics = pd.concat([
        hc_vs_rest,
        mci_vs_rest,
        dem_vs_rest,
        hc_vs_dem
    ], axis=0)
        
    # save metadata
    cv_report.addMetadata(roi_stats_test=roi_stats_test_df)
    cv_report.addMetadata(pmci_metrics=pmci_metrics)
    cv_report.addMetadata(multiclass_metrics=multiclass_metrics)

    # save the report
    util.io.serialize(
        cv_report,
        path=os.path.join(
            REPORT_OUTPUT_PATH, '%s-%s-%s.joblib' % (DATA_LOADING_KEY, 'optimized_model_v5', MODEL_KEY)),
        time_prefix=True, backend='joblib_gzip'
    )

