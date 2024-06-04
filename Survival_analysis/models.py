import warnings
import pandas as pd
import numpy as np
from sksurv.datasets import get_x_y
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score, concordance_index_censored, brier_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import t
from scipy import stats
import yaml


from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from lifelines.utils import k_fold_cross_validation

def cox_ph_grid_search(df, penalizer_values, duration_col, event_col, verbose, l1_ratio=0.0, k=5):
    best_penalizer = None
    best_c_index = -np.inf

    for penalizer in penalizer_values:
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        
        # Perform k-fold cross-validation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            scores = k_fold_cross_validation(cph, df, duration_col=duration_col, event_col=event_col, k=k, scoring_method="concordance_index")
        
        mean_score = np.mean(scores)
        if verbose:
            print(f"Penalizer: {penalizer}, Mean C-index: {mean_score}")
        if mean_score > best_c_index:
            best_c_index = mean_score
            best_penalizer = penalizer
    if verbose:
        print(f"Best penalizer: {best_penalizer}, Best C-index: {best_c_index}")
    return best_penalizer, best_c_index



def cox_ph(df_train, df_test, cross_validate=True, verbose=True, l1_ratio=0.0, 
           pen_val=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
    if cross_validate:
        penalizer_values = pen_val
        duration_col = 'M'
        event_col = 'Event'
        penalizer, best_c_index = cox_ph_grid_search(df_train, penalizer_values, duration_col, event_col, l1_ratio=l1_ratio, verbose=verbose, k=5)
    else:
        penalizer = 0
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df_train, duration_col='M', event_col='Event', show_progress=verbose)
    c_index = cph.score(df_test, 'concordance_index')
    return cph, c_index, penalizer





def survival_forest(df_train, df_test, config=None, cross_validate=True, verbose=True):
    random_state = 42

    x_train, y_train = get_x_y(df_train, attr_labels=["Event", "M"], pos_label=1)
    x_test, y_test = get_x_y(df_test, attr_labels=["Event", "M"], pos_label=1)

    if cross_validate:
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'min_samples_split': [4, 6, 8, 10],
            'min_samples_leaf': [4, 6, 8, 10],
            'max_features': ['sqrt', 'log2']
        }

        rsf_test = RandomSurvivalForest(n_jobs=-1, random_state=random_state)

        cv = KFold(n_splits=5, random_state=random_state, shuffle=True)
        grid_search = GridSearchCV(rsf_test, 
                                param_grid, 
                                cv=cv, 
                                error_score=0.5,
                                n_jobs=12, 
                                verbose=3)

        grid_search.fit(x_train, y_train)
        print(grid_search.best_params_)

        n_estimators = grid_search.best_params_['n_estimators']
        max_features = grid_search.best_params_['max_features']
        min_samples_split = grid_search.best_params_['min_samples_split']
        min_samples_leaf = grid_search.best_params_['min_samples_leaf']
    
    elif config != None: 
        n_estimators = config['Best Parameters']['n_estimators']
        max_features = config['Best Parameters']['max_features']
        min_samples_split = config['Best Parameters']['min_samples_split']
        min_samples_leaf = config['Best Parameters']['min_samples_leaf']
        
    else:  
        n_estimators = 200
        max_features = 'sqrt'
        min_samples_split = 4
        min_samples_leaf = 4

    rsf = RandomSurvivalForest(n_estimators=n_estimators, 
                                min_samples_split=min_samples_split, 
                                min_samples_leaf=min_samples_leaf, 
                                max_features=max_features,
                                n_jobs=1, 
                                random_state=random_state,
                                verbose=verbose)

    # Fit the model on the training data
    rsf.fit(x_train, y_train)

    score = rsf.score(x_test, y_test)
    X_test = rsf.predict(x_test)
    score_ipcw = concordance_index_ipcw(y_train, y_test, X_test)
    
    lower, upper = np.percentile(y_train["M"], [10, 90])
    time_points = np.arange(lower, upper + 1)
    
    surv_prob = np.row_stack([
        fn(time_points)
        for fn in rsf.predict_survival_function(x_test)
    ])
    score_bri = integrated_brier_score(y_train, y_test, surv_prob, time_points)
    rsf.predict(x_test)
    return rsf, score, score_ipcw, score_bri


def rubin_eval_cox(models):    
    
    coefficients = np.array([model.params_.values for model in models])
    variances = np.array([model.variance_matrix_.values.diagonal() for model in models])
    column_names = models[0].params_.index  # Column names from the first model

    Q_bar = np.mean(coefficients, axis=0)  # Pooled coefficients
    U_bar = np.mean(variances, axis=0)     # Within-imputation variance
    B = np.var(coefficients, axis=0, ddof=1)  # Between-imputation variance

    # Total variance
    T = U_bar + (1 + 1/len(models)) * B

    # Standard errors
    se = np.sqrt(T)

    # Confidence intervals
    alpha = 0.05
    z_value = stats.norm.ppf(1 - alpha / 2)
    ci_lower = Q_bar - z_value * se
    ci_upper = Q_bar + z_value * se

    # Hazard ratios
    exp_coef = np.exp(Q_bar)
    exp_ci_lower = np.exp(ci_lower)
    exp_ci_upper = np.exp(ci_upper)

    # Z-scores and p-values
    z_scores = Q_bar / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    # Combine results into a DataFrame for easy viewing
    results = pd.DataFrame({
        'coef': Q_bar,
        'exp(coef)': exp_coef,
        'se(coef)': se,
        'coef lower 95%': ci_lower,
        'coef upper 95%': ci_upper,
        'exp(coef) lower 95%': exp_ci_lower,
        'exp(coef) upper 95%': exp_ci_upper,
        'z': z_scores,
        'p': p_values,
        '-log2(p)': -np.log2(p_values)
    }, index=column_names)

    return results

def set_coef_test(coef, dfs_train, dfs_test, baseline_hazards):


    model = CoxPHSurvivalAnalysis(alpha=0.1)
    x_train, y_train = get_x_y(dfs_train[0], attr_labels=["Event", "M"], pos_label=1)


    model.fit(x_train, y_train)
    model.coef_ = coef
    
    concatenated_baseline_hazards = pd.concat(baseline_hazards, axis=1)
    pooled_baseline_hazard = concatenated_baseline_hazards.mean(axis=1)
    model.baseline_hazard_ = pooled_baseline_hazard

    scores_c_ind = []
    scores_ipcw = []
    scores_ibs = []

    lower, upper = np.percentile(y_train["M"], [10, 90])
    time_points = np.arange(lower, upper + 1)

    for df_train, df_test in zip(dfs_train, dfs_test):
        x_train, y_train = get_x_y(df_train, attr_labels=["Event", "M"], pos_label=1)
        x_test, y_test = get_x_y(df_test, attr_labels=["Event", "M"], pos_label=1)

        prediction = model.predict(x_test)
        score_c_ind = concordance_index_censored(y_test["Event"], 
                                                y_test["M"], 
                                                prediction)
        X_test = model.predict(x_test)
        score_ipcw = concordance_index_ipcw(y_train, y_test, X_test)
        
        surv_prob = np.row_stack([
            fn(time_points)
            for fn in model.predict_survival_function(x_test)
        ])
        
        ibs = integrated_brier_score(y_train, y_test, surv_prob, time_points)

        scores_c_ind.append(score_c_ind[0])
        scores_ipcw.append(score_ipcw[0])
        scores_ibs.append(ibs)

        # print(f"Result of model after pooling: {score[0], score_ipcw[0]}")
    return scores_c_ind, scores_ipcw, scores_ibs
