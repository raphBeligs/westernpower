#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import pacf
import sys
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, BayesianRidge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
from tqdm import tqdm
import common_functions_by_date as cfbd
import datetime
from scipy.signal import correlate

class DataPreprocesser:
    def create_lag_features(df, field,exog_andendog_cols, lags_list=None):
   
        scaler = StandardScaler()
        if lags_list is None:
            partial = pd.Series(data=pacf(df[field].values, nlags=48))
            lags = list(partial[np.abs(partial) >= 0.2].index)
            lags.remove(0)
        else:
            lags = lags_list
        
        # avoid to insert the time series itself
        features = df[exog_andendog_cols].copy()
        for l in lags:
            features[f"lag_{l}"] = df[field].shift(l)
    
        features = pd.DataFrame(scaler.fit_transform(features[features.columns]),
                            columns=features.columns)
        features.index = df.index
    
        return features, scaler, lags
    def scale_features(features):
        features_df = features.copy()
        scaler = StandardScaler()
        features_df = pd.DataFrame(scaler.fit_transform(features_df[features_df.columns]),
                            columns=features_df.columns)
        features_df.index = features.index
        return features_df
    def apply_phaze_correction_on_signal(signal, phaze):
        if phaze < 0:
            phaze_signal = np.concatenate((signal[abs(phaze):], np.ones(abs(phaze))*signal[np.isnan(signal) == False][-1]))
            phaze_signal[np.isnan(phaze_signal)] = signal[np.isnan(signal) == False][-1]
            phaze_signal[np.isnan(signal)] = np.nan
            return phaze_signal
        elif phaze > 0:
            phaze_signal = np.concatenate((np.ones(abs(phaze))*signal[np.isnan(signal) == False][0], signal[:-abs(phaze)]))
            phaze_signal[np.isnan(phaze_signal)] = signal[np.isnan(signal) == False][-1]
            phaze_signal[np.isnan(signal)] = np.nan
            return phaze_signal
        else:
            return signal
    def force_zero_irradiance_for_smooth_signal(smooth_irr, irr):
        if irr == 0:
            return 0
        else:
            return smooth_irr
    def smooth_preprocess_signal(df, signal_field, smooth_type = 'mean', smooth_k=5):
        signals_df = df.copy()
        smooth_field = 'smooth_{}_{}'.format(smooth_type, signal_field)
        if smooth_type == 'mean':
            signals_df[smooth_field] = DataPreprocesser.moving_mean_avarage_smoothing(signals_df[signal_field].values, smooth_k)
        smooth_phaze = DataPreprocesser.compute_phaze_shift(signals_df[signals_df[signal_field].notna()][signal_field].values, 
                                       signals_df[signals_df[signal_field].notna()][smooth_field].values)
        print('smooth_phaze :', smooth_phaze)
        signals_df[smooth_field] = DataPreprocesser.apply_phaze_correction_on_signal(signals_df[smooth_field].values, smooth_phaze)
        if signal_field == 'irradiance_Wm-2':
            signals_df[smooth_field] = signals_df.apply(lambda x: DataPreprocesser.force_zero_irradiance_for_smooth_signal(
                x[smooth_field], x[signal_field]), axis=1)
        return signals_df
    def ajust_phaze_shift_of_weather_columns(df, field_ref, field_to_phaze, weather_cols):
        signals_df = df.copy()
        dephasage = DataPreprocesser.compute_phaze_shift(signals_df[signals_df[field_ref].notna()][field_ref].values, 
                                    signals_df[signals_df[field_ref].notna()][field_to_phaze].values)
        print('weather dephasage :', dephasage)
        for col_name in weather_cols:
            signals_df[col_name] = DataPreprocesser.apply_phaze_correction_on_signal(signals_df[col_name].values, dephasage)
        return signals_df
    def moving_mean_avarage_smoothing(X,k):
        S = np.zeros(X.shape[0])
        for t in range(X.shape[0]):
            if t < k:
                S[t] = np.mean(X[:t+1])
            else:
                S[t] = np.sum(X[t-k:t])/k
        return S
    
    def compute_phaze_shift(x1, x2):
        xcorr = correlate(x1,x2)
        nsamples = x1.size
        dt = np.arange(1-nsamples, nsamples)
        return dt[xcorr.argmax()]
    def get_square_irradiance(df, columns):
        square_df = df.copy()
        for column in columns:
            square_df['square_{}'.format(column)] = square_df[column]**2
        return square_df
    def get_solar_temp_product_columns(df, locations):
        product_df = df.copy()
        for i in locations:
            product_df['solar_temp_location{}'.format(str(i))] = product_df['solar_location{}'.format(str(i))] * product_df['temp_location{}'.format(str(i))]
        return product_df
    def build_lags_periods_columns_names(lags_period_list):
        return ['lag_{}'.format(str(i)) for i in (np.array(lags_period_list)*48*7).tolist()], [i*48*7 for i in lags_period_list]
    def get_lags_median_and_mean_train_df(df, field, lags_list, lags_cols, first_day_pred):
        df_with_lags = df.loc[df.index.date < first_day_pred,[field]].copy()
        for i in range(len(lags_list)):
            df_with_lags[lags_cols[i]] = df_with_lags[field].shift(lags_list[i])
        df_with_lags['median_{}'.format(field)] = df_with_lags[lags_cols].median(axis=1).values
        df_with_lags['mean_{}'.format(field)] = df_with_lags[lags_cols].mean(axis=1).values
        return df_with_lags
    def get_lags_median_and_mean_predicted_df(df_with_lags, lags_cols, first_day_pred, field):
        lags_cols_for_prediction = [field] + lags_cols[:-1]
        df_with_lags_pred = df_with_lags.loc[(df_with_lags.index.date >= first_day_pred+datetime.timedelta(days=-7)) & 
                                             (df_with_lags.index.date < first_day_pred), lags_cols_for_prediction].copy()
        df_with_lags_pred.index = df_with_lags_pred.index + pd.DateOffset(days=7)
        dict_col = {}
        for i in range(len(lags_cols)):
            dict_col[lags_cols_for_prediction[i]] = lags_cols[i]
        df_with_lags_pred = df_with_lags_pred.rename(columns=dict_col)
        df_with_lags_pred['median_{}'.format(field)] = df_with_lags_pred[lags_cols].median(axis=1).values
        df_with_lags_pred['mean_{}'.format(field)] = df_with_lags_pred[lags_cols].mean(axis=1).values
        return df_with_lags_pred
    
    def get_lags_and_median_and_mean_predict_and_train_df(df, field,first_day_pred,lags_period_list=[1,2,3]):
        lags_names_cols, lags_list = DataPreprocesser.build_lags_periods_columns_names(lags_period_list)
        df_with_lags = DataPreprocesser.get_lags_median_and_mean_train_df(df, field, lags_list, lags_names_cols, first_day_pred)
        df_with_lags_pred = DataPreprocesser.get_lags_median_and_mean_predicted_df(df_with_lags, lags_names_cols, first_day_pred,field)
        all_df_with_lags = pd.concat([df_with_lags, df_with_lags_pred])
        return all_df_with_lags

class Forecaster:
    
    def rectify_recursive_forecast(self,y, model_recursive, lags, exog_and_endog_features, 
                       n_steps=48*7, step="30T",model_rectify_type = 'KNN', only_pred=True):
    
        """
        Parameters
        ----------
        y: pd.Series holding the input time-series to forecast
        model: pre-trained machine learning model
        lags: list of lags used for training the model
        n_steps: number of time periods in the forecasting horizon
        step: forecasting time period
    
        Returns
        -------
        fcast_values: pd.Series with forecasted values 
        """
        
        def get_best_knn(X,y):
            param_grid = {'n_neighbors': np.arange(20,60).tolist()}
            search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)
            search.fit(X,y)
            print(search.best_estimator_)
            return search.best_estimator_
        def predict_knn(X_train,y_train, X_to_pred, best_rectify_estimator):
            if best_rectify_estimator is None:
                best_rec_estimator = get_best_knn(X_train,y_train)
                best_model = best_rec_estimator
                best_model.fit(X_train, y_train)
                return best_model.predict(X_to_pred), best_rec_estimator
            else:
                best_model = best_rectify_estimator
                best_model.fit(X_train, y_train)
                return best_model.predict(X_to_pred), None
    
        def predict_next_value(y_df, exog_and_endog_df, lags_list, datetime_pred, model, best_rectify_estimator):
            df = exog_and_endog_df.copy()
            y_rec = y_df.copy()
            y_rec.loc[datetime_pred, :] = 0
            df = pd.merge(y_rec, df, how='inner',left_index=True, right_index=True)
            features, _, _ = DataPreprocesser.create_lag_features(df, y_df.columns[0],exog_and_endog_df.columns, lags_list)
            prediction_rec = model.predict(features[features.index == datetime_pred])
            y_rec_rec = y_df.copy()
            y_rec_rec.loc[datetime_pred, :] = prediction_rec
            exog_and_endog_rec_rec = exog_and_endog_df.copy()
            X_train_rec_rec = pd.merge(y_rec_rec, exog_and_endog_rec_rec, how='inner',left_index=True, right_index=True)
            features_rec_rec, _, _ = DataPreprocesser.create_lag_features(X_train_rec_rec, y_df.columns[0],exog_and_endog_df.columns, lags_list + [0])
            features_rec_rec = features_rec_rec.dropna(how='any')
            columns_rec_rec = features_rec_rec.columns
            residuals = y_df.copy()
            rediduals = residuals[residuals.index >= features_rec_rec.index[0]][y_df.columns[0]].values - model.predict(
                features_rec_rec[features.columns][:-1].values)
            if model_rectify_type == 'KNN':
                residual_predict, best_rec_estimator = predict_knn(features_rec_rec[:-1],rediduals, 
                                                               features_rec_rec.values[-1:], best_rectify_estimator=best_rectify_estimator)
            if model_rectify_type == 'Ridge':
                residual_predict = predict_ridge(features_rec_rec[:-1],rediduals, features_rec_rec.values[-1:])
            prediction_rec_rec = prediction_rec + residual_predict
            y_df.loc[datetime_pred, :] = prediction_rec_rec
            return (y_df, best_rec_estimator)
    
        # get the dates to forecast
        pred_datetime = y.index[-1] + pd.Timedelta(minutes=30)
        fcast_range = pd.date_range(pred_datetime, 
                                periods=n_steps, 
                                freq=step)
        fcasted_values = []
        target = y.copy()
        best_rectify_estimator = None
        with tqdm(total=len(fcast_range), file=sys.stdout) as pbar:
            for datetime in fcast_range:
                if datetime == pred_datetime:
                    print(datetime)
                    target, best_rec_estimator = predict_next_value(target, exog_and_endog_features, 
                                                                    lags, datetime, model_recursive, best_rectify_estimator=None)
                    best_rectify_estimator = best_rec_estimator
                else:
                    target, _ = predict_next_value(target, exog_and_endog_features, 
                                                                    lags, datetime, model_recursive, best_rectify_estimator=best_rectify_estimator)
            
                pbar.update()
        if only_pred:
            return target[target.index >= pred_datetime]
        else:
            return target
        
    def recursive_rectify_weeks_before(self, df, field,first_day_pred, endog_exog_df, lags, predictions_path, recursive_ml_model, 
                                       nb_days_for_train=365, weeks_before_nb=4):
        for i in range(weeks_before_nb+1):
            pred_date = first_day_pred + datetime.timedelta(days=-i*7)
            print('week prediction with start day : ', pred_date)
            lags_endog_and_exog, _, lags= DataPreprocesser.create_lag_features(df[(df.index.date < pred_date) & 
                                                             (df.index.date >= pred_date+datetime.timedelta(days=-nb_days_for_train))], 
                                                          field,endog_exog_df.columns.tolist(), lags)
            X_train_pred = lags_endog_and_exog[(max(lags)+1):].values
            y_train_pred = df.loc[(df.index.date < pred_date) & (df.index.date >= pred_date+datetime.timedelta(days=-nb_days_for_train)), 
                                     field][(max(lags)+1):].values
            model = recursive_ml_model
            model.fit(X_train_pred, y_train_pred)
        
            rrf = self.rectify_recursive_forecast(df.loc[(df.index.date < pred_date) & (df.index.date >= pred_date+datetime.timedelta(days=-nb_days_for_train)), 
                                                [field]], model, lags,endog_exog_df)
            directory = os.path.join(predictions_path, field)
            if not os.path.exists(directory):
                os.makedirs(directory)
            rrf.to_csv(os.path.join(directory, 'rectify_{}_{}.csv'.format(field, cfbd.BatteryPowerDispatcher.format_dispatch_columns(pred_date))))
        return 
    
    def read_and_concat_preds_in_directory(pred_dir, nb_weeks_before, first_day_pred, field):
        demand_preds=[]
        bpd = cfbd.BatteryPowerDispatcher
        for i in range(5):
            date_pred = first_day_pred + datetime.timedelta(days=-(nb_weeks_before-i)*7)
            demand_preds.append(pd.read_csv(os.path.join(pred_dir,'rectify_{}_{}.csv'.format(field, bpd.format_dispatch_columns(date_pred)))
                                           ,parse_dates=['datetime'],index_col=['datetime']))
        demand_preds = pd.concat(demand_preds)
        demand_preds = demand_preds.rename(columns={field: 'pred_{}'.format(field)})
        return demand_preds
    
    def apply_ml_on_predictions_weeks_before(df, preds_dir, first_day_pred, endog_exog_df, field, nb_weeks_before=4, 
                                          ml_model_for_demand = RandomForestRegressor(random_state=2019, n_estimators = 450)):
        demand_preds = Forecaster.read_and_concat_preds_in_directory(preds_dir, nb_weeks_before, first_day_pred, field)
        data_weeks_before = pd.merge(demand_preds, endog_exog_df, how='inner', right_index=True, left_index=True)
        scaler_pred = StandardScaler()
        data_weeks_before_scaled = data_weeks_before.copy()
        data_weeks_before_scaled = scaler_pred.fit_transform(data_weeks_before_scaled)
        X_train = data_weeks_before_scaled[:-336,:]
        y_train = df.loc[(df.index.date >= first_day_pred+datetime.timedelta(days=-7*nb_weeks_before)) & 
                            (df.index.date < first_day_pred), field].values
        model_demand = ml_model_for_demand
        model_demand.fit(X_train, y_train)
        demand_pred = model_demand.predict(data_weeks_before_scaled[-336:,:])
        return demand_pred

    
class MLPredictor:
    @staticmethod
    def preprocess_data_for_solar_predictions(train_df, predicted_df, first_day_pred, nb_days_for_train_model=400):
        weather_colums = cfbd.DataPreprocesser.get_columns_of_group_names(cfbd.DataPreprocesser,['solar', 'temp'], [1,2,3,4,5,6], df=train_df)
        weather_colums.append('sp')
        weather_colums.append('POA')
        weather_colums.append('GHI')
        df_smooth = predicted_df[weather_colums].copy()
        df_smooth = pd.concat([train_df[train_df.index.date >= first_day_pred+datetime.timedelta(days=-nb_days_for_train_model)], df_smooth])
        df_smooth = DataPreprocesser.smooth_preprocess_signal(df_smooth, 'irradiance_Wm-2')
        df_smooth = DataPreprocesser.ajust_phaze_shift_of_weather_columns(df_smooth, 'smooth_mean_irradiance_Wm-2', 'solar_location1', 
                                                 cfbd.DataPreprocesser.get_columns_of_group_names(cfbd.DataPreprocesser,
                                                                                                  ['solar', 'temp'], np.arange(1,7).tolist(),df=train_df))
        df_smooth = DataPreprocesser.smooth_preprocess_signal(df_smooth, 'pv_power_mw')
        df_smooth = DataPreprocesser.get_square_irradiance(df_smooth, cfbd.DataPreprocesser.get_columns_of_group_names(cfbd.DataPreprocesser,
                                                                                                                       ['solar'], [1,2,3,5,6], df=train_df))
        df_smooth = DataPreprocesser.get_solar_temp_product_columns(df_smooth, [1,2,3])
        return df_smooth
    @staticmethod
    def predict_pv_power(df, predicted_df, first_day_pred, field_direct = 'smooth_mean_pv_power_mw', columns_for_pv_power_pred=None, 
                         model_direct = RandomForestRegressor(random_state=2019, n_estimators=400), nb_days_for_train_model=400):
        if columns_for_pv_power_pred is None:
            columns_pred_power = cfbd.DataPreprocesser.get_columns_of_group_names(cfbd.DataPreprocesser,
                                                                                  ['solar'], [1,2,3,5,6], df=df)
            for i in [1,2,3]:
                columns_pred_power.append('solar_temp_location{}'.format(str(i)))
            for i in [1,2,3,5,6]:
                columns_pred_power.append('square_solar_location{}'.format(str(i)))
            columns_pred_power.append('sp')
            columns_pred_power.append('POA')
            columns_pred_power.append('GHI')
        else:
            columns_pred_power = columns_for_pv_power_pred
        final_field = 'pv_power_mw'
        df_smooth = MLPredictor.preprocess_data_for_solar_predictions(df, predicted_df, first_day_pred, nb_days_for_train_model=nb_days_for_train_model)
        power_features = DataPreprocesser.scale_features(df_smooth[columns_pred_power])
        X_train = power_features[power_features.index.date < first_day_pred].values
        X_test = power_features[power_features.index.date >= first_day_pred].values
        y_train = df_smooth.loc[df_smooth.index.date < first_day_pred, [field_direct, final_field]].values
        y_train_smooth = y_train[:,0]
        y_train_final = y_train[:,1]
        model_direct.fit(X_train, y_train_smooth)
        y_train_pred = model_direct.predict(X_train)
        residuals_train = y_train_final-y_train_pred
        y_test_pred = model_direct.predict(X_test)
        all_y_pred = np.concatenate([y_train_pred, y_test_pred])
        scaler_pred = StandardScaler()
        all_y_pred_scaled = scaler_pred.fit_transform(all_y_pred.reshape(all_y_pred.shape[0],1))
        y_train_pred_scaled = all_y_pred_scaled[:-(y_test_pred.shape[0])]
    
        param_grid = {'n_neighbors': np.arange(3,100).tolist()}
        model_rectify = KNeighborsRegressor()
        search = GridSearchCV(model_rectify, param_grid, cv=5)
        search.fit(np.append(X_train,y_train_pred_scaled, axis=1), residuals_train)
        best_model_rectify = search.best_estimator_
        model_rectify.fit(np.append(X_train,y_train_pred_scaled, axis=1), residuals_train)
        y_test_pred_scaled = all_y_pred_scaled[-(y_test_pred.shape[0]):]
        y_test_pred_rectify = y_test_pred + model_rectify.predict(
            np.append(X_test,y_test_pred_scaled, axis=1))
        return y_test_pred_rectify, y_test_pred
    
    

