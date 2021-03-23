#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
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
import sys
import os
import datetime
from pvlib import location
from pvlib import irradiance
from pvlib import clearsky, atmosphere, solarposition
import predictions_algorithmes as pa

class DataPreprocesser:
    weather_path=None
    demand_path=None
    solar_path=None
    df=None
    def __init__(self, weather_path, demand_path, solar_path):
        self.weather_path = weather_path
        self.demand_path = demand_path
        self.solar_path = solar_path
    def load_df(self):
        weather_df = pd.read_csv(self.weather_path,parse_dates=['datetime'],index_col=['datetime'])
        solar_df = pd.read_csv(self.solar_path,parse_dates=['datetime'],index_col=['datetime'])
        demand_df = pd.read_csv(self.demand_path,parse_dates=['datetime'],index_col=['datetime'])
        df=pd.merge(demand_df,solar_df , how='outer', left_index=True, right_index=True)
        df=pd.merge(df,weather_df, how='outer', left_index=True, right_index=True)
        df['week']=pd.Int64Index(df.index.isocalendar().week)
        df['dow']=df.index.dayofweek
        df['date'] = df.index.date
        df['hour'] = df.index.hour
        df['sp'] = df.hour*2 +df.index.minute/30 + 1
        self.df = df
    def set_df(self,df):
        self.df = df
    def remove_nan(self):
        self.df = self.df.dropna(subset = ['demand_MW']).interpolate()
    def interpolate_df(self):
        self.df = self.df.interpolate()
    def get_zenith_angle(self, df=None):
        lat = -4.034
        long = 50.33
        if df is None:
            self.df['zenith_angle'] = solarposition.get_solarposition(self.df.index, lat, long)['apparent_zenith'].values
            return self.df
        else:
            df['zenith_angle'] = solarposition.get_solarposition(df.index, lat, long)['apparent_zenith'].values
            return df
    def get_poa_and_ghi_irradiance(self, df=None):
    
        # For this example, we will be using Golden, Colorado
        tz = 'UTC'
        lat = 50.33
        lon = -4.034
        # lat, lon = 39.755, -105.221

        # Create location object to store lat, lon, timezone
        site = location.Location(lat, lon, tz=tz)


        # Calculate clear-sky GHI and transpose to plane of array
        # Define a function so that we can re-use the sequence of operations with
        # different locations
        def get_irradiance(site_location, date, tilt, surface_azimuth, periods):
            # Creates one day's worth of 10 min intervals
            times = pd.date_range(date, freq='30min', periods=periods,
                                  tz=site_location.tz)
            # Generate clearsky data using the Ineichen model, which is the default
            # The get_clearsky method returns a dataframe with values for GHI, DNI,
            # and DHI
        #     print(times)
            clearsky = site_location.get_clearsky(times,model='ineichen')
            # Get solar azimuth and zenith to pass to the transposition function
            solar_position = site_location.get_solarposition(times=times)
            # Use the get_total_irradiance function to transpose the GHI to POA
            POA_irradiance = irradiance.get_total_irradiance(
                surface_tilt=tilt,
                surface_azimuth=surface_azimuth,
                dni=clearsky['dni'],
                ghi=clearsky['ghi'],
                dhi=clearsky['dhi'],
                solar_zenith=solar_position['apparent_zenith'],
                solar_azimuth=solar_position['azimuth'])

            # Return DataFrame with only GHI and POA
            return pd.DataFrame({'GHI': clearsky['ghi'],
                                 'POA': POA_irradiance['poa_global']})
        if df is None:
            
            df_solar_irr = get_irradiance(site, '{}-{}-{}'.format(self.df.index.date[0].year, self.df.index.date[0].month, 
                                                                  self.df.index.date[0].day), 90, 180, len(self.df))
    #         new_df = df.copy()
            self.df['GHI'] = df_solar_irr['GHI'].values
            self.df['POA'] = df_solar_irr['POA'].values

            return self.df
        else:
            df_solar_irr = get_irradiance(site, '{}-{}-{}'.format(df.index.date[0].year, df.index.date[0].month, 
                                                                  df.index.date[0].day), 90, 180, len(df))
    #         new_df = df.copy()
            df['GHI'] = df_solar_irr['GHI'].values
            df['POA'] = df_solar_irr['POA'].values
            return df

    def build_input_for_ml_algo(self, X_column_names, y_column_names):
        if self.df is None:
            print('df is None, you have to load it')
            return self.df
        X = self.df[X_column_names].to_numpy()
        y = self.df[y_column_names].to_numpy()[:,0]
        return X, y
    def get_columns_of_group_names(self,group_names, location_numbers, df=None):
        columns_names = []
        for name in group_names:
            for i in location_numbers:
                column_name = '{}_location{}'.format(name, i)
                if df is None:
                    this_df = self.df
                else:
                    this_df = df
                if column_name in this_df.columns.to_list():
                    columns_names.append(column_name)
                else:
                    print('{} is not in df columns'.format(column_name))
        return columns_names
    

class BatteryPowerDispatcher:
    def __init__(self):
        return 
    def format_dispatch_columns(date):
        year = date.year
        month = date.month
        day = date.day
        return "{}{}{}".format(str(year).zfill(4), str(month).zfill(2),str(day).zfill(2))
    def get_ideal_discharge_dispatch(df,date, battery_charge=6):#afternoon discharge
    
        sl  = df.loc[(df.index.date == date)&(df['sp']>=32)&(df['sp']<=42),['demand_MW','pv_power_mw','hour']]
        peak_ini = sl['demand_MW'].max()

        peak_target  = peak_ini-2.5
        discharge = (sl['demand_MW']-peak_target).clip(lower = 0)

        energy =discharge.sum()*0.5
        sp = len(discharge[discharge>0])
    
        while (energy>battery_charge):

            peak_target  =  peak_target +0.01
            discharge = (sl['demand_MW']-peak_target).clip(lower=0)
            energy =discharge.sum()*0.5
        
        return(discharge,peak_ini,peak_target)
    
    def get_ideal_discharge_dispatch_in_a_week(self,df, start_date, max_battery_charge_in_week=[6,6,6,6,6,6,6]):
        res = pd.DataFrame(columns = ['peak_ini','peak_target','energy','duration','week','dow'])
        dispatch_summary = pd.DataFrame(index= range(32,43))
        idx = 0
        for i in range(0,7):
            date = start_date + datetime.timedelta(days=i)
            week = date.isocalendar()[1]
            dow = date.weekday()
            discharge,peak_ini,peak_target = self.get_ideal_discharge_dispatch(df,date, max_battery_charge_in_week[i])
            energy = discharge.sum()*0.5
            duration = len(discharge[discharge>0])
            dispatch_summary[self.format_dispatch_columns(date)]=discharge.values
            idx = idx+1
            res.loc[idx,:] = [peak_ini,peak_target,energy,duration,week,dow]
        return (dispatch_summary, res)
    
    def get_charge_of_battery_repartition(df, date, max_charge=6):
        solar_power  = df.loc[(df.index.date == date)&(df['sp']<=31),['pv_power_mw','sp']]
        max_power = 2.5
        solar_power['pv_power_norm'] = solar_power['pv_power_mw'].apply(lambda x: min(x, max_power))
        max_charge_from_solar = solar_power['pv_power_norm'].sum()*0.5
        charge_from_solar = min(max_charge_from_solar, max_charge)
        solar_power['pv_power_norm'] = solar_power['pv_power_norm']*charge_from_solar / max_charge_from_solar
        max_charge_from_grid = max_charge-charge_from_solar
        battery_B = pd.DataFrame(data=solar_power['pv_power_norm'].to_list(),columns=['solar_B'],index=solar_power.index)
        charge_power_from_grid = pd.DataFrame(columns=['grid_B'])
        charge_from_grid = 0
        solar_power_ordered = solar_power.copy()
        solar_power_ordered['ind'] = solar_power_ordered['sp'].values
        solar_power_ordered = solar_power_ordered.sort_values('pv_power_mw', ascending=False)
        for i in range(len(solar_power_ordered)):
            idx = solar_power_ordered['ind'].values[i]
            power_from_solar = solar_power_ordered['pv_power_norm'].values[i]
            charge_power_from_grid.loc[idx,:] = min(max(max_power - power_from_solar,0), max(max_charge_from_grid - charge_from_grid,0)*2)
            charge_from_grid = charge_power_from_grid['grid_B'].sum()*0.5
        charge_power_from_grid.sort_index()
        battery_B['sp'] = solar_power['sp'].to_list()
        battery_B['grid_B'] = charge_power_from_grid['grid_B'].to_list()
        battery_B['B'] = battery_B['solar_B'] + battery_B['grid_B']
        return (charge_from_solar, charge_from_grid, battery_B)
    
    def get_charge_of_battery_repartition3(df, date, max_charge=6):
        solar_power  = df.loc[(df.index.date == date)&(df['sp']<=31),['pv_power_mw','sp', 'GHI', 'POA']]
        max_power = 2.5
        solar_power['pv_power_norm'] = solar_power['pv_power_mw'].apply(lambda x: min(max(x,0), max_power))
        max_charge_from_solar = solar_power['pv_power_norm'].sum()*0.5
        charge_from_solar = min(max_charge_from_solar, max_charge)
        solar_power['pv_power_norm'] = solar_power['pv_power_norm']*charge_from_solar / max_charge_from_solar
        max_charge_from_grid = max_charge-charge_from_solar
        max_GHI = solar_power['GHI'].max()
        solar_power['diff_GHI'] = (solar_power['GHI']*2.5/max_GHI-solar_power['pv_power_norm']).apply(lambda x: max(x,0))
        max_diff = solar_power['diff_GHI'].sum()*0.5

        battery_B = pd.DataFrame(data=solar_power['pv_power_norm'].to_list(),columns=['solar_B'],index=solar_power.index)
        battery_B['grid_B'] = solar_power['diff_GHI'].values*max_charge_from_grid/max_diff

        battery_B['sp'] = solar_power['sp'].to_list()
        battery_B['B'] = battery_B['solar_B'] + battery_B['grid_B']
        return (charge_from_solar, max_charge_from_grid, battery_B)
    
    def get_charge_of_battery_repartition2(df, date, max_charge=6):
        solar_power = df.loc[(df.index.date == date)&(df['sp']<=31),['pv_power_mw','sp']]
        solar_power['ind'] = solar_power['sp'].apply(lambda x: int(x))
        max_power = 2.5
        uncertainty = 0.8
        ratio = solar_power['pv_power_mw'].apply(lambda x: min(x,max_power)).sum()*0.5
        if ratio > 1.5:
            coeff = uncertainty
        elif ratio > 1.2:
            coeff = 0.9
        else:
            coeff = 1
        solar_power = solar_power.sort_values('pv_power_mw', ascending=False)
        max_solar_charge = min(max_charge,solar_power['pv_power_mw'].apply(lambda x: min(x,2,5)).sum()*0.5)
        max_grid_charge = max_charge - max_solar_charge
        solar_charge = 0
        grid_charge = 0
        battery_B = pd.DataFrame(columns = ['ind', 'sp', 'solar_B', 'grid_B'])
        for i in range (len(solar_power)):
            solar_B = min(coeff*min(solar_power['pv_power_mw'][i], max_power), max(0,max_solar_charge - solar_charge)*2)
            grid_B = min(max_power-solar_B, max(0,(max_grid_charge-grid_charge)*2))
            battery_B.loc[i, :] = [solar_power['ind'][i], solar_power['sp'][i], solar_B, grid_B]
            solar_charge += solar_B*0.5
            grid_charge += grid_B*0.5
        battery_B['B'] = battery_B['solar_B'] + battery_B['grid_B']
        battery_B =battery_B.sort_values('ind', ascending = True)
        battery_B.index = df.loc[(df.index.date == date)&(df['sp']<=31),:].index
        battery_B = battery_B.drop('ind', axis=1)
        return (solar_charge, grid_charge, battery_B)
    
    def get_solar_energy_proportion_by_day_in_a_week(self,df,start_date, charge_method=3,max_battery_charge_in_week=[6,6,6,6,6,6,6]):
        B = pd.DataFrame(index= range(1,32))
        p_solar = []
        for i in range(7):
            date = start_date + datetime.timedelta(days=i)
            if charge_method == 1:
                charge_from_solar, charge_from_grid, battery_B = self.get_charge_of_battery_repartition(
                    df, date, max_battery_charge_in_week[i])
            elif charge_method == 2:
                charge_from_solar, charge_from_grid, battery_B = self.get_charge_of_battery_repartition2(
                    df,date, max_battery_charge_in_week[i])
            elif charge_method == 3:
                charge_from_solar, charge_from_grid, battery_B = self.get_charge_of_battery_repartition3(
                    df,date, max_battery_charge_in_week[i])
            else:
                print("ERROR : charge method in (1,2)")
                return None, None
            B[self.format_dispatch_columns(date)] = battery_B['B'].to_list()
            p_solar.append(charge_from_solar/6)
        return p_solar, B
    def get_max_solar_energy_available(df, date):
        solar_power  = df.loc[(df.index.date == date)&(df['sp']<=31),['pv_power_mw','sp']]
        return min(6,solar_power['pv_power_mw'].apply(lambda x: min(2.5, x)).sum()*0.5)
    def get_max_solar_energy_available_in_a_week(self,df, start_date):
        max_solar_energy_available = []
        for i in range(7):
            date = start_date + datetime.timedelta(days=i)
            max_solar_energy_available.append(self.get_max_solar_energy_available(df, date))
        return max_solar_energy_available
    def get_end_of_the_day_dispatch(self,start_date):
        B_end_of_the_day = pd.DataFrame(index= range(43,49))
        for i in range(0,7):
            date = start_date + datetime.timedelta(days=i)
            B_end_of_the_day[self.format_dispatch_columns(date)]=0
        return B_end_of_the_day
    def get_all_dispatch_in_a_week(self,df, start_date, charge_method=3,full_solar=False):
        if full_solar:
            max_battery_charge_in_week = self.get_max_solar_energy_available_in_a_week(self, df, start_date)
        else:
            max_battery_charge_in_week = [6,6,6,6,6,6,6]
        p_solar, B_charge = self.get_solar_energy_proportion_by_day_in_a_week(self,df, start_date, charge_method,max_battery_charge_in_week)
        B_discharge, res = self.get_ideal_discharge_dispatch_in_a_week(self, df, start_date, max_battery_charge_in_week)
        max_battery_charge_in_week = (B_discharge.sum()/2).values.tolist()
        p_solar, B_charge = self.get_solar_energy_proportion_by_day_in_a_week(self,df, start_date, charge_method,max_battery_charge_in_week)
        B_discharge = -B_discharge
        B_end_of_the_day = self.get_end_of_the_day_dispatch(self,start_date)
        B_total = B_charge.append(B_discharge)
        B_total = B_total.append(B_end_of_the_day)
        return B_total
    def format_dispatching_for_competition(B, index):
        final_B = []
        for column in B.columns.to_list():
            final_B.append(B[column])
        final_B = pd.concat(final_B, ignore_index=True)
        final_B.index = index
        final_B = pd.DataFrame(final_B, columns=['charge_MW'])
        return final_B
    
class MachineLearningResearcher:
    param_grids = {}
    models = {}
    X = None
    y = None
    scores = {}
    best_score = {}
    def __init__(self, X, y):
        self.param_grids['SVR'] = {'kernel': ['rbf'], 'C': [0.2, 0.4, 0.6, 0.8, 1.0], 'gamma': ['scale', 'auto'], 'epsilon': [0.1,0.05,0.2]}
        self.param_grids['KNeighborsRegressor'] = {'n_neighbors': np.arange(3,100).tolist()}
        self.param_grids['GradientBoostingRegressor'] = {'loss': ['ls'], 'n_estimators': [50,100, 150, 200], 'learning_rate': [0.025,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.5]}
        self.param_grids['Lasso, Ridge, ElasticNet'] = {'alpha': np.round(np.arange(0.1,1.1, 0.1), decimals=1).tolist()}
        self.param_grids['RandomForestRegressor'] = {'n_estimators': np.arange(50, 500, 50).tolist()}
        self.models['SVR'] = [SVR()]
        self.models['KNeighborsRegressor'] = [KNeighborsRegressor()]
        self.models['GradientBoostingRegressor'] = [GradientBoostingRegressor()]
        self.models['Lasso, Ridge, ElasticNet'] = [Lasso(), Ridge(), ElasticNet()]
        self.models['RandomForestRegressor'] = [RandomForestRegressor()]
        self.X = X
        self.y = y
        
    def compute_best_scores_and_params_of_ml_algos(self,X, y, models, param_grid):
        search_results = {}
        search_results['best_model'] = None
        search_results['best_scores'] = 0
        search_results['best_parameters'] = None
        for model in models:
            search = GridSearchCV(model, param_grid, cv=5)
            search.fit(X, y)
            if (search.best_score_ > search_results['best_scores']):
                search_results['best_model'] = model
                search_results['best_scores'] = search.best_score_
                search_results['best_parameters'] = search.best_params_
        return search_results
    def display_ml_algo_scores(self, scores=None):
        fig, ax = plt.subplots(figsize=(14,14))
        if scores is None:
            algo_scores = self.scores
        else:
            algo_scores = scores
        for model_name in algo_scores:
            ax.scatter([model_name], [algo_scores[model_name]['best_scores']], label=model_name)
        ax.legend()
        plt.show()
        return
    def get_best_scores_and_params_of_ml_algos(self, models_names, param_grid=None):
        if param_grid is None:
            self.scores[models_names] = self.compute_best_scores_and_params_of_ml_algos(self.X, self.y, self.models[models_names], self.param_grids[models_names])
        else:
            self.scores[models_names] = self.compute_best_scores_and_params_of_ml_algos(self.X, self.y, self.models[models_names], param_grid) 
        return self.scores[models_names]
    def get_best_model_with_best_score(self, scores=None):
        def get_best_score(scores):
            best_score = {}
            best_score['best_scores'] = 0
            for score_name in scores:
                if float(scores[score_name]['best_scores']) > float(best_score['best_scores']):
                    best_score = scores[score_name]
            return best_score
        if scores is None:
            self.best_score = get_best_score(self.scores)
        else:
            self.best_score = get_best_score(scores)
        return self.best_score
    
class DataVisualiser:
    def __init__():
        return
    def display_correlation_color_map(df, columns_names):
        f, ax = plt.subplots(figsize=(20,20))
        sns.heatmap(df[columns_names].corr(), annot=True, square=True, cmap = 'coolwarm')
        plt.show()
    def pair_plot(df, columns_names, hue_name):
        if len(columns_names) != 3:
            print('columns_names should have exactly 3 names, that\'s not the case here : ', columns_names)
        elif not(isinstance(hue_name, str)):
            print('hue_name should be a string, that\'s not the case here : ', hue_name)
        else:
            sns.pairplot(data=df[columns_names], hue=hue_name)


class MLPredictor:
    def __init__(self, data_preprocess, start_date_pred):
        self.data_preprocess = data_preprocess
        self.start_date_pred = start_date_pred
        self.predicted_df = data_preprocess.df.loc[(data_preprocess.df.index.date >= start_date_pred+datetime.timedelta(days=-7)) &
                                                   (data_preprocess.df.index.date < start_date_pred), 
                                                   ['week', 'dow', 'sp', 'hour', 'zenith_angle']]
        self.predicted_df.index = self.predicted_df.index + pd.DateOffset(7)
        self.predicted_df['week']=pd.Int64Index(self.predicted_df.index.isocalendar().week)
        self.predicted_df['dow']=self.predicted_df.index.dayofweek
        self.predicted_df['hour'] = self.predicted_df.index.hour
        self.predicted_df['sp'] = self.predicted_df.hour*2 +self.predicted_df.index.minute/30 + 1
        DataPreprocesser.get_zenith_angle(DataPreprocesser, self.predicted_df)
        DataPreprocesser.get_poa_and_ghi_irradiance(DataPreprocesser, self.predicted_df)
        
    def get_field_previous_week(self, field_name):
        field_prediction = self.data_preprocess.df.loc[(self.data_preprocess.df.index.date >= self.start_date_pred+datetime.timedelta(days=-7)) &
                                                   (self.data_preprocess.df.index.date < self.start_date_pred), field_name].values
        if self.predicted_df is not None:
            self.predicted_df[field_name] = field_prediction 
            return self.predicted_df
        return field_prediction
    def get_demand_previous_week(self):
        return self.get_field_previous_week('demand_MW')
    def get_solar_power_previous_week(self):
        return self.get_field_previous_week('pv_power_mw')
    def get_weather_prediction(self,weather_path, pred_df=None):
        if pred_df is None:
            predicted_df = self.predicted_df
        else:
            predicted_df = demand_pred
        weather_prediction = pd.read_csv(weather_path,parse_dates=['datetime'],index_col=['datetime'])
        predicted_df = pd.merge(predicted_df,weather_prediction, how='outer', left_index=True, right_index=True)
        predicted_df = predicted_df.dropna(subset = ['demand_MW']).interpolate()
        self.predicted_df = predicted_df
        return self.predicted_df
    def predict_demand_from_past_and_weather(self, model, nb_week_before=4, start_date_prediction=None, data=None, 
                                             pred_df=None, weather_cols=None):
        if data is None:
            data_train = self.data_preprocess.df
        else:
            data_train = data
        if pred_df is None:
            predicted_df = self.predicted_df
        else:
            predicted_df = pred_df
        if weather_cols is None:
            weather_columns = self.data_preprocess.get_columns_of_group_names(['temp'], [1,2,5,6])
            weather_columns.append('dow')
            weather_columns.append('sp')
        else:
            weather_columns = weather_cols
        if start_date_prediction is None:
            start_date_pred = self.start_date_pred
        else:
            start_date_pred = start_date_prediction
        data_train = data_train[(data_train.index.date >= (start_date_pred+datetime.timedelta(days=-nb_week_before*7))) & 
                                (data_train.index.date < start_date_pred)]
        X = data_train[weather_columns].to_numpy()
        y = data_train['demand_MW'].to_numpy()
        model.fit(X,y)
        predicted_df['demand_MW'] = model.predict(predicted_df[weather_columns].values)
        self.predicted_df = predicted_df
        return predicted_df
        
    def pred_demand_with_forecast_method_and_average_with_previous_weeks(self,df, forecast_dir, first_day_pred, field, 
                                                                         predicted_df=None,compute_forecast=False,
                                                                        forecast_model=RandomForestRegressor(random_state=2019, n_estimators=450),
                                                                        nb_weeks_for_train=5, lags_list=[1,2,3,4,42,47,48,48*7],
                                                                        columns_pred = None):
        if predicted_df is None:
            pred_df = self.predicted_df
        else:
            pred_df = predicted_df
        if columns_pred is None:
            cols_pred = DataPreprocesser.get_columns_of_group_names(DataPreprocesser,['temp'], [1,2,5,6], df=df)
            cols_pred += DataPreprocesser.get_columns_of_group_names(DataPreprocesser,['solar'], [1,4],df=df)
            cols_pred.append('sp')
            cols_pred.append('dow')
        else:
            cols_pred = columns_pred
        endog_exog_df = pred_df[cols_pred].copy()
        endog_exog_df = pd.concat([df[cols_pred], endog_exog_df])
        f = pa.Forecaster()
        if compute_forecast:
            f.recursive_rectify_weeks_before(df, field,first_day_pred, endog_exog_df, lags_list, forecast_dir, 
                                     RandomForestRegressor(random_state=2019, n_estimators=450),nb_days_for_train=nb_weeks_for_train*7)
        rec_preds = pa.Forecaster.apply_ml_on_predictions_weeks_before(df, os.path.join(forecast_dir, field), first_day_pred, endog_exog_df, field)
        lags_demand_3_weeks_before = pa.DataPreprocesser.get_lags_and_median_and_mean_predict_and_train_df(df, field,first_day_pred)
        
        pred_df[field] = (rec_preds + lags_demand_3_weeks_before.loc[(lags_demand_3_weeks_before.index.date >= first_day_pred)
                                                                                             & (lags_demand_3_weeks_before.index.date < first_day_pred + 
                                                                                            datetime.timedelta(days=7)), 'median_{}'.format(field)].values)/2
        return pred_df
    def predict_pv_power_smooth_and_square_irr(self, predicted_df=None, df=None, first_day_pred=None, nb_weeks_for_train_model=5):
        if predicted_df is None:
            pred_df = self.predicted_df
        else:
            pred_df = predicted_df
        if df is None:
            this_df = self.data_preprocess.df
        else:
            this_df = df
        if first_day_pred is None:
            start_date_pred = self.start_date_pred
        else:
            start_date_pred = first_day_pred
        mlp = pa.MLPredictor
        pv_power_pred_rectify, pv_power_pred = mlp.predict_pv_power(this_df, pred_df, start_date_pred, nb_days_for_train_model=7*nb_weeks_for_train_model)
        pred_df['pv_power_mw'] = pv_power_pred_rectify
        if predicted_df is None:
#             self.predicted_df['pv_power_mw'] = pv_power_pred_rectify
            return
        return this_df, pv_power_pred_rectify, pv_power_pred
    def predict_solar_power_from_weather(self, model, data=None, pred_df=None, weather_cols=None):
        def solar_power_prediction_function(x, model, x_solar):
            if x_solar == 0:
                return 0
            else:
                return model.predict(x)[0]
        if data is None:
            data_train = self.data_preprocess.df
        else:
            data_train = data
        if pred_df is None:
            predicted_df = self.predicted_df
        else:
            predicted_df = pred_df
        if weather_cols is None:
            weather_columns = self.data_preprocess.get_columns_of_group_names(['solar'], [1,2,3,5,6])
            weather_columns += self.data_preprocess.get_columns_of_group_names(['temp'], [1,2])
            weather_columns.append('sp')
        else:
            weather_columns = weather_cols
        X = data_train.loc[data_train['solar_location1'] > 0, weather_columns].values
        y = data_train.loc[data_train['solar_location1'] > 0, 'pv_power_mw'].values
        model.fit(X,y)
        
        predicted_df['pv_power_mw'] = predicted_df.apply(lambda x: solar_power_prediction_function(
            np.array([x[weather_columns].to_numpy()]), model, x['solar_location1']), axis=1)
        self.predicted_df = predicted_df
        return predicted_df
    def predict_solar_power_weeks_before(self, model, nb_week_before=5, start_date_prediction=None, 
                                         data=None, pred_df=None, weather_cols=None):
        if data is None:
            data_train = self.data_preprocess.df
        else:
            data_train = data
        if pred_df is None:
            predicted_df = self.predicted_df
        else:
            predicted_df = pred_df
        if start_date_prediction is None:
            start_date_pred = self.start_date_pred
        else:
            start_date_pred = start_date_prediction
        data_train = data_train[(data_train.index.date >= (start_date_pred+datetime.timedelta(days=-nb_week_before*7))) & 
                                (data_train.index.date < start_date_pred)]
        return self.predict_solar_power_from_weather(model, data=data_train, weather_cols=weather_cols)
class ScoreComputer:
    def __init__(self, B_path):
        B = pd.read_csv(B_path, parse_dates=['datetime'],index_col=['datetime'])
        B['week']=pd.Int64Index(B.index.isocalendar().week)
        B['dow']=B.index.dayofweek
        B['hour'] = B.index.hour
        B['sp'] = B.hour*2 + B.index.minute/30 + 1
        self.B = B
        
    def compute_r_peak(self,demand_and_solar_power,date, B_pred=None):
        if B_pred is None:
            B = self.B
        else:
            B = B_pred
        demand_discharge = demand_and_solar_power.loc[(demand_and_solar_power.index.date == date)
                                                      &(demand_and_solar_power['sp']>=32)&
                                          (demand_and_solar_power['sp']<=42),'demand_MW']
        B_discharge = B.loc[(B.index.date == date) &(B['sp']>=32) & (B['sp']<=42),'charge_MW']
        old_peak = demand_discharge.max()
        if old_peak == 0:
            return 0
        new_peak = (B_discharge + demand_discharge).max()
        return 100*(old_peak-new_peak)/old_peak
    def compute_p_solar(self,demand_and_solar_power, date, B_pred=None):
        if B_pred is None:
            B = self.B
        else:
            B = B_pred
        solar_power_for_charge = demand_and_solar_power.loc[(demand_and_solar_power.index.date == date) &
                                                            (demand_and_solar_power['sp']<=31),['pv_power_mw']]
        B_charge = B.loc[(B.index.date == date) & (B['sp']<=31),['charge_MW']]
        solar_power_for_charge = solar_power_for_charge.merge(B_charge, how='outer', left_index=True, right_index=True)
        battery_charge = B_charge['charge_MW'].sum()*0.5
        solar_power_for_charge['charge_from_solar_MW'] = solar_power_for_charge.apply(lambda x: min(x['pv_power_mw'], x['charge_MW']), axis=1)
        battery_charge_from_solar = solar_power_for_charge['charge_from_solar_MW'].sum()*0.5
        if battery_charge == 0:
            p_solar = 0
        else:
            p_solar = battery_charge_from_solar / battery_charge
        p_grid = 1-p_solar
        return p_solar, p_grid, battery_charge, solar_power_for_charge
    def compute_scores(self, demand_and_solar_power, start_date, B_pred=None):
        if B_pred is None:
            B = self.B
        else:
            B = B_pred
        scores = pd.DataFrame(columns = ['r_peak', 'p_solar', 's'])
        idx = 0
        with tqdm(total=7, file=sys.stdout) as pbar:
            for i in range(7):
                date = start_date + datetime.timedelta(days=i)
                r_peak = self.compute_r_peak(demand_and_solar_power, date, B)
                p_solar, p_grid, battery_charge, solar_power_for_charge = self.compute_p_solar(demand_and_solar_power, date, B)
                s = r_peak*(3*p_solar + p_grid)
                scores.loc[idx, :] = [r_peak, p_solar, s]
                idx += 1
                pbar.update()
        scores.index = [BatteryPowerDispatcher.format_dispatch_columns(start_date+datetime.timedelta(days=i)) for i in range(7)]
        self.scores = scores
        self.scores_mean = scores.mean()
        return scores, scores.mean()

class MultiScoresComparator:
    def __init__(self,dp, pred_first_days):
        self.dp = dp
        self.pred_first_days = pred_first_days
    def write_B_on_several_weeks_with_one_method(self,B_dir, data_preprocess= None, predict_first_days=None,
                                                pred_demand=False, pred_pv=False,
                                                weather_cols_demand=None, weather_cols_pv=None,
                                                nb_weeks_before_demand=4, nb_weeks_before_pv= 5,
                                                model_demand=RandomForestRegressor(random_state=2019, n_estimators=450),
                                                model_pv=RandomForestRegressor(random_state=2019, n_estimators=300)):
        if data_preprocess is None:
            dp = self.dp
        else:
            dp=data_preprocess
        if predict_first_days is None:
            pred_first_days = self.pred_first_days
        else:
            pred_first_days = predict_first_days
        with tqdm(total=len(pred_first_days), file=sys.stdout) as pbar:
            for pred_first_day in pred_first_days:
                mp=MLPredictor(dp, pred_first_day)
                mp.get_demand_previous_week()
                if (pred_demand or pred_pv):
                    mp.get_weather_prediction(dp.weather_path)
                if pred_demand:
                    mp.predict_demand_from_past_and_weather(model_demand, nb_week_before=nb_weeks_before_demand, 
                                                            weather_cols=weather_cols_demand)
                if pred_pv:
                    mp.predict_solar_power_weeks_before(model_pv, nb_week_before=nb_weeks_before_pv, 
                                                        weather_cols=weather_cols_pv)
                else:
                    mp.get_solar_power_previous_week()
                bdp = BatteryPowerDispatcher
                B_total = bdp.get_all_dispatch_in_a_week(bdp,mp.predicted_df, pred_first_day)
                B = bdp.format_dispatching_for_competition(B_total, mp.predicted_df.index)
                B.to_csv('{}first_day_{}.csv'.format(B_dir, BatteryPowerDispatcher.format_dispatch_columns(pred_first_day)))
                pbar.update()
        
    def get_scores_on_several_weeks(self,B_dir,predict_first_days=None,data_preprocesser=None):
        if data_preprocesser is None:
            dp = self.dp
        else:
            dp=data_preprocesser
        if predict_first_days is None:
            pred_first_days = self.pred_first_days
        else:
            pred_first_days = predict_first_days
        scores = []
        scores_mean = []
        with tqdm(total=len(pred_weeks), file=sys.stdout) as pbar:
            for pred_first_day in pred_first_days:
                sc = ScoreComputer('{}first_day_{}.csv'.format(B_dir, BatteryPowerDispatcher.format_dispatch_columns(pred_first_day)))
                score, score_mean = sc.compute_scores(dp.df, pred_first_day)
                scores.append(score)
                scores_mean.append(score_mean)
                pbar.update()
        return scores, scores_mean
    
    def compare_scores(self, scores, predict_first_days=None):
        if predict_first_days is None:
            pred_first_days = self.pred_first_days
        else:
            pred_first_days = predict_first_days
        score_cols = ['week', 'dow']
        for name in list(scores.keys()):
            score_cols.append('{}_r_peak'.format(name))
        for name in list(scores.keys()):
            score_cols.append('{}_p_solar'.format(name))
        for name in list(scores.keys()):
            score_cols.append('{}_s'.format(name))
        score_comps = []
        for i in range(len(pred_first_days)):
            score_comp = pd.DataFrame(index=scores[list(scores.keys())[0]][0].index)
            score_comp['date_in_number'] = score_comp.index
            score_comp['first_day_pred'] = pred_first_days[i]
            for name in list(scores.keys()):
                scores[name][i] = scores[name][i].rename(columns={'r_peak': '{}_r_peak'.format(name), 'p_solar': '{}_p_solar'.format(name), 
                                                  's': '{}_s'.format(name)})
                score_comp = pd.merge(score_comp,scores[name][i] , how='outer', left_index=True, right_index=True)
            score_comps.append(score_comp)
        score_comps = pd.concat(score_comps)
        score_comps = score_comps[score_cols]
        score_comps.index = range(score_comps.shape[0])
        return score_comps


# In[ ]:




