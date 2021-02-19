#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
        df['week']=df.index.week
        df['dow']=df.index.dayofweek
        df['hour'] = df.index.hour
        df['sp'] = df.hour*2 +df.index.minute/30 + 1
        self.df = df
    def set_df(self,df):
        self.df = df
    def remove_nan(self):
        self.df = self.df.dropna(subset = ['demand_MW']).interpolate()
    def interpolate_df(self):
        self.df = self.df.interpolate()
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
    def get_ideal_discharge_dispatch(df,week,dow):#afternoon discharge
    
        sl  = df.loc[(df['week']==week)&(df['dow']==dow)&(df['sp']>=32)&(df['sp']<=42),['demand_MW','pv_power_mw','hour']]
        peak_ini = sl['demand_MW'].max()

        peak_target  = peak_ini-2.5
        discharge = (sl['demand_MW']-peak_target).clip(lower = 0)

        energy =discharge.sum()*0.5
        sp = len(discharge[discharge>0])
    
        while (energy>6):

            peak_target  =  peak_target +0.01
            discharge = (sl['demand_MW']-peak_target).clip(lower=0)
            energy =discharge.sum()*0.5
        
        return(discharge,peak_ini,peak_target)
    
    def get_ideal_discharge_dispatch_in_a_week(self,df, week):
        res = pd.DataFrame(columns = ['peak_ini','peak_target','energy','solar_energy','duration','week','dow'])
        dispatch_summary = pd.DataFrame(index= range(32,43))
        idx = 0
        for dow in range(0,7):
            discharge,peak_ini,peak_target = self.get_ideal_discharge_dispatch(df,week,dow)
            energy = discharge.sum()*0.5
            sp = len(discharge[discharge>0])
            dispatch_summary[str(week*10)+str(dow)]=discharge.values
            idx = idx+1
            solar_available = df.loc[(df['week']==week)&(df['dow']==dow)&(df['sp']<=31),'pv_power_mw'].sum()*0.5
            res.loc[idx,:] = [peak_ini,peak_target,energy,solar_available,sp,week,dow]
        return (dispatch_summary, res)
    
    def get_charge_of_battery_repartition(df, week, dow):
        solar_power  = df.loc[(df['week']==week)&(df['dow']==dow)&(df['sp']<=31),['pv_power_mw','sp']]
        max_power = 2.5
        max_charge = 6
        max_charge_from_solar = min(solar_power['pv_power_mw'].sum()*0.5, max_charge)
        max_charge_from_grid = max_charge-max_charge_from_solar
        charge_power_from_solar = pd.DataFrame(columns=['power'])
        charge_power_from_grid = pd.DataFrame(columns=['power'])
        charge_from_solar = 0
        charge_from_grid = 0
        for idx in range(31):
            power_from_solar = solar_power['pv_power_mw'].values[idx]
            charge_power_from_solar.loc[idx,:] = min(max_power, power_from_solar, max((max_charge_from_solar - charge_from_solar)*2,0))
            charge_power_from_grid.loc[idx,:] = min(max(max_power - power_from_solar,0), max(max_charge_from_grid - charge_from_grid,0)*2)
            charge_from_solar = charge_power_from_solar['power'].sum()*0.5
            charge_from_grid = charge_power_from_grid['power'].sum()*0.5
        return (charge_from_solar, charge_from_grid, charge_power_from_solar, charge_power_from_grid)
    
    def get_solar_energy_proportion_by_day_in_a_week(self,df,week):
        B = pd.DataFrame(index= range(1,32))
        p_solar = []
        for dow in range(7):
            charge_from_solar, charge_from_grid, B_solar, B_grid = self.get_charge_of_battery_repartition(df, week, dow)
            B[str(week*10)+str(dow)] = B_solar.values + B_grid.values
            p_solar.append(charge_from_solar/6)
        return p_solar, B
    def get_end_of_the_day_dispatch(week):
        B_end_of_the_day = pd.DataFrame(index= range(43,49))
        for dow in range(0,7):
            B_end_of_the_day[str(week*10)+str(dow)]=0
        return B_end_of_the_day
    def get_all_dispatch_in_a_week(self,df, week):
        p_solar, B_charge = self.get_solar_energy_proportion_by_day_in_a_week(self,df, week)
        B_discharge, res = self.get_ideal_discharge_dispatch_in_a_week(self, df, week)
        B_discharge = -B_discharge
        B_end_of_the_day = self.get_end_of_the_day_dispatch(week)
        B_total = B_charge.append(B_discharge)
        B_total = B_total.append(B_end_of_the_day)
        return B_total
    
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
    def __init__(self, data_preprocess, week_prediction):
        self.data_preprocess = data_preprocess
        self.week_prediction = week_prediction
    def get_demand_previous_week(self):
        demand_prediction = self.data_preprocess.df.loc[self.data_preprocess.df['week'] == (self.week_prediction-1), ['demand_MW', 'week', 'dow', 'hour', 'sp']]
        demand_prediction.index = demand_prediction.index + pd.DateOffset(7)
        demand_prediction['week']=demand_prediction.index.week
        demand_prediction['dow']=demand_prediction.index.dayofweek
        demand_prediction['hour'] = demand_prediction.index.hour
        demand_prediction['sp'] = demand_prediction.hour*2 +demand_prediction.index.minute/30 + 1
        self.predicted_df = demand_prediction
        return self.predicted_df
    def get_weather_prediction(self,weather_path, demand_pred=None):
        if demand_pred == None:
            demand_prediction = self.predicted_df
        else:
            demand_prediction = demand_pred
        weather_prediction = pd.read_csv(weather_path,parse_dates=['datetime'],index_col=['datetime'])
        demand_and_weather_prediction = pd.merge(demand_prediction,weather_prediction, how='outer', left_index=True, right_index=True)
        demand_and_weather_prediction = demand_and_weather_prediction.dropna(subset = ['demand_MW']).interpolate()
        self.predicted_df = demand_and_weather_prediction
        return self.predicted_df
    
    
    def predict_solar_power_from_weather(self, model, data_prep=None, pred_df=None, weather_cols=None):
        def solar_power_prediction_function(x, model, x_solar):
            if x_solar == 0:
                return 0
            else:
                return model.predict(x)[0]
        if data_prep is None:
            data_preprocess = self.data_preprocess
        else:
            data_preprocess = data_prep
        if pred_df == None:
            predicted_df = self.predicted_df
        else:
            predicted_df = pred_df
        if weather_cols == None:
            weather_columns = data_preprocess.get_columns_of_group_names(['temp', 'solar'], [1,2])
            weather_columns.append('sp')
        else:
            weather_columns = weather_cols
        X,y = data_preprocess.build_input_for_ml_algo(weather_columns, ['pv_power_mw'])
        model.fit(X,y)
    
        predicted_df['pv_power_mw'] = predicted_df.apply(lambda x: solar_power_prediction_function(np.array([x[weather_columns].to_numpy()]), model, x['solar_location1']), axis=1)
        self.predicted_df = predicted_df
        return predicted_df
        

