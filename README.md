# westernpower

datascience competition  in energy

## Install environnemnt

in westernpower root directory write conda command :

conda env create -f environment.yml

This will create a conda environmnent with all the package needed

Then to activate environment write conda command :

conda activate westernpower

To launch jupyter notebook, write in conda command :

jupyter notebook

## Description of the data structure

The main codes (forecast, data preprocessor, dispatch of battery charge) are in the common directory.

There are common_function_by_dates.py and predictions_algorithms.py, the last one is used for forecast and prediction of demand and power

Each task has its own directory with Input and Output and jupyter scripts. In each of these directories, there is at least one script to predict the load and discharge of the battery and another one to calculate the model score, the maximum score (with the real values of demand and power) and the Benchmark score

## Data preprocessor

The 3 inputs csv (demand, pv solar and weather) are (inner) joined and the nan values are corrected by a linear interpolation.

```python 
weather_df = pd.read_csv(self.weather_path,parse_dates=['datetime'],index_col=['datetime'])
solar_df = pd.read_csv(self.solar_path,parse_dates=['datetime'],index_col=['datetime'])
demand_df = pd.read_csv(self.demand_path,parse_dates=['datetime'],index_col=['datetime'])
df=pd.merge(demand_df,solar_df , how='outer', left_index=True, right_index=True)
df=pd.merge(df,weather_df, how='outer', left_index=True, right_index=True)
self.df = self.df.dropna(subset = ['demand_MW']).interpolate()
```

Columns are added to the data frame: hour, half hour, day of the week, week of the year

```python 
df['week']=pd.Int64Index(df.index.isocalendar().week)
df['dow']=df.index.dayofweek
df['date'] = df.index.date
df['hour'] = df.index.hour
df['sp'] = df.hour*2 +df.index.minute/30 + 1
```

The calculation of the azimuthal angle and then the GHI and POA irradiances are done using the pvlib library and the latitude, longitude and date and time data

```python 
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
```

## Dipatch of battery charge

Discharging and charging are done separately

The charge dispatching is done first, from which the maximum charge of the battery is deduced. Then the discharge dispatch is calculated. If the total discharge is slightly lower than the maximum charge of the battery, the load dispatch is calculated again by imposing a lower maximum charge.

### Charge

We tested 3 methods, they all start by recovering the power production pv between midnight and 4 pm, they set a threshold at 2.5 MW for each point and adjust proportionally to this power production the load of the battery

If the power production pv is greater than 6MWh between 0 am and 4 pm then the 3 methods give the same result, otherwise :

- Method 1, calculates the remaining load and distributes it at the beginning of the day (at midnight), which would have the advantage of using the grid's production during an off-peak period, but this method is less advantageous for the solar score

- Method 2, calculates the remaining load and distributes it in priority on the half hour where the production of power pv was the strongest, it would thus support the hours where the production of pv seems the highest

- Method 3, calculates the equivalent GHI for the day, scales it to the battery load, i.e. the maximum power is reduced to 2.5 MW, while keeping the GHI curve. Then it calculates the difference between the GHI equivalent curve and the solar production at any point, and it distributes the rest of the load proportionally to this difference

We have kept the method 3 which decreases the errors of predictions of the power pv


```python 
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
    
```

### discharge

For the discharge, we know the maximum charge and we suppose the demand prediction optimal.


```python 
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
    
```

## Forecast/prediction of demand

### First model

The first method consists in training a RandomForestRegressor model on the last 5 weeks before the prediction date. Assuming that the demand profile does not vary greatly over these 5 weeks and that the temperature, schedule and day of the week data would be sufficient to estimate the variation in demand from one week to another.
Columns used: temp location 1,2,5,6 , dow and sp

The RandomForestRegressor model was tested on all the dataset and presented the best results with a CVSearch (other models were tested like GradientBoostingRegressor, LinearRegression, KNeightbourghRegressor, SVM rtf, ElasticNetRegressor)

This method seemed to be better than the Benchmark but after task1 and task2 where our score was just below the Benchmarck, we decided to use a forecasting


```python 
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
    
```

### forecast model (recursive with rectify)

First of all, to identify the lags that influence the signal the most, we applied a pacf on our signal with a max delay of 336 (1 week), we kept the lags with the highest pacf, i.e. lags 1,2,3,4,42,47,48,336

```python 
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
    
```

Then by applying a cross-validation for time series, we have once again retained the RandomForestRegression model (nb_estimator = 400) which gave an r2= 0.983

The RamdomForestRegression was then used as a recursive ML model, with for endogenous and exogenous columns the temperatures location 1,2,5,6, sp, dow and the lags seen previously.

Thanks to the work of the thesis https://robjhyndman.com/papers/rectify.pdf we have implemented a RECTIFY strategy which consists for each forecast to correct the residual error between the prediction of the direct model and the true value. This correction is done with a KNN model trained on the lags and the endogenous and exogenous data

```python 
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
    
```

### Last model

Task 3 we were slightly better than the Benchmark but still quite far from the best results.

We observed that our forecast followed the demand pattern well but tended to smooth out the peaks, a factor that caused us to lose many points on the r_peak score. We also noticed that the peaks were quite similar from one week to another, so we adjusted the forecast with the median of the demand of the last 3 weeks.

Finally, adding the solar location 4 and 1 components improved the forecast 

```python 
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
                                     forecast_model,nb_days_for_train=nb_weeks_for_train*7)
        rec_preds = pa.Forecaster.apply_ml_on_predictions_weeks_before(df, os.path.join(forecast_dir, field), first_day_pred, endog_exog_df, field)
        lags_demand_3_weeks_before = pa.DataPreprocesser.get_lags_and_median_and_mean_predict_and_train_df(df, field,first_day_pred)
        
        pred_df[field] = (rec_preds + lags_demand_3_weeks_before.loc[(lags_demand_3_weeks_before.index.date >= first_day_pred)
                                                                                             & (lags_demand_3_weeks_before.index.date < first_day_pred + 
                                                                                            datetime.timedelta(days=7)), 'median_{}'.format(field)].values)/2
        return pred_df
    
```

## Prediction of pv power

### first model

Following cross validation, the model we retained is the RandomForestRegressor (n_estimator =300), which presented an r2 = 0.85

Using the following columns: sp, temp location 1,2,3,5,6 and solar location 1,2,5,6,4 the zenith angle

The model is trained over the last 4 weeks

```python 
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
    
```

### last model

#### Preprocessing

We have smoothed the pv power, this smoothing leading to a dephasing, a rephasing function (correlation with the initial signal) is used.This rephasing function is also used for the solar and temp location (shift phaze calculated between pv_irradiance_Mw-2 and solar_location1)

We added the columns of solar location squared and the products of solar location by temp location. These calculations are used to take into account the temperature at the solar panel which depends on the irradiance squared, the ambient temperature times the irradiance and the ambient temperature

```python 
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
    
```

#### model

We use the solar location 1,2,4,5,6 columns and their square, as well as the product of solar and temp for locations 1,2,3. With also sp, GHI and POA 

The machine learning model is always the RandomForestRegressor (n_estimator=300) on 4 previous weeks

Finally, we use the smoothed pv power as data to predict

```python 
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
    
```

## Others functions

In this repository there are also methods used for the calculation of scores, comparison of scores between different models, hyperparameter optimization functions, analyses on the pcf and pacf of the demand, curves (histograms, boxplot etc) on the data.

For each task there is a directory with : 
- a notebook raph_prediction_task[week_nb]_last_model.ipynb (calculation of the battery charge with the Benchmark method, with the real values and with our last model)
- a notebook raph_score_computation_task[week_nb].ipynb (calculation of the scores for each method)
- Input : the challenge data for each week
- Ouput : predictions and calculations of battery loads


## Future Improvement

- manage outliers and missing data
- Use a probabilistic ML model to quantify the uncertainty of the pv power prediction
- Use an autoregressive model with the Rectify forecast
- Try a deep learning model for the forecast
- use other data (humidity, wind etc)


Thanks to Stephen Haben and all the POD Challenge team, as well as to all the challenge teams, for this competition, we enjoyed participating
