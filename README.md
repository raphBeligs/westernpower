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

Columns are added to the data frame: hour, half hour, day of the week, week of the year

The calculation of the azimuthal angle and then the GHI and POA irradiances are done using the pvlib library and the latitude, longitude and date and time data

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

### discharge

For the discharge, we know the maximum charge and we suppose the demand prediction optimal.


## Forecast/prediction of demand

### First model

The first method consists in training a RandomForestRegressor model on the last 5 weeks before the prediction date. Assuming that the demand profile does not vary greatly over these 5 weeks and that the temperature, schedule and day of the week data would be sufficient to estimate the variation in demand from one week to another.
Columns used: temp location 1,2,5,6 , dow and sp

The RandomForestRegressor model was tested on all the dataset and presented the best results with a CVSearch (other models were tested like GradientBoostingRegressor, LinearRegression, KNeightbourghRegressor, SVM rtf, ElasticNetRegressor)

This method seemed to be better than the Benchmark but after task1 and task2 where our score was just below the Benchmarck, we decided to use a forecasting

### forecast model (recursive with rectify)

First of all, to identify the lags that influence the signal the most, we applied a pacf on our signal with a max delay of 336 (1 week), we kept the lags with the highest pacf, i.e. lags 1,2,3,4,42,47,48,336

Then by applying a cross-validation for time series, we have once again retained the RandomForestRegression model (nb_estimator = 400) which gave an r2= 0.983

The RamdomForestRegression was then used as a recursive ML model, with for endogenous and exogenous columns the temperatures location 1,2,5,6, sp, dow and the lags seen previously.

Thanks to the work of the thesis https://robjhyndman.com/papers/rectify.pdf we have implemented a RECTIFY strategy which consists for each forecast to correct the residual error between the prediction of the direct model and the true value. This correction is done with a KNN model trained on the lags and the endogenous and exogenous data

### Last model

Task 3 we were slightly better than the Benchmark but still quite far from the best results.

We observed that our forecast followed the demand pattern well but tended to smooth out the peaks, a factor that caused us to lose many points on the r_peak score. We also noticed that the peaks were quite similar from one week to another, so we adjusted the forecast with the median of the demand of the last 3 weeks.

Finally, adding the solar location 4 and 1 components improved the forecast 

## Prediction of pv power

### first model

Following cross validation, the model we retained is the RandomForestRegressor (n_estimator =300), which presented an r2 = 0.85

Using the following columns: sp, temp location 1,2,3,5,6 and solar location 1,2,5,6,4 the zenith angle

The model is trained over the last 4 weeks

### last model

#### Preprocessing

We have smoothed the pv power, this smoothing leading to a dephasing, a rephasing function (correlation with the initial signal) is used.This rephasing function is also used for the solar and temp location (shift phaze calculated between pv_irradiance_Mw-2 and solar_location1)

We added the columns of solar location squared and the products of solar location by temp location. These calculations are used to take into account the temperature at the solar panel which depends on the irradiance squared, the ambient temperature times the irradiance and the ambient temperature

#### model

We use the solar location 1,2,4,5,6 columns and their square, as well as the product of solar and temp for locations 1,2,3. With also sp, GHI and POA 

The machine learning model is always the RandomForestRegressor (n_estimator=300) on 4 previous weeks

Finally, we use the smoothed pv power as data to predict

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
