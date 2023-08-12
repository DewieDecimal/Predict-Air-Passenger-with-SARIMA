# Predict-Air-Passenger-with-SARIMA

## Project Overview
This project focuses on utilizing the SARIMA (Seasonal AutoRegressive Integrated Moving Average) package to predict the number of air passengers. The analysis is conducted on the "Air Passenger Data for Time Series Analysis" dataset from Kaggle. The data is a list of passenger data from the year 1949 to 1960. The primary aim is to develop a predictive model that effectively captures air passenger data's underlying patterns, trends, and seasonality.

## Problem Statement
Air passenger traffic is a critical factor in the aviation industry, impacting various aspects such as flight scheduling, resource allocation, and revenue forecasting. Accurate prediction of air passenger numbers is essential for airlines, airports, and related stakeholders to make informed decisions. By developing a robust prediction model, I am aiming to address the challenge of anticipating future passenger demand, allowing for better planning and resource utilization.

## Why it Matters
Efficiently predicting air passenger numbers can yield several benefits, including:
1. Optimized Resource Allocation: Airlines and airports can adjust their operations, staffing, and logistics based on predicted passenger numbers, avoiding overstaffing or insufficient resources.
2. Enhanced Customer Experience: Accurate predictions enable airlines to provide better services, such as timely flights, adequate seating, and sufficient in-flight amenities.
3. Revenue Management: Accurate forecasts can lead to improved revenue management by optimizing ticket pricing and promotional strategies, as Ill as overselling seats to hedge the risks of cancellations and no-shows.
4. Infrastructure Planning: Airports can plan expansions, renovations, and infrastructure improvements based on projected passenger growth.


## Dependencies
To run this project, you'll need the following Python packages:
* numpy: A library for numerical computations in Python.
* pandas: A data manipulation and analysis library.
* matplotlib: A plotting library for creating visualizations.
* pmdarima: A library for Auto ARIMA modeling.
* statsmodels: A library for time series analysis and modelings like ARIMA and SARIMAX
* joblib: A library for lightIight pipelining in Python such as dumping and loading models


## Key Steps
### Exploratory Data Analysis:
#### Cleaned data

| Statistics metric | Information    |
| ------------------| -------------- |
| Count             | 144.00         |
| Mean              | 280.30         |
| Standard Deviation| 119.97         |
| Minimum           | 104.00         |
| 25th Percentile   | 180.00         |
| Median (50th %ile)| 265.50         |
| 75th Percentile   | 360.50         |
| Maximum           | 622.00         |

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/5d491c38-4bfe-4b28-b966-0c6abc938841)

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/220bd68a-af76-4c98-876b-93f8c66d44aa)

The dataset displayed no abnormal patterns, and no outliers or missing values are observed. The dataset is cleaned.

#### Normal trends

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/8b7cfeba-3907-44d2-a397-5bf3e52b8128)

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/b509e3f8-954b-44cb-b932-415d86cc7240)

I noticed that the data follows a general pattern that can usually be observed in years that are not influenced by extreme economic conditions, travel restrictions, etc.
* Big spikes in the summer (starting from June to August) due to summer vacations and a high demand for leisure travel:
    * June marks the beginning of the vacation season
    * July is the peak travel month
    * The end of August is when the vacation season starts to end for many people
* There are also small spikes:
    * March is when students and families have trips due to students having Spring break
    * December is the holiday season with Christmas at the end of the month and New Year's right after

#### Time-series analysis
![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/42d4af59-30ab-4c8e-920e-b1a9fc71dc38)

A few things that I can draw from the decomposer:
* There is a noticeable upward trend
* There is seasonality, which will make SARIMA a better choice than ARIMA
* The data is unlikely to be stationary
* Residual plot doesn't show any trend or seasonality, so the decomposer is working fine

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/89a5d421-59e4-4a32-b171-0081e1b9b4f0)

The first seasonal difference is still seasonal, plus, there still appears to be a trend in our differenced time-series. An additional non-seasoning differencing is needed.

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/ad566ccc-cef9-4e10-8d6b-62b3a389d02f)
![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/60ba71a5-79e2-434b-90e6-e399e4cf0850)

Finally achieved stationarity after one seasonal differencing and one non-seasonal differencing

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/4c7b1a4a-6332-45c3-a581-7aea8391fbb5)
![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/399bc55c-f736-4485-95f0-e1ddb780f1b4)

Generally, to maintain a balanced configuration, the sum of the parameters should not exceed 10 (p + d + q ≤ 10). 
Upon a preliminary examination of the ACF and PACF plots, I tentatively inferred that p,P,q, and Q should fall into a range of 0 to 1.  
It's safe to say the best model for our data is something like this:
ARIMA([0-1] ,1 ,[0-1]) x ([0-1], 1, [0-1]) 12

Building upon this rationale, I will now proceed with parameter refinement using Auto ARIMA - Auto ARIMA will try every possible combination of the parameters to choose the best combination with the lowest AIC score as a low AIC score indicates a good fit.


### Build Model
#### Iterating and Evaluating model
![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/938f0804-6189-4c9f-b3bf-c5ecc23517cb)

All variables are significant.

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/8b8bee11-66d6-4622-9328-6af0c5f795f5)

The "free" model identified is ARIMA(1,1,0)(1,1,3)[12]. It's important to highlight that among the 6 variables, 2 exhibit p-values exceeding 0.05. This suggests that these variables might not possess a substantial statistically significant influence on the model's outcome.

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/e7fb94c1-22f4-4d04-b2ce-6aaddc9a210a)

As expected, the "free" model has a higher accuracy (lower AIC indicates better fit), but it is also more complex (higher BIC indicates the higher complexity).

I can do further evaluation but generally, it's not worth the time, since the difference in accuracy is low. The decision of whether or not which would be the better model is more than just a simple trade-off between the model's accuracy and simplicity. Other factors such as domain expertise and context should be taken into account as well.

I proceeded with the simple model as it aligns with my ACF and PACF analysis.

![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/e8a62029-58b5-484c-b3aa-ecef551a8297)

The residuals plots look good as there is no trend or patterns.


![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/9d9d0439-5468-45ae-8db7-2b30d53aae87)

Generally, for time series, rolling validation is the best method of cross-validation. Employing this approach, I derived an MSE (Mean Squared Error) score which demonstrates a reduction of 27-fold compared to the baseline MSE.


![image](https://github.com/DewieDecimal/Predict-Air-Passenger-with-SARIMA/assets/125356334/6c573e1c-1311-4650-98b2-74b5f100fb5e)

The visualized outcome appears to meet our satisfaction; however, there remains potential for enhancement. Subsequent refinements and adjustments can be considered in future tuning efforts.



## Next Steps:
Expanding our analysis to include other methods can provide a comprehensive understanding of the data and potentially lead to improved predictions. Hence, there are some possible next steps for me:
* Test other models/packages such as Meta's Prophet, LSTMs, CNNs, GPVAR, etc.
* Predict volatility using GARCH model
* Engineer new features that will potentially help regressor models like XGBoost account for seasonality, etc.
* Create an ensemble model that combines predictions from multiple models


## Conclusion
Accurate prediction of air passenger numbers is vital for the aviation industry's efficient operation and planning. By utilizing the SARIMA model and the "Air Passenger Data for Time Series Analysis" dataset, this project addresses the challenge of forecasting passenger demand. The insights gained from this predictive model can facilitate better decision-making, leading to improved resource allocation, enhanced customer experiences, and overall revenue management.
