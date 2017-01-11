# AXA_challenge


### Models

- The training data should be in a file named  `train_2011_2012_2013.csv`, and the submission data should be in a file 
named `submission.txt`. Both should be in the directory `data/`
- `mean_predictor.py` makes predictions based on the mean of the values for the same assignment, time slot and day of the week. 
This was our first model and used information from the future, contrarily to all the other models that looked only at the past.
- `usual_methods.py` uses date and holiday features and ensemble methods to make the predictions. 
- `usual_methods_parallel.py` applies the same model as `usual_methods.py`, but is implemented in parallel.
- `time_series.py` makes prediction using a SARIMAX model, however we have not succeeded in building a proper procedure to select the parameters.
- `dummy_prediction.py` makes predictions using the number of calls using past weeks

### Visualisation

- `figures/all_received_calls` contains visualisations of the received calls for each assignment
- `figures/seasonal_decompositions` contains visualisations of seasonal ARIMA decompositions
- `viz_daily` contains visualisations of the received calls with a daily frequency

Our best submission file can be obtained by running `usual_methods.py`, and will be stored in `data/test_submission_overestimated.csv`