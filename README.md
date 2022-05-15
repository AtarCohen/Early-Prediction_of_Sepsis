# Early_Prediction_of_Sepsis

This repo contains several methods for early prediction of sepsis. <br> 
To run the code first use conda to install the environment.yml file

## Prediction

<li> To predict with our best model (XGB), run predict.py. </li>
<li> To predict with the best RNN model, run predict_LSTM.py </li>

## Training

<li> To train XGB classifier, Random forest or logistic regression, use Feature_Selection_Not_Serial.
  Use the model argument to choose model, the mode argument to determine whether to run regular training or feature selection.

    parser.add_argument('--model', choices=['RF','XGB','LR'], default='XGB', type=str)
    parser.add_argument('--mode', choices=['selector','trainer'], default='trainer', type=str)
</li>
<li> To train RNN model run LSTM_main.py. Use the file args to choose layer types and other model hyper parameters.  </li>
 
## Exploration
For the feature exploration we used 2 notebooks:
<li> NonTemporalExploration </li>
<li> TemporalDataExploration </li>

## Explainability

The trees feature importance graphs can be found in Explain_Models.ipynb