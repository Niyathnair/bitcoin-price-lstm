# Bitcoin Price Regression (Keras)

A notebook that fits a Keras neural network to one year of minute-level BTC/USD data and predicts the `high` price of each one-minute bar from its timestamp and traded volume. Note on naming: the saved model file is named after an LSTM, and `LSTM` is imported in the notebook, but the network that is built, trained and saved is a fully connected (Dense) network with no recurrent layers.

## Approach

1. Load `BTC-2017min.csv` (525,599 rows; columns `unix`, `date`, `symbol`, `open`, `high`, `low`, `close`, `Volume BTC`, `Volume USD`). The file has no missing values.
2. Exploratory plots: histogram of `close`, correlation heatmap, and a pairplot of `open`, `high`, `low`, `close`, `Volume USD`.
3. Parse `date` and derive `day`, `month`, `year` and `time` (the hour of day as an integer); drop `date`.
4. Features: `unix`, `Volume BTC`, `Volume USD`, `day`, `month`, `year`, `time` (7 columns). The other price columns and `symbol` are dropped. Target: `high`.
5. Scale the features to [0, 1] with `MinMaxScaler`, fitted on the full dataset before splitting.
6. Split 80/20 with `train_test_split(test_size=0.20, random_state=42)`. The split is random, not chronological.
7. Model: `Sequential` with five `Dense(84, activation="relu")` layers and a `Dense(1)` output (29,317 parameters), compiled with the Adam optimizer and MSE loss.
8. Train with batch size 128, using the test split as validation data. The code cell sets `epochs=25`; the saved output is from a 50-epoch run.
9. Evaluate on the test split with R2 and RMSE, then save the model to `lstm_model.h5`. The evaluation outputs saved in the notebook are from an earlier session, so no figures are quoted here; re-run the notebook to reproduce them.

## Repository contents

- `Bitcoin_Project.ipynb` - data loading, exploratory plots, feature engineering, model training, evaluation, and model export.
- `Load_Model_prediction.ipynb` - placeholder; contains a single empty code cell.
- `lstm_model.h5` - the saved Keras model (the Dense network described above, input shape 7).
- `README.md` - this file.

## Running it

```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

- Download `BTC-2017min.csv` from the Kaggle dataset below and place it in the repository root, next to the notebook; it is read with a relative path.
- Open with `jupyter notebook Bitcoin_Project.ipynb` and run the cells in order.
- The saved run used a Python 3.7 kernel. On recent pandas, `data.corr()` in the heatmap cell needs `numeric_only=True` because the frame still contains text columns at that point.

## Data

Bitcoin Historical Dataset on Kaggle: https://www.kaggle.com/datasets/prasoonkottarathil/btcinusd

Only the 2017 one-minute file (`BTC-2017min.csv`) is used. The data is not included in this repository.
