# Used Car Price Prediction

A machine learning project that predicts the selling price of used cars using a Random Forest regression model.

## Dataset
CarDekho used car dataset from Kaggle: [https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data?resource=download]
The file `cardekho_data.csv` contains about 15,000 used car listings with features such as vehicle age, kilometers driven, engine size, max power, mileage, seats, fuel type, transmission type and seller type.

## Steps
1. Loaded and explored the data (`info`, `describe`, missing values, duplicates)
2. Plotted the price distribution (right-skewed, with a few very expensive cars as outliers)
3. Dropped the index column and name columns (`car_name`, `brand`, `model`), which have many unique values and do not help the model generalize
4. Removed duplicate rows
5. One-hot encoded the categorical columns
6. Split the data 80/20 into training and test sets
7. Trained a Random Forest Regressor
8. Evaluated it with MAE and R², compared train vs test scores, and checked feature importance
9. Predicted the price of a sample car

## Results
| Metric | Value |
|---|---|
| Test R² | about 0.79 |
| MAE | about ₹1.14 lakh |
| Median price in data | about ₹5.59 lakh |
| Train R² | about 0.99 |

- The average error is roughly 20% of the median car price.
- Train R² is much higher than test R², so the model overfits to some extent.
- Most important features: `max_power` (about 69%), `vehicle_age` (about 14%) and `km_driven` (about 8%).

## Example Prediction
For a petrol, manual car sold by an individual, 5 years old, with 40,000 km driven, a 1200 cc engine and 120 hp, the model predicts a price of roughly ₹7.2 lakh. Reducing max power to 80 hp lowers the prediction to roughly ₹4.4 lakh, as expected.

## A Data Issue I Found and Fixed
My first run gave R² of 0.92. The CSV's row-index column (`Unnamed: 0`) made every row look unique, so duplicate cars were not removed and could appear in both the training and test sets. After dropping that column and removing the duplicates, the honest test R² was about 0.79.

## Future Improvements
- Reduce overfitting with hyperparameter tuning and cross-validation (a first attempt at limiting tree depth lowered the test score)
- Predict the log of the price to handle the skewed distribution
- Try other models such as Gradient Boosting
- Save the trained model and build a small web app for predictions

## How to Run
1. Clone or download this repository
2. `pip install pandas matplotlib seaborn scikit-learn`
3. Open `Price.ipynb` in Jupyter Notebook, VS Code or Google Colab
4. Keep `cardekho_data.csv` in the same folder and run all cells

## Tools
Python, Pandas, Matplotlib, Seaborn, Scikit-learn
