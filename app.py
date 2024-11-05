import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Sample data for training the model (replace with your actual data)
data = {
    'Date': pd.date_range(start='2021-01-01', periods=36, freq='D'),
    'Beans Sold': [100, 150, 200, 120, 180, 160, 220, 90, 250, 140, 130, 170, 110, 210, 190, 230, 250, 160, 140, 180,
                   200, 120, 170, 190, 140, 130, 150, 110, 180, 210, 220, 230, 180, 200, 160, 150]
}
df = pd.DataFrame(data)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create the 'Date_Num' feature
df['Date_Num'] = (df['Date'] - df['Date'].min()).dt.days

# Define a list of holidays
holidays = pd.to_datetime(['2021-01-01', '2021-12-25', '2021-07-04'])  # Add relevant holidays here

# Create a 'Holiday' feature
df['Holiday'] = df['Date'].isin(holidays).astype(int)  # 1 for holiday, 0 for non-holiday

# Prepare features and target variable
X = df[['Date_Num', 'Holiday']]
y = df['Beans Sold']

# Train the model
model = LinearRegression()
model.fit(X, y)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        date_str = request.form['date']
        date = pd.to_datetime(date_str)
        date_num = (date - df['Date'].min()).days
        is_holiday = int(date in holidays)

        # Make prediction
        prediction = model.predict(np.array([[date_num, is_holiday]]))[0]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
