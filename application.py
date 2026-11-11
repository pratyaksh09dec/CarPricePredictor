from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open("car_price_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

df = pd.read_csv("Cleaned_Data.csv")

companies = sorted(df['company'].unique())
years = sorted(df['year'].unique(), reverse=True)
fuel_types = sorted(df['fuel_type'].unique())

company_model_dict = {}

for company in companies:
    models = sorted(df[df['company'] == company]['name'].unique())
    company_model_dict[company] = models
companies.insert(0,'Select Company')

def predict_price(name, company, year, kms_driven, fuel_type):

    data = pd.DataFrame([[name,company,year,kms_driven,fuel_type]],
        columns=['name','company','year','kms_driven','fuel_type'])

    data = pd.get_dummies(data)

    data = data.reindex(columns=model_columns,fill_value=0)

    prediction = model.predict(data)

    return round(prediction[0], 2)


@app.route('/')
def home():
    return render_template(
        'index.html',
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        company_model_dict=company_model_dict,
        prediction=None,

        selected_company=None,
        selected_model=None,
        selected_year=None,
        selected_fuel=None,
        selected_kms=None
    )


@app.route('/predict', methods=['POST'])
def predict():

    company = request.form['company']
    model_name = request.form['model']
    year = int(request.form['year'])
    fuel_type = request.form['fuel_type']
    kms_driven = int(request.form['kms_driven'])

    predicted_price = predict_price(
        model_name,
        company,
        year,
        kms_driven,
        fuel_type
    )

    return render_template(
        'index.html',
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        company_model_dict=company_model_dict,

        prediction=f"₹ {predicted_price:,.2f}",

        selected_company=company,
        selected_model=model_name,
        selected_year=year,
        selected_fuel=fuel_type,
        selected_kms=kms_driven
    )


if __name__ == '__main__':
    app.run(debug=True)