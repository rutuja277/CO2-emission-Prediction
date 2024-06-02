import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

# Create Flask app
app = Flask(__name__)

# Load model
model = joblib.load("random_forest_regressor.pkl")

df_freq = {
    'FORD': 628,
    'CHEVROLET': 587,
    'BMW': 527,
    'MERCEDES-BENZ': 419,
    'PORSCHE': 376,
    'TOYOTA': 330,
    'GMC': 328,
    'AUDI': 286,
    'NISSAN': 259,
    'JEEP': 251,
    'DODGE': 246,
    'KIA': 231,
    'HONDA': 214,
    'HYUNDAI': 210,
    'MINI': 204,
    'VOLKSWAGEN': 197,
    'MAZDA': 180,
    'LEXUS': 178,
    'JAGUAR': 160,
    'CADILLAC': 158,
    'SUBARU': 140,
    'VOLVO': 124,
    'INFINITI': 108,
    'BUICK': 103,
    'RAM': 97,
    'LINCOLN': 96,
    'MITSUBISHI': 95,
    'CHRYSLER': 88,
    'LAND ROVER': 85,
    'FIAT': 73,
    'ACURA': 72,
    'MASERATI': 61,
    'ROLLS-ROYCE': 50,
    'ASTON MARTIN': 47,
    'BENTLEY': 46,
    'LAMBORGHINI': 41,
    'ALFA ROMEO': 30,
    'GENESIS': 25,
    'SCION': 22,
    'SMART': 7,
    'BUGATTI': 3,
    'SRT': 2
}
Model_freq={
    'F-150 FFV 4X4': 32,
    'F-150 FFV': 32,
    'MUSTANG': 27,
    'FOCUS FFV': 24,
    'SONIC': 20,
    'SONIC 5': 20,
    'HIGHLANDER': 7,
    'GT-R': 7,
    'MDX SH-AWD': 6,
    'MKT AWD': 6,
    '500X AWD': 5,
    'SIENNA AWD': 5,
    'ESCALADE 4WD': 5,
    'Charger FFV': 4,
    'CRUZE ECO': 4,
    'GENESIS AWD': 4,
    'Charger': 4,
    'CX-5 4WD (Cylinder Deactivation)': 3,
    'FOCUS ST': 3,
    '328i xDRIVE GRAN TURISMO': 3,
    'SHELBY GT350 MUSTANG': 3,
    'Tucson': 2,
    'F-PACE 20d': 2,
    'Optima Hybrid': 2,
    'FLYING SPUR': 9,
    'CIVIC SEDAN': 9,
    'WRX AWD': 9,
    'Sierra': 9
}
vehicle_class_freq = {
    "SUV - SMALL": 1217,
    "MID-SIZE": 1132,
    "COMPACT": 1022,
    "SUV - STANDARD": 735,
    "FULL-SIZE": 639,
    "SUBCOMPACT": 606,
    "PICKUP TRUCK - STANDARD": 538,
    "TWO-SEATER": 460,
    "MINICOMPACT": 326,
    "STATION WAGON - SMALL": 252,
    "PICKUP TRUCK - SMALL": 159,
    "MINIVAN": 80,
    "SPECIAL PURPOSE VEHICLE": 77,
    "VAN - PASSENGER": 66,
    "STATION WAGON - MID-SIZE": 53,
    "VAN - CARGO": 22
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get features from the form
    make = request.form['make']
    model_name = request.form['model']
    vehicle_class = request.form['vehicle_class']
    engine_size = float(request.form['engine_size'])
    cylinders = int(request.form['cylinders'])
    fuel_city = float(request.form['fuel_city'])
    fuel_hwy = float(request.form['fuel_hwy'])
    fuel_comb = float(request.form['fuel_comb'])
    fuel_mpg = float(request.form['fuel_mpg'])
  
    # Fuel type mapping
    fuel_type = request.form['fuel_type']
    fuel_ethanol = 1 if fuel_type == 'ethanol' else 0
    fuel_premium = 1 if fuel_type == 'premium' else 0
    fuel_regular = 1 if fuel_type == 'regular' else 0

    # Transmission type mapping
    transmission = request.form['transmission']
    am = 1 if transmission == 'AM' else 0
    as_value = 1 if transmission == 'AS' else 0
    cvt = 1 if transmission == 'CVT' else 0
    m = 1 if transmission == 'M' else 0

    # Prepare features for prediction
    features = np.array([[df_freq.get(make, 0), Model_freq.get(model_name, 0), vehicle_class_freq.get(vehicle_class, 0),
                        engine_size, cylinders, fuel_city, fuel_hwy, fuel_comb, fuel_mpg,
                        fuel_ethanol, fuel_premium, fuel_regular,
                        am, as_value, cvt, m]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Render prediction
    return render_template("index.html", prediction_text="Predicted value:  {:.3f} grams per kilometer".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)





# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import joblib

# # Create Flask app
# app = Flask(__name__)

# # Load model
# model = joblib.load("random_forest_regressor.pkl")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     # Get features from the form
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
    
#     # Make prediction without scaling
#     prediction = model.predict(features)
    
#     # Render prediction
#     return render_template("index.html", prediction_text="Predicted value: {}".format(prediction[0]))

# if __name__ == "__main__":
#     app.run(debug=True)
