import joblib

model = joblib.load("diabetes_79.pkl")

result = model.predict([[1,1,1,1,1,1,1,1]])

if result[0]==1:
    print('d')
else:
    print('nd')