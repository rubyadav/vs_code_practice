import pickle

model = pickle.load(open("diabetes_79.pkl",'rb'))

result = model.predict([[1,1,1,1,1,1,1,1]])

if result[0]==1:
    print('d')
else:
    print('nd')