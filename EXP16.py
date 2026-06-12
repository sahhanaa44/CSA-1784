from sklearn.neural_network import MLPClassifier

X = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

y = [0,1,1,0]

model = MLPClassifier(hidden_layer_sizes=(4,),
                      max_iter=1000)

model.fit(X, y)

prediction = model.predict([[1,0]])

print("Prediction =", prediction[0])
