import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

# custom method for generating predictions
def getPredictions(model, scalermodel, sepal_length, sepal_width, petal_length, petal_width):
    prediction = model.predict(scalermodel.transform([[sepal_length, sepal_width, petal_length, petal_width]]))
    return prediction


# Main function
if __name__ == "__main__":
    # Prompt the user for inputs
    print("Enter the following inputs for prediction:")
    sepal_length = float(input("Sepal Length: "))
    sepal_width = float(input("Sepal Width: "))
    petal_length = float(input("Petal Length: "))
    petal_width = float(input("Petal Width: "))

    model = pickle.load(open("model.sav", "rb"))
    scalermodel = pickle.load(open("scalermodel.sav", "rb"))

    result = getPredictions(model, scalermodel, sepal_length, sepal_width, petal_length, petal_width)

    # Print the result
    print(f"Prediction: {result}")

    
