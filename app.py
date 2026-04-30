from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

# Load and train model at startup
iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

class_names = iris.target_names


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(sl: float, sw: float, pl: float, pw: float):
    sample = [[sl, sw, pl, pw]]
    pred = int(model.predict(sample)[0])

    return {
        "prediction": pred,
        "class_name": class_names[pred]
    }
