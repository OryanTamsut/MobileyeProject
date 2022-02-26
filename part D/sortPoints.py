import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model


class SortPoints:

    def __init__(self):
        self.__neural_network = load_model("model.h5")

    def run(self, pictures):
        predict = self.__neural_network.predict(pictures)
        prediction = [1 if predict[i] >= 0.2 else 0 for i in range(len(pictures))]
        return prediction
