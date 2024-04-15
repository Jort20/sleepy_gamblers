import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pkg_resources
import pickle
import numpy as np


class Kerstboom:
    """
    Deze klasse bevat voor de Kerstboom classifier een predict functie.
    Deze klasse wordt dus gebruikt voor het voorspellen van de prognose10jaar
    van een dataset over een bepaalde hartziekte.
    """
    def __init__(self, true_or_false=False):
        """
        Deze functie dient als opvulling voor de class.
        Deze heeft geen verdere doel voor het machine learning model.
        :param true_of_false: Deze waarde kan True of False zijn en veranderd
        hiermee de self naar True of False.
        """
        self.true_or_false = true_or_false

    def predict(self,  filename='data/competition_test.csv'):
        """
        Deze functie ontvangt een test dataset voor de prognose10jaar
        van een hartziekte, en doet met een eerder geselecteerd machine learning
        model een voorspelling aan de hand van deze dataset.
        Voor de voorspelling, wordt de data eerst omgezet naar een bruikbare
        formaat voor het machine learning model. Ook wordt er een PCA uitgevoerd
        op de gegeven test data.
        :param filename: Dit is de path naar het bestand met erin de test data,
        waarop een voorspelling moet worden gedaan. Wanneer er geen filename wordt
        meegegeven, wordt de competition_test.csv file gebruikt.
        :output prediction: Er wordt een lijst gevormd met erin de voorspellingen
        van het machine learning model. 
        """
        data = pd.read_csv(filename)
        data['prognose10jaar'] = data['prognose10jaar'].astype('category')
        # Preprocessing moet hierna komen
        columns_to_replace = ['hypertensie', 'hartinfarct', 'diabetes', 'nierziekte']

        for column in columns_to_replace:
            data[column] = data[column].replace({'-': 0, '+': 1})
        data = data.drop('Individu-ID', axis=1)

        data = data.drop('prognose10jaar', axis=1)
        data['geslacht'] = data['geslacht'].replace({'M': 1, 'V': 0})

        # Maak een LabelEncoder object
        label_encoder = LabelEncoder()

        # Pas labelencodering toe op 'opleidingsniveau'
        data['opleidingsniveau'] = label_encoder.fit_transform(data['opleidingsniveau'])

        # Vervang '?' door NaN in de hele dataset
        data.replace('?', np.nan, inplace=True)

        # Converteer de gehele dataset naar numerieke waarden
        data = data.apply(pd.to_numeric)

        # Bereken de gemiddelde waarde van elke kolom
        column_means = data.mean()

        # Vervang NaN-waarden door de gemiddelde waarden per kolom
        data.fillna(column_means, inplace=True)

        scaled = StandardScaler().fit_transform(data.select_dtypes(include='number'))

        # Fit PCA on the scaled data
        pca = PCA().fit(scaled)

        # Transform the scaled data using PCA
        components = pca.transform(scaled)

        # Iterate over the first 4 principal components
        for n in range(4):
            # Create attribute names for the principal components
            attribute = f"PC-{n+1}"

            # Add new columns to the 'stroke' DataFrame with the principal component values
            data[attribute] = components[:, n]

        # y = data.iloc[:, :-1]
        X = data.to_numpy()
        modelfile = pkg_resources.resource_filename(__name__, 'kerstboom.pkl')

        with open(modelfile, 'rb') as filehandle:
            model = pickle.load(filehandle)
        yhat = model.predict(X)
        prediction = []
        for number in yhat:
            if number == 0:
                prediction += ['CHD-']
            else:
                prediction += ['CHD+']
        return prediction

if __name__ in '__main__':
    model = Kerstboom(true_or_false=False)
    print(model.predict())
   