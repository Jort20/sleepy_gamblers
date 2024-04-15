"""
Dit classifier model is door Jort Gommers en Giel Bakker gevormd. In dit project zijn er twee .ipynb bestanden gevormd.
In de comp.ipynb en machine_log_delete.ipynb zijn dezelfde stappen genomen in het vinden van het machine learning model met de
hoogste accuracy. Het enige verschil is dat de vraagtekens bij comp.ipynb zijn vervangen door de gemiddelde waarde van deze feature,
en dat de machine_log_delete.ipynb de instances verwijderd wanneer er een vraagteken is gevonden.
Alhoewel er in de machine_log_delete.ipynb met de LinearDiscriminant analysis een hogere test accuracy is behaald, is deze niet werkend
gekregen in de class. Dit komt doordat de test data voor de prognose10jaar altijd vraagetekens bevat. Hierdoor moet deze features worden
verwijderd, alleen zorgt dit er ook voor dat de PCA niet goed verloopt. Zonder de PCA wordt er een lagere accuracy behaald, waardoor
ervoor gekozen is om het model te vormen door bij de missende waarde het gemiddelde in te vullen.
Er wordt een RandomForestClassifier model gebruikt in de predict class.
"""

from kerstman import kerstboom

def model_factory():
    return kerstboom(true_or_false=True)