from tabnanny import verbose
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from pytorch_tabular import TabularModel
import pickle
import pandas as pd
import torch

import sys
import os

# Aggiungi il percorso della directory 'models'
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../TrainingModule')))
# import feedforward
from feedforward import FeedForward, MyDataset, test_model

''' 
 Codice che implementa 4 funzioni Python:
    1. getName() --> restituisce un identificativo dello studente o del gruppo.
    2. preprocess(df, clfName) --> effettua il pre-processing dei dati di test forniti in input.
    3. load(clfName) --> istanzia la tecnica di ML con nome clfName.(LR, RF, KNR, SVR, FF, TB, TF)
    4. predict(df, clfName, clf) --> esegue il modello ML (oggetto clf) sui dati di TEST preprocessati (df)
'''

MY_UNIQUE_ID = "Pair Forza"

# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


def preprocess(df, clfName):
    # Descrizione parametri in input
    # - df: Dataframe Pandas contenente i dati di TEST. I dati hanno la stessa struttura (numero 
    #       e nomi attributi) di quelli usati per il TRAINING.
    # - clfName: stringa che identifica la tecnica di ML da utilizzare, e che può assumere i 
    #            valori indicati nella slide successiva

    # Output:
    # Dataframe Pandas ottenuto come risultato del pre-processamento

    X=df.iloc[:,1:]
    y=df.iloc[:,0]
    dfNew=pd.DataFrame()

    try:
        if clfName in ['LR', 'RF', 'KNR', 'SVR', 'TB', 'TF']: # Applico solo MinMaxScaling 
            scaler = pickle.load(open("pickle_saves/preprocess/minMaxScaler.save", 'rb'))
            X = pd.DataFrame(scaler.transform(X))

            dfNew = pd.concat([y, X],axis=1)
            dfNew.columns=df.columns
            
        elif clfName in ['FF']: # Applico sia MinMaxScaling che PCA
            scaler = pickle.load(open("pickle_saves/preprocess/minMaxScaler.save", 'rb'))
            pca_scaler = pickle.load(open("pickle_saves/preprocess/pca.save", 'rb'))

            X = scaler.transform(X)
            X = pd.DataFrame(pca_scaler.transform(X))
            
            # Garantisce che il numero di colonne nel DataFrame dfNew sia corretto dopo l'applicazione della PCA. 
            dfNew = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
            
        else:
            raise ValueError(f"preprocess name {clfName} is not supported")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
    return dfNew


def load(clfName):
    # Descrizione parametri in input
    # - clfName: stringa che identifica la tecnica di ML da utilizzare, e che può assumere i valori 
    #            indicati nella slide successiva

    # Output:
    # Oggetto Python relativo all’istanza dell’algoritmo di ML, con iper-parametri determinati 
    # durante la fase di TRAINING. Se l’algoritmo non è stato implementato, deve restituire un 
    # valore None.

    if clfName == 'LR':
         clf=pickle.load(open("pickle_saves/models/LR.save", 'rb'))
    elif clfName == 'RF':
        clf=pickle.load(open("pickle_saves/models/RFR.save", 'rb'))
    elif clfName == 'KNR':
        clf=pickle.load(open("pickle_saves/models/KNR.save", 'rb'))
    elif clfName == 'SVR':
        clf=pickle.load(open("pickle_saves/models/SVR.save", 'rb'))
    elif clfName == 'FF':
        checkpoint = torch.load('pickle_saves/models/FF.save')
        best_params = checkpoint['params']
        # Create a new model instance with the best parameters
        clf = FeedForward(
            52, # applichiamo PCA con 52 n_components
            best_params['hidden_size1'], 
            best_params['hidden_size2'], 
            best_params['hidden_size3'],  
            best_params['hidden_size4'], 
            best_params['hidden_size5'], 
            best_params['hidden_size6'], 
            best_params['hidden_size7'], 
            best_params['hidden_size8'], 
            best_params['negative_slope']
        )

        clf.load_state_dict(checkpoint['model_state_dict'])
        clf.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    elif clfName == 'TB':
        clf=pickle.load(open("pickle_saves/models/TB.save", 'rb'))
    elif clfName == 'TF':
        clf=TabularModel.load_model("pickle_saves/models/TT")
    else:
        # Ritorna None se l'algoritmo non è stato implementato
        return None
    
    return clf   


def predict(df, clfName, clf):
    # Descrizione parametri in input
    # - df: dataframe di TEST, ottenuto come output del metodo di preprocess descritta precedentemente 
    # - clfName: stringa che identifica la tecnica di ML da utilizzare
    # - clf: istanza dell’algoritmo di ML, ottenuto come output del metodo di load
    #        descritto precedentemente

    # Output:
    # Dizionario Python contenente i valori di prestazioni dell’algoritmo di ML quando 
    # eseguito sui dati di TEST preprocessati

    X=df[df.columns[1:]]
    y=df[df.columns[:1]]

    if clfName == 'LR':
        ypred=clf.predict(X.values)
    if clfName == 'RF':
        ypred=clf.predict(X.values)
    if clfName == 'KNR':
        ypred=clf.predict(X.values)
    if clfName == 'SVR':
        ypred=clf.predict(X.values)
    if clfName == 'FF':
        dataset = MyDataset(X.values, y)
        data_loader= DataLoader(dataset)
        y, ypred = test_model(clf, data_loader ,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if clfName == 'TB':
        pred_df=clf.predict(df)
        ypred = pred_df['Year_prediction']
    if clfName == 'TF':
        pred_df=clf.predict(df)
        ypred = pred_df['Year_prediction']

    # Calcolo delle metriche
    mse = mean_squared_error(y, ypred)
    mae = mean_absolute_error(y, ypred)
    mape = mean_absolute_percentage_error(y, ypred)
    r2 = r2_score(y, ypred)

    # Ritorna un dizionario contenente le metriche di prestazione
    performance_metrics = {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'r2score': r2
    }

    return performance_metrics

# main usato per testare l'esecuzione e i vari modelli caricati
# def main():
#     FILENAME = '../TrainingModule/data.zip'
#     CLF_NAME_LIST = ['LR','RF','KNR','SVR','FF','TB','TF']
#     df_test = pd.read_csv(FILENAME)

#     #Esecuzione degli algoritmi
#     for clfName in CLF_NAME_LIST:
#         dfProcessed = preprocess(df_test, clfName)
#         clf = load(clfName)
#         perf = predict(dfProcessed, clfName, clf)
#         print("RESULT team: "+str(getName())+" algoName: "+ clfName + " perf: "+ str(perf))


# if __name__ == "__main__":
#     main()