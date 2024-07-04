# Necessario eseguire il file requirements.txt per installare le librerie necessarie

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from feedforward_test import FeedForward, MyDataset, test_model
from pytorch_tabular import TabularModel
import pickle
import pandas as pd
import torch

MY_UNIQUE_ID = "Pair Forza"

# Output: ID univico del gruppo
def getName():
    return MY_UNIQUE_ID

def preprocess(df, clfName):
    # Descrizione parametri in input
    # - df: Dataframe Pandas contenente i dati di TEST. I dati hanno la stessa struttura (numero 
    #       e nomi attributi) di quelli usati per il TRAINING.
    # - clfName: stringa che identifica la tecnica di ML da utilizzare.

    # Output:
    # Dataframe Pandas ottenuto come risultato del pre-processamento

    X=df.iloc[:,1:]
    y=df.iloc[:,0]
    dfNew=pd.DataFrame()

    try:
        if clfName in ['LR', 'RF', 'KNR', 'SVR', 'TB', 'TF']: # Applichiamo solo MinMaxScaling 
            scaler = pickle.load(open("pickle_saves/preprocess/minMaxScaler.save", 'rb'))
            X = pd.DataFrame(scaler.transform(X))

            dfNew = pd.concat([y, X],axis=1)
            dfNew.columns=df.columns
            
        elif clfName in ['FF']: # Applichiamo sia MinMaxScaling che PCA
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
    # - clfName: stringa che identifica la tecnica di ML da utilizzare.

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
        # o se clfName ha un valore diverso da quelli gestiti
        return None
    
    return clf   


def predict(df, clfName, clf):
    # Descrizione parametri in input
    # - df: dataframe di TEST
    # - clfName: stringa che identifica la tecnica di ML da utilizzare
    # - clf: istanza dell’algoritmo di ML, ottenuto come output del metodo di load

    # Output:
    # Dizionario Python contenente i valori di prestazioni dell’algoritmo di ML

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

#     df_test=df_test.head(100)
#     #Esecuzione degli algoritmi
#     for clfName in CLF_NAME_LIST:
#         dfProcessed = preprocess(df_test, clfName)
#         clf = load(clfName)
#         perf = predict(dfProcessed, clfName, clf)
#         print("RESULT team: "+str(getName())+" algoName: "+ clfName + " perf: "+ str(perf))


# if __name__ == "__main__":
#     main()