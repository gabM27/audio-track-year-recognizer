from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import pickle
import pandas as pd
import torch

import sys
import os
# Aggiungi il percorso della directory 'models'
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../TrainingModule')))

from feedforward import FeedForward, MyDataset, test_model

''' 
 Codice che implementa 4 funzioni Python:
    1. getName() --> restituisce un identificativo dello studente o del gruppo.
    2. preprocess(df, clfName) --> effettua il pre-processing dei dati di test forniti in input.
    3. load(clfName) --> istanzia la tecnica di ML con nome clfName.(LR, RF, KNR, SVR, FF, TB, TF)
    4. predict(df, clfName, clf) --> esegue il modello ML (oggetto clf) sui dati di TEST preprocessati (df)
'''

##############################################################################
# Teachers' example:
'''
MY_UNIQUE_ID = "343435631"

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import pickle

# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID

# Input: Test dataframe
# Output: PreProcessed test dataframe
def preprocess(df, clfName):
    if ((clfName == "lr") or (clfName == "svr")):
        X = df.iloc[:, :5]
        y = df.iloc[:, 5]
        scaler = pickle.load(open("scaler.save", 'rb'))
        X = pd.DataFrame(scaler.transform(X))
        dfNew = pd.concat([X, y], axis = 1)
        return dfNew

# Input: Regressor name ("lr": Linear Regression, "SVR": Support Vector Regressor)
# Output: Regressor object
def load(clfName):
    if (clfName == "lr"):
        clf = pickle.load(open("regression.save", 'rb'))
        return clf
    elif (clfName == "svr"):
        clf = pickle.load(open("svr.save", 'rb'))
        return clf
    else:
        return None
    
# Input: PreProcessed dataset, Regressor Name, Regressor Object 
# Output: Performance dictionary
def predict(df, clfName, clf):
    X = df.iloc[:, :5]
    y = df.iloc[:, 5]
    ypred = clf.predict(X)
    mse = mean_squared_error(ypred, y)
    mae = mean_absolute_error(ypred, y)
    mape = mean_absolute_percentage_error(ypred, y)
    r2 = r2_score(ypred, y)
    perf = {"mse": mse, "mae": mae, "mape": mape, "r2square": r2}
    return perf
'''

##############################################################################

MY_UNIQUE_ID = "Pair Forza" #Tocca dai, dai dai! Si va a LETTOOOOOOH
# Output: unique ID of the team
def getName():
    return MY_UNIQUE_ID


##############################################################################
# versione iniziale, da cambiare

def preprocess(df, clfName):
    
    # Carica i parametri determinati durante la fase di training
    # Questi parametri dovrebbero essere salvati in qualche modo durante il training
    # e successivamente caricati qui durante il pre-processing.

    # Applica il pre-processing ai dati di test in base al regressore scelto
    # In questo caso, che è solo un esempio di come potremmo implementare la funzione preprocess, 
    # assumo che la tecnica di preprocess sia diversa differenziando le reti neurali dalle altre tecniche

    X=df.iloc[:,1:]
    y=df.iloc[:,0]

    try:
        if clfName in ['LR', 'RF', 'KNR', 'SVR', 'TB', 'TF']:
            scaler = pickle.load(open("../pickle_saves/preprocess/minMaxScaler.save", 'rb'))
            X = pd.DataFrame(scaler.transform(X))

            dfNew = pd.concat([y, X], axis=1)

        elif clfName in ['FF']:
            scaler = pickle.load(open("../pickle_saves/preprocess/minMaxScaler.save", 'rb'))
            pca_scaler = pickle.load(open("../pickle_saves/preprocess/pca.save", 'rb'))

            X = scaler.transform(X)
            X = pca_scaler.transform(X)

            dfNew = MyDataset(X, y)

        else:
            raise ValueError(f"Classifier name {clfName} is not supported")
    
        
        # dfNew = pd.concat([y, X], axis=1)

        # Ritorna il DataFrame pre-processato
        return dfNew

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


##############################################################################
# versione iniziale, da cambiare

def load(clfName):
    # Implementare la creazione delle istanze per ogni algoritmo di ML desiderato
    
    # ATTENZIONE: bisogna caricare gli algoritmi con gli iperparametri determinati nella fase di training
    
    # Se non abbiamo implementato un algoritmo, dobbiamo ritornare None anche in quel caso.
   
    if clfName == 'LR':
         clf=pickle.load(open("../pickle_saves/models/LR.save", 'rb'))
    elif clfName == 'RF':
        model=pickle.load(open("../pickle_saves/models/RFR.save", 'rb'))
         #pass #da togliere quando si implementerà il resto
    elif clfName == 'KNR':
        model=pickle.load(open("../pickle_saves/models/KNN.save", 'rb'))
    elif clfName == 'SVR':
        clf=pickle.load(open("../pickle_saves/models/SVR.save", 'rb'))
    elif clfName == 'FF':
        
        checkpoint = torch.load('../pickle_saves/models/FF.save')
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
        # Caricamento del modello TabNet
        clf = torch.load("../pickle_saves/models/TB.save", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
    elif clfName == 'TF':
        return None
    else:
        # Ritorna None se l'algoritmo non è stato implementato
        return None
    
    return clf   



##############################################################################
# versione iniziale, da cambiare

def predict(df, clfName, clf):

    if (clfName != "FF"):
        #Divisione dataset X, y_true
        X=df[df.columns[1:]]
        y=df[df.columns[:1]]

    # per fixare errore: TypeError: Feature names are only supported if all input features have string names, 
    # but your input has ['int', 'str'] as feature name / column name types. 
    # If you want feature names to be stored and validated, you must convert them all to strings, by using X.columns = X.columns.astype(str) for example. 
    # Otherwise you can remove feature / column names from your input data, or convert them all to a non-string data t
    #X.columns=X.columns.astype(str)

    # TODO esecuzione del modello di machine learning sui dati di test preprocessati
    if clfName == 'LR':
        ypred=clf.predict(X)
    if clfName == 'RF':
        ypred=clf.predict(X)
    if clfName == 'KNR':
        ypred=clf.predict(X)
    if clfName == 'SVR':
        ypred=clf.predict(X)
    if clfName == 'FF':
        y, ypred = test_model(clf, df,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if clfName == 'TB':
        pred_df=clf.predict(df)
        ypred = pred_df['Year_prediction']
    if clfName == 'TF':
        ypred=clf.predict(X)

    #calcolo delle metriche
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


# def main():
#     FILENAME = '../data.zip'
#     CLF_NAME_LIST = ["LR","RF","KNR","SVR","FF","TB","TF"]
#     df = pd.read_csv(FILENAME)

#     #Esecuzione degli algoritmi
#     #for modelName in CLF_NAME_LIST:
#     dfProcessed = preprocess(df, modelName)
#     clf = load(modelName)
#     perf = predict(dfProcessed, modelName, clf)
#     print("RESULT team: "+str(getName())+" algoName: "+ modelName + " perf: "+ str(perf))


# if __name__ == "__main__":
#     main()