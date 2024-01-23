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
    if clfName in ['LR', 'RF', 'KNR', 'SVR']:
        # Implementare il pre-processing specifico
        
        pass #da togliere quando si implementerà il resto
        
    elif clfName in ['FF', 'TB', 'TF']:
        # Implementare il pre-processing specifico 
        
        pass #da togliere quando si implementerà il resto
    else:
        
        raise ValueError("Stringa che identifica la tecnica di ML da utilizzare NON identificata.\nRiprovare con una tra le seguenti:'LR', 'RF', 'KNR', 'SVR', 'FF', 'TB', 'TF'.")

    # Ritorna il DataFrame pre-processato
    return df


##############################################################################
# versione iniziale, da cambiare

def load(clfName):
    # Implementare la creazione delle istanze per ogni algoritmo di ML desiderato
    
    # ATTENZIONE: bisogna caricare gli algoritmi con gli iperparametri determinati nella fase di training
    
    # Se non abbiamo implementato un algoritmo, dobbiamo ritornare None anche in quel caso.
    
    if clfName == 'LR':
        # TODO return LinearRegression()
         pass #da togliere quando si implementerà il resto
    elif clfName == 'RF':
        # TODO return RandomForestRegressor()
         pass #da togliere quando si implementerà il resto
    elif clfName == 'KNR':
        # TODO return KNeighborsRegressor()
         pass #da togliere quando si implementerà il resto
    elif clfName == 'SVR':
        # TODO return SVR()
         pass #da togliere quando si implementerà il resto
    elif clfName == 'FF':
        # TODO return MLPRegressor()
         pass #da togliere quando si implementerà il resto
    elif clfName in ['TB', 'TF']:
        # TODO return TabNetRegressor()
         pass #da togliere quando si implementerà il resto
    else:
        # Ritorna None se l'algoritmo non è stato implementato
        return None        



##############################################################################
# versione iniziale, da cambiare

def predict(df, clfName, clf):
    # TODO esecuzione del modello di machine learning sui dati di test preprocessati

    # TODO predizione
    
    # TODO calcolo delle metriche di prestazione
    mse = 'qualcosa'
    mae = 'qualcosa1'
    mape = 'qualcosa2'
    r2 = 'qualcosa3'

    # Ritorna un dizionario contenente le metriche di prestazione
    performance_metrics = {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'r2score': r2
    }

    return performance_metrics