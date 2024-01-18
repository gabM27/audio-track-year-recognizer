''' 
 Codice che implementa 4 funzioni Python:
    1. getName() --> restituisce un identificativo dello studente o del gruppo.
    2. preprocess(df, clfName) --> effettua il pre-processing dei dati di test forniti in input.
    3. load(clfName) --> istanzia la tecnica di ML con nome clfName.
    4. predict(df, clfName, clf) --> esegue il modello ML (oggetto clf) sui dati di TEST preprocessati (df)
'''

##############################################################################
class ComponenteGruppo:
    def __init__(self, nome, cognome, email, matricola):
        self.nome = nome
        self.cognome = cognome
        self.email = email
        self.matricola = matricola            

##############################################################################
def getName():
    componente1 = ComponenteGruppo("Andrea","Bianchi", "andrea.bianchi26@studio.unibo.it", "0001100494") 
    componente2 = ComponenteGruppo("Gabriele","Magazzù", "gabriele.magazzu@studio.unibo.it", "0001102322")
    
    nome_gruppo = "Pair Forza"
    
    output = f"Nome gruppo: {nome_gruppo}\nComponenti del gruppo:\n{componente1.nome} {componente1.cognome} ({componente1.email}, {componente1.matricola})\n{componente2.nome} {componente2.cognome} ({componente2.email}, {componente2.matricola})"

    return output
    
    
    
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