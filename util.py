import nltk
import numpy as np

def objeto_nltk(df_candidato,col):
    '''conversão tokens[dataframe] em text NLTK'''
    candidato = nltk.Text(list(df_candidato[col]))
    return candidato
 
def flatten(l):
    '''remover lista de dentro de uma lista -- torna-la 'achatada'''
    return [item for sublist in l for item in sublist]

def grouped_array_mean(array_p, divisor):     
    '''retorna a média dos valores da array de x (divisor) em x elementos '''
    if array_p.shape[0]%divisor == 0:
        multi_0 = 0
    else:
        multi_0 =  divisor - (array_p.shape[0] % divisor)

    array_p = np.append(array_p, [0]*multi_0)
    array_p = np.reshape(array_p, (-1, divisor))
    grouped_array_mean = array_p.mean(axis=1)

    return grouped_array_mean