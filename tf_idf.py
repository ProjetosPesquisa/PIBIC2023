import pandas as pd
import numpy as np
from util import flatten


def percentual_term_frequency(df, words_column, separation, new_column_name, separation_2=None):
    '''
    retorna a frequência de cada termo, podendo ser de dois modos: frequência de cada categoria 
    frequência de subcategorias dentro das categorias retorna DataFrame com coluna adicional
    ''' 

    df1 = df.copy()
    if separation_2 == None: # 1 categoria

        # encontrar o total de palavras por cada nome_separador 
        palavra_por_candidato = df1[[separation, words_column]].groupby(separation).count()

        list_lemma = [] # importa ordem
        list_candidato = [] # não importa ordem 
        list_term_frequency = [] # importa ordem 

        # criação de lista de frequência palavra ao nível do nome_separador 
        for i in set(df1[separation]):
            frequency = list(df1[(df1[separation] == i)][words_column].value_counts() / palavra_por_candidato.loc[i][0])
            candidato = [i] * len(frequency)
            lemma_df = list(df1[(df1[separation] == i)][words_column].value_counts().index)

            # adiciona a lista 
            list_lemma.append(lemma_df)
            list_candidato.append(candidato)
            list_term_frequency.append(frequency)

        # transforma as três listas criadas em um dataframe pandas com: nome_separação - palavra - frequência 
        all_frequency =  pd.DataFrame(zip(flatten(list_lemma), flatten(list_candidato),
            flatten(list_term_frequency)),
            columns =[words_column, separation, new_column_name ])

        # Merge (junta-se) a frequência atribuída a cada palavra de cada nome_separação ao df original, criando-se uma nova coluna.   
        df1['copy_index'] = list(df1.index)
        df_m = pd.merge(df1, all_frequency, on=[words_column, separation], how='inner')
        df_m.index = df_m['copy_index'].values
        df_m.drop('copy_index', inplace = True, axis = 1)
        df_m = df_m.sort_index()
 
    else: # 2 categorias (segunda categoria inerente a primeira)  
        list_lemma2 =  [] # importa ordem 
        list_candidato2 = [] # não importa ordem
        list_freq2 = [] # importa ordem 
        list_evento2 = [] #importa ordem

        for i in set(df1[separation]):
            list_freq = [] # importa ordem
            list_lemma = [] # Importa ordem 
            list_evento = [] # importa ordem

            palavra_por_entrevista = df1[(df1[separation] == i)].groupby(separation_2)[separation].count()
            for j in set(df1[(df1[separation] == i)][separation_2]):
                frequency = list(df1[(df1[separation] == i) & (df1[separation_2] == j)][words_column].value_counts()/palavra_por_entrevista[j])
                lemma_df1 = list(df1[(df1[separation] == i) & (df1[separation_2] == j)][words_column].value_counts().index)
                evento = [j] * len(lemma_df1)
                
                list_lemma.append(lemma_df1) # adiciona a lista
                list_freq.append(frequency)
                list_evento.append(evento)
                
            candidato = [i] * len(flatten(list_lemma))

            list_lemma2.append(flatten(list_lemma))  # adiciona a lista
            list_freq2.append(flatten(list_freq)) 
            list_evento2.append(flatten(list_evento))
            list_candidato2.append(candidato)

        # transforma as quatro listas criadas em um dataframe com: nome_separação - nome_separação2 - palavra - frequência
        all_frequency =  pd.DataFrame(zip(flatten(list_candidato2),flatten(list_evento2), flatten(list_lemma2), flatten(list_freq2)),
                columns =[separation, separation_2, words_column, new_column_name])   

        # Merge (junta-se) a frequência atribuída a cada palavra de cada nome_separação e nome_separação2 ao df original, criando-se uma nova coluna.
        df1['copy_index'] = list(df1.index)
        df_m = pd.merge(df1, all_frequency, on=[words_column, separation, separation_2], how='inner')
        df_m.index = df_m['copy_index'].values
        df_m.drop('copy_index', inplace = True, axis = 1)
        df_m = df_m.sort_index()
        
    return df_m #retorna dataframa modificado
       

def inverse_document_frequency(df, words_column, separation, new_column_name, separation_2=None):
    '''
    retorna --> 'inverse document frequency': podendo ser de dois modos: frequência de cada categoria 
    frequência de subcategorias dentro das categorias retorna DataFrame com coluna adicional
    '''

    df1 = df.copy()
    if separation_2 == None:     # 1 categoria

        list_lemma1 = []
        
        for i in set(df1[separation]):
            df_sem_duplicadas = df1[(df1[separation] == i)].drop_duplicates(words_column)
            list_lemma = list(df_sem_duplicadas[words_column].values)
            list_lemma1.append(list_lemma)

         # transforma as duas listas criadas em um dataframe pandas com: palavra - frequência    
        all_events =  pd.DataFrame(zip(flatten(list_lemma1)), columns = [words_column])        

        # encontra-se a quantidade de palavras em cada documento e ln(quant. docs/ freq. palavras)
        frequency_docs = all_events.value_counts(words_column)
        df_idf =  pd.DataFrame(zip(frequency_docs.index, frequency_docs), columns = [words_column, new_column_name])  
        df_idf[new_column_name] = np.log(len(set(df1[separation])) / df_idf[new_column_name])

        # Merge (junta-se) a frequência inversa do doc atribuída a cada palavra ao df original, criando-se uma nova coluna.
        df1['copy_index'] = list(df1.index)
        df_m = pd.merge(df1, df_idf, on=[ words_column], how='inner')
        df_m.index = df_m['copy_index'].values
        df_m.drop('copy_index', inplace = True, axis = 1)
        df_m = df_m.sort_index()
     
    else: # 2 categorias (segunda categoria inerente a primeira)  
    
        list_lemma2 =  []
        list_candidato2 = []
        list_idf2 = []
        for i in set(df1[separation]):

            list_evento = [] # não importa ordem
            list_lemma = [] 
            for j in set(df1[separation_2]):
                    a = df1[(df1[separation] == i) & (df1[separation_2] == j)].drop_duplicates(words_column)
                    lemma_df1 = list(a[words_column].values)
                    list_lemma.append(lemma_df1)
                    evento = list(set(a[separation_2])) * len(lemma_df1)
                    list_evento.append(evento)
            candidato = [i] * len(flatten(list_lemma))

            # transforma as três listas criadas em um dataframe pandas com: candidato- palavra - lemma  
            all_events =  pd.DataFrame(zip(flatten(list_evento), candidato, flatten(list_lemma)),
                            columns =[separation_2, separation, words_column])

            # encontra-se a quantidade de palavras em cada documento e ln(quant. docs/ freq. palavras)
            idf_values = np.log(len(set(flatten(list_evento))) / all_events.value_counts(words_column))
            lemma2 = list(idf_values.index)
            candidato2 = len(lemma2) * [i]
            idf = list(idf_values.values)

            # adiciona a lista
            list_lemma2.append(lemma2)
            list_candidato2.append(candidato2)
            list_idf2.append(idf) 

        # transforma as três listas criadas em um dataframe pandas com: candidato- palavra - idf 
        all_idf =  pd.DataFrame(zip(flatten(list_candidato2), flatten(list_lemma2), flatten(list_idf2)),
                    columns =[separation, words_column, new_column_name])   

        # Merge (junta-se) a frequência inversa do doc atribuída a cada palavra a cada separação ao df original, criando-se uma nova coluna.
        df1['copy_index'] = list(df1.index)
        df_m = pd.merge(df1, all_idf, on=[separation, words_column], how='inner')
        df_m.index = df_m['copy_index'].values
        df_m.drop('copy_index', inplace = True, axis = 1)
        df_m = df_m.sort_index()

    return df_m #retorna dataframa modificado


def get_tf_idf(df, words_column, separation, new_column_name, separation2=None):
    '''
    retorna a multiplicação do TF com IDF podendo de ser com 1 ou  2 categorias 
    retorna DataFrame com coluna adicional the frequency of a term adjusted for how rarely it is used
    '''

    df = df.copy()
    df1 = percentual_term_frequency(df, words_column, separation, new_column_name, separation_2 = separation2)
    df2 = inverse_document_frequency(df, words_column, separation, new_column_name, separation_2 = separation2)
    df[new_column_name] = df1[new_column_name] * df2[new_column_name]
    
    return df
