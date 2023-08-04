import pandas as pd 
import numpy as np
import nltk

from util import flatten

def get_files_tokens(folder, instrucoes):

    lista_tokens = []
    lista_evento = []
    lista_candidato = []

    for i in range(instrucoes.shape[0]):
        file = instrucoes['arquivo'][i]
        file_path = '{}//{}'.format(folder, file)
        with open(file_path, encoding = "ANSI") as f:
            textos = f.read()
        
        tokens = nltk.word_tokenize(textos)  
        lst_tokens = list(tokens)
        lst_evento = [instrucoes['evento'][i]] * len(tokens)
        lst_candidato = [instrucoes['candidato'][i]] * len(tokens)
        lista_candidato.append(lst_candidato)
        lista_evento.append(lst_evento)
        lista_tokens.append(tokens)
 
    arquivos = pd.DataFrame(zip(flatten(lista_candidato), flatten(lista_evento),
        flatten(lista_tokens)),
        columns =['candidato', 'evento', "tokens"])    
    
    return arquivos


class Pre_processing():
    """Pré processamento dos dados"""
  
    def __init__(self, dataframe, tokens_column_name, refined_column = None): 
       self.df = dataframe.copy()
       self.tokens = tokens_column_name
       self.df_refined = refined_column
   
    
    def remove_stopwords(self):
      'remoção de stopwords'
      stopwords = nltk.corpus.stopwords.words('portuguese')
      for palavra in stopwords:
         self.df.loc[(self.df[self.tokens].values == palavra)] = np.nan

      return self.df

    def remove_unnecessarywords(self):
      'remoção de outras palavras desnecessárias'
      unwords = ['senhor','senhora','música', ',','[',']',"né","lá","aí",
                "aplausos","`",'vai', "g1", 'sao',"tá", "ta", "gente",
                "porque","sabe","vou","então", 'ter', 'coisa', 'tô',
                'cardoso', 'henrique', 'luiz', 'gomes', 'temer', 'tebete', 'tebet',
                'ah','ee','isso','de','_','uai','fi','cê','ô','tf','êh','bds','ó',
                '_ ','/','d','.','guilh','$','%']

      for palavra in unwords:
         self.df.loc[(self.df[self.tokens].values == palavra)] = np.nan

      return self.df

   
    def replace_words(self):
      '''substituição de palavras'''

      replaced_dict = {'guedes':'paulo_guedes', 'jefferson':'roberto_jefferson', 
      'bolsa':'bolsa_familia', 'fernando':'fernando_henrique_cardoso','inácio':'luiz_inácio', 
      'ciro':'ciro_gomes', 'michel':'michel_temer', 'simone':'simone_tebet', 
      'supremo':'supremo_tribunal', 'emergencial':'auxílio_emergencial', 'petrobra': 'petrobras', 'pic':'pix',
      'tef':'stf', 'stef':'stf','i2022':'2022','dpt':'pt'}
      replaced_list = list(replaced_dict.keys())
      for palavra in replaced_list:
         self.df.loc[(self.df[self.tokens].values == palavra), self.tokens] = replaced_dict.get(palavra) 
      
      return self.df   


    def column_refined(self):
      '''função para retornar objeto'coluna' após todas as mudanças'''
      return self.df[self.tokens]
    

def process_tokens(arquivos):
    '''remoção de stopwords, normalização e substituição de palavras'''
    arquivos = arquivos.copy()
    arquivos['tokens'] = arquivos['tokens'].apply(lambda x: x.lower())

    arquivo_modified = Pre_processing(dataframe=arquivos, tokens_column_name='tokens')
    arquivo_modified.remove_stopwords()
    arquivo_modified.replace_words()
    arquivo_modified.remove_unnecessarywords()
    arquivos['tokens'] = arquivo_modified.column_refined()
    arquivos.dropna(inplace=True)
    arquivos.reset_index(drop=True, inplace=True)
    
    return arquivos


def read_liwc(file):
    '''Leitura do dicionário LIWC para Dataframe'''
    liwc = pd.read_csv(file)
    liwc.rename(columns={'DicTerm': 'lemma'}, inplace=True)

    #seleção de colunas
    liwc = liwc[['lemma', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad',
                        'family', 'health', 'ingest', 'work', 'money', 'relig', 'death'
                        ]].copy()

    # criação coluna 'score' com valore 1 (+) e -1 (-)
    posemo = liwc['posemo'].values
    negemo = liwc['negemo'].values
    liwc['score_liwc2015'] = np.where(posemo == 'X', 1, 0) + np.where(negemo == 'X', -1, 0)

    cols = list(liwc.columns) # alteração da ordem das colunas 
    cols.reverse()
    liwc = liwc[cols]
    
    liwc.drop(['posemo','negemo'],axis=1, inplace=True)
    
    return liwc
