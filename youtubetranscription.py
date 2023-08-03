# script para obter, salvar e nomear as transcrições a partir de links armazenados em arquivo .xlsx 

import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import os

# 0 bolsonaro 
# 1 lula
# 2 ciro
# 3 tebet     

file_path = "teste"

# obtenção das transcrições 
matriz_entrevistas = pd.read_excel("entrevistas_candidatos_2022.xlsx")
plataformas_comunicacao = matriz_entrevistas.columns[1:]

lista_links = []
for plataforma in plataformas_comunicacao:
    for a in range(matriz_entrevistas.shape[0]):
        u = matriz_entrevistas[plataforma][a]
        conjunto = []
        conjunto.append(u)
        conjunto.append(plataforma)
        conjunto.append(a)
        if type(conjunto[0]) == str:
            lista_links.append(conjunto)


candid = []
arquiv = []
evento = []
for link in lista_links:
    a = link[0]
    id = a.split("=")[1].split("&")[0]
    transcript = YouTubeTranscriptApi.get_transcript(id,  languages=['pt']) # retorna transcrição como lista

    meio_c = link[1]
    candidato = matriz_entrevistas.iloc[link[2],0]
    data = "DATA"  # campo preenchido posteriormente
    
    nome_arquivo = "{}_{}_{}{}.txt".format(meio_c, candidato, data, id)
    
    candid.append(candidato) 
    arquiv.append(nome_arquivo)  
    evento.append(meio_c)

    # criação de arquivo txt
    with open(os.path.join(file_path, nome_arquivo), "w") as f:
    
        for u in transcript: # interação de cada linha
            f.write("{} \n ".format(u['text']))

# criação de arquivo para instrução de leitura 
instrucoes = pd.DataFrame(list(zip(candid,arquiv,evento)),
               columns =['candidato', 'arquivo', "evento"])
instrucoes.set_index('candidato', inplace = True)

instrucoes.to_csv(os.path.join(file_path,"instruções.csv"))   