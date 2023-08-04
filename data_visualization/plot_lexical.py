from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import nltk

import util
import data_visualization.colors as co

# global variables 
dicionario = 'score_liwc2015'

def get_frequency(df_candidato, token_column, lista_destaques=[], color_destaque='k', n=15, y_subtitle=0.955):
    '''retorna gráfico das palavras mais utilizadas'''

    freq_dist = nltk.FreqDist(util.objeto_nltk(df_candidato, token_column))
    pd_freq_dist = pd.DataFrame(list(zip(list(freq_dist.keys()), list(freq_dist.values()))),
               columns =['word', 'val'])

    multi = 1.55

    xy = pd_freq_dist.sort_values('val', ascending=False).head(n)
    xy = xy.sort_values('val', ascending=True)
    
    labels = [str.capitalize(j) for j in xy['word'].values.astype(str)]
    y_int = np.arange(len(labels)) * multi
    
    width = xy['val'].values.astype(int)
    height = 1.15

    # plotagem da figura
    fig, ax = plt.subplots(figsize=(7,5))

    # padrão de cor para elementos básicos da figura
    cmap = plt.get_cmap('Greys')
    color = cmap(0.85)

    ax.barh(y_int, width=width, height=height, color=color)
    ax.set_yticks(y_int)
    ax.set_yticklabels(labels, color=color)

    # definição da cor para xlabel, para o rotulo dos dados e a barra (bottom spine)
    ax.set_xlabel('Frequência absoluta', color=co.gray_8, loc='left')
    ax.tick_params(axis='x', colors=co.gray_8)
    ax.spines['bottom'].set_edgecolor(co.gray_8)

    # remoção de linhas de borda do gráfico
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # remover linha que indica label no eixo y
    ax.yaxis.set_tick_params(length=0)  

    if len(lista_destaques)>0:
        # destaque de cor no gráfico
        condition = xy['word'].isin(lista_destaques)
        y_int_slice = y_int[condition]
        width_slice = xy[condition]['val'].values.astype(int)
        ax.barh(y_int_slice, width=width_slice, height=height, color=color_destaque)

    name = str.capitalize(list(set(df_candidato['candidato']))[0])

    fig.suptitle(name, y=y_subtitle, x=0.1655, fontweight='bold')
    ax.set_title('Palavras mais utilizadas ', pad=12, color=color, loc='left', y=0.97)
    plt.show()


def get_cor(data, n, title):
    '''correlação das n palavras'''
    corr_matrix = data.head(n).drop('POS',axis=1).corr()
    color = 'viridis'
    sns.heatmap(corr_matrix, vmin=-1, vmax=1, cmap=color, annot=True)
    plt.title(title)
    plt.show()


def word_scatter(pessoa1, pessoa2, word_frequency, x_position=0.210):
    '''Comparação da utilização das palavras entre dois candidatos'''
    col1 = pessoa1
    col2 = pessoa2

    dados = word_frequency[[col1,col2]].copy()
    dados['sorting'] = dados[col1] + dados[col2]
    dados_nomeacao = dados.sort_values('sorting', ascending=False).head(30)

    with plt.style.context('seaborn'):
        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(x=col1, y=col2, legend='full',  alpha=0.4,
                        s=(70),
                        data=dados)

        # nomeação de pontos 
        for i in range(dados_nomeacao.shape[0]):
            plt.text(x=dados_nomeacao[col1][i], y=dados_nomeacao[col2][i] , s=list(dados_nomeacao.index)[i],
            fontdict = dict(color='darkblue', size=12.5))

        # linha pontilhada 
        x = np.linspace(0, max(dados[[col1,col2]].max()),100000)
        y = x
        plt.plot(x, y, linewidth=1, linestyle='dashed', color='k')
        plt.xlabel(col1.capitalize(), fontsize=12.2,fontweight='normal')
        plt.ylabel(col2.capitalize(), fontsize=12.2,fontweight='normal')

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2, xmax=1))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=2, xmax=1))

        cmap = plt.get_cmap('Greys')
        color = cmap(0.65)
        ax.tick_params(axis='y', colors=color)
        ax.tick_params(axis='x', colors=color)

        cmap = plt.get_cmap('Greys')
        color = cmap(0.85)

        fig.suptitle(col2.capitalize() + ' x ' + col1.capitalize(), y=0.94, x=x_position, fontweight='bold', fontsize=14.5)
        ax.set_title("Comparação da utilização de palavras", pad=12, loc='left', y=0.992, color=color)
   
    plt.show()    


# palavras mais importantes por candidato
def create_subplot(candidato, color, dfcandidato, title_color, num_palavras, axes):
    '''Gráfico das palavras mais importantes analisando único candidato'''

    palavras_importancia = dfcandidato[dfcandidato['candidato']==candidato][['candidato','lemma',
                'tf_idf_candidato','POS']].sort_values('tf_idf_candidato',
                ascending=False).drop_duplicates('lemma').head(num_palavras)

    cmap = plt.get_cmap(color)
    colors = cmap(list(np.linspace(1,0.1,10)))

    lemmas= palavras_importancia['lemma'].values
    valores = palavras_importancia['tf_idf_candidato'].values
    axes.bar(lemmas,valores,color=colors)

    axes.set_xticks(lemmas)
    axes.set_xticklabels(labels=lemmas,rotation=45, ha='right', color=co.gray_9, fontsize=9.5)

    axes.tick_params(axis='y', colors='grey')

    # # remoção de linhas de borda do gráfico
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['bottom'].set_visible(False)

    axes.yaxis.set_tick_params(length=0)
    axes.xaxis.set_tick_params(color=co.gray_42)

    axes.set_title(str.capitalize(candidato), loc='left', color=title_color)
    

def highlight_important_words(dfcandidato, candidatos, num_palavras=10):
    '''Gráfico das palavras mais importantes dos 4 candidatos  de acordo com tf_idf'''

    fig, axes = plt.subplots(2,2,figsize=(10,8))
    fig.suptitle('Palavras mais importantes por candidato', fontsize=13, fontweight='bold', y=0.960, x=0.315)
 

    # [0,0]
    color = 'Greens'
    title_color = co.c_bolsonaro
    create_subplot(candidatos[0], color, dfcandidato, title_color, num_palavras, axes[0,0])

    # [0,1]
    color = 'Reds'
    title_color = co.c_lula
    create_subplot(candidatos[1], color, dfcandidato, title_color, num_palavras, axes[0,1])

    # [1,0]
    color = 'Blues'
    title_color = co.c_ciro
    create_subplot(candidatos[2], color, dfcandidato, title_color, num_palavras, axes[1,0])

    # [1,1]
    color = 'YlOrBr'
    title_color = co.c_tebet
    create_subplot(candidatos[3], color, dfcandidato, title_color, num_palavras, axes[1,1])

    plt.subplots_adjust(wspace=.33, hspace=0.60)

    axes[1,0].set_ylabel('Inverse Document Frequency x Term Frequency', labelpad=12, color='grey')

    plt.show()


# palavras mais importantes por entrevista
def create_subplot_2(entrevista, color, dfcandidato, num_palavras, axes):
    '''gráfico das palavras mais importantes analisando única entrevista de determinado candidato'''
    
    #define as palavras mais importantes de acordo com tf*idf por entrevista do candidato
    palavras_importancia = dfcandidato[dfcandidato['evento']==entrevista][['candidato','lemma',
                'tf_idf_evento2','POS']].sort_values('tf_idf_evento2',
                ascending=False).drop_duplicates('lemma').head(num_palavras)
    
    cmap = plt.get_cmap(color)
    color = cmap(list(np.linspace(1,0.1,10)))

    lemmas= palavras_importancia['lemma'].values
    valores = palavras_importancia['tf_idf_evento2'].values
    axes.bar(lemmas,valores, color=color)

    axes.set_xticks(lemmas)
    axes.set_xticklabels(labels=lemmas,rotation=45, ha='right', color=co.gray_9, fontsize=9.5)

    # remoção de linhas de borda do gráfico
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['bottom'].set_visible(False)

    axes.yaxis.set_tick_params(length=0)
    axes.xaxis.set_tick_params(color=co.gray_42)

    axes.tick_params(axis='y', colors='grey')
    
    axes.set_title(entrevista,fontsize=10.7, loc='left')

def highlight_words_interview(dfcandidato, color, num_palavras=10):
    '''gráfico das palavras mais importantes analisando 4 entrevistas do candidato'''

    nome = str.capitalize(list(set(dfcandidato['candidato']))[0])
    entrevistas = list(set(dfcandidato['evento']))
    fig, axes = plt.subplots(2,2,figsize=(10,8))

    # [0,0]
    create_subplot_2(entrevistas[0], color, dfcandidato, num_palavras, axes[0,0])

    # [0,1]
    create_subplot_2(entrevistas[1], color, dfcandidato, num_palavras, axes[0,1])

    # [1,0]
    create_subplot_2(entrevistas[2], color, dfcandidato, num_palavras, axes[1,0])

    # [1,1]
    create_subplot_2(entrevistas[3], color, dfcandidato, num_palavras, axes[1,1])

    axes[1,0].set_ylabel('Inverse Document Frequency x Term Frequency', labelpad=12, color='grey')

    plt.subplots_adjust(wspace=.35, hspace=0.65)
    fig.suptitle('Palavras mais importantes por entrevista: '+nome, fontsize=13, fontweight='bold', y=0.960, x=0.425)
    
    plt.show()    


def get_wordcloud(dfcandidato):
    """Nuvem de palavras"""

    wordcloud = WordCloud(width=900, height=500,
                    background_color='white',  # 'black'
                    min_font_size=10,
                    random_state=21,
                    max_font_size=180
                         ).generate(" ".join(list(dfcandidato['lemma'])))
    
    # plot figura WordCloud                     
    plt.figure(figsize = (10, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    plt.title(dfcandidato['candidato'].values[1].capitalize(), loc='left', fontsize=10, pad=20)

    plt.show()