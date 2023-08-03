
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import numpy as np
import pandas as pd

import util
import data_visualization.colors as co

# global variables 
dicionario = 'score_liwc2015'


def polarity_score(p_table):
    """Gráfico da polaridade dos discursos dos candidatos"""
    fig, ax = plt.subplots()

    candidatos = [str.capitalize(x) for x in p_table.index.values]

    y_pos = np.arange(len(candidatos))

    x = p_table[dicionario].values

    bar = ax.barh(y_pos,x)

    ax.get_children()[2].set_color(co.c_tebet1)
    ax.get_children()[3].set_color(co.c_lula)
    ax.get_children()[0].set_color(co.c_bolsonaro)
    ax.get_children()[1].set_color(co.c_ciro)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(candidatos, weight='bold')

    colors=[co.c_bolsonaro, co.c_ciro, co.c_tebet1, co.c_lula]
    for color,tick in zip(colors,ax.yaxis.get_major_ticks()):
        tick.label1.set_color(color) #define a propriedade da cor

    ax.xaxis.set_major_formatter(mtick.NullFormatter())

    ax.set_title('Polaridade do discurso por candidato', fontsize=12,fontweight='semibold',pad=15,  y=0.970, x=0.35, color=co.gray_9)

    for index in range(len(x)):
      ax.text(x[index]-.005, y_pos[index]-.055, x[index].round(4), size=11, fontweight='black', color='w')

    # remoção de linhas de borda do gráfico
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # remover linha que indica label no eixo y
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

    plt.show()


def subplot_polarity(candidato,color, interview_score_polarity, axes):
    '''gráfico para determinar as polaridades das entrevistas de determinado candidato'''

    p_table_candidato = interview_score_polarity[interview_score_polarity['candidato'] == candidato]
    entrevista = p_table_candidato.index.values
    y_pos = np.arange(len(entrevista))

    x = p_table_candidato[dicionario].values

    axes.barh(y_pos,x,color=color)
    axes.set_yticks(y_pos)
    axes.set_yticklabels(entrevista)

    # remoção de linhas de borda do gráfico
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['bottom'].set_visible(False)

    axes.set_title(candidato.capitalize(), fontsize=10, fontweight='bold', loc='left', color=color)
    
    # definição de cor
    axes.xaxis.set_tick_params(labelcolor=co.gray_9)

    # remover linha que indica label no eixo y
    axes.yaxis.set_tick_params(length=0)

    # cor eixo  y    
    axes.tick_params(axis='y', colors=co.gray_9)

    # escala 
    minl=0
    maxl=0.042
    axes.set_xlim(xmin=minl, xmax=maxl)  


def text_polarity_trend(titulo, df_texto):
    '''determinação da polaridade e sua variação ao longo de um discurso'''
    df_texto = df_texto.copy()
    
    dicionario = 'score_liwc2015'

    # valores x e y 
    y = util.grouped_array_mean(df_texto[dicionario].values,10)
    x = np.arange(y.shape[0])

    fig, ax = plt.subplots(figsize=(10,4))

    ax.bar(x,y,color='black')

    # linha horizontal
    ax.axhline(0, color='grey', lw=0, alpha=0.7)
    ax.axhline(0, color='grey', lw=0.6, alpha=0.7, ls='--')

    # limites eixos x e y
    lim=.45
    margem = .0
    ax.set_xlim(xmin=x[0], xmax=x[-1])
    ax.set_ylim(ymin=-lim-margem, ymax=lim+margem)

    # remoção de linhas de borda do gráfico
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # remover linha que indica label no eixo y
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

    # nomeação tick eixo y 
    ax.set_ylabel('Polaridade de sentimentos', labelpad=12, color=co.gray_9,loc='bottom')

    # alteração de cor nos rótulos dos eixos 
    ax.tick_params(axis='y', colors=co.gray_9)
    ax.xaxis.set_tick_params(labelcolor='gray')

    # polaridade na legenda 
    ax.legend(['Polaridade:  ', y.mean().round(4)],handlelength=0, handletextpad=0, loc=0)

    ax.set_title(titulo, fontsize=12, fontweight='semibold', pad=12, y=1.0, x=0.225, color=co.gray_9)

    plt.show()       


def get_opinion(count_opinion_words, opinions, multi, titulo, destaque='none'):
    '''gráfico da frequência de utilização de palavras que remetem opiniões'''

    labels = [str.capitalize(x) for x in count_opinion_words.index.values]

    y = np.arange(len(labels))*multi

    fig, ax = plt.subplots(figsize=(8,6))

    width_position = 0
    width = 0.45
    color_cmap = 0.4

    for opinion in opinions:
        if destaque == opinion:
            cmap = plt.get_cmap('Blues')
            color  = cmap(0.85)
        else:
            cmap = plt.get_cmap('Greys')
            color = cmap(color_cmap)
            color_cmap += 0.10

        df_opinion = count_opinion_words[opinion]
        bar = ax.barh((y + width_position), df_opinion.values, width, label=df_opinion.name, color=color) #cmap(0.65)
        width_position += width
        
    ax.set_yticks(y+0.45)
    ax.set_yticklabels(labels)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best')

    # ax.set_xlim(xmin=0,xmax=400 )
    cmap = plt.get_cmap('Greys')
    gray_8 = cmap(0.88)
    ax.set_xlabel('Frequência relativa', color=gray_8, loc='left')
    ax.tick_params(axis='x', colors=gray_8)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=2, xmax=1))

    plt.grid(True, 'major', 'x', ls='--', lw=.5, c='k', alpha=.3)

    # remoção de linhas de borda do gráfico
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # remoção de linhas que indica label no eixo y
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

    ax.set_title(titulo, fontsize=12 ,pad=15,loc='left' , color=gray_8)

    plt.show()