## Iniciando o documento com permissão para uso de caracteres do português (acentos)...
# -*- coding: utf-8 -*-

## Definindo o nome do documento
""" geometria.py """



## Importando as bibliotecas a serem usadas
""" bibliotecas usadas """
import numpy as np


## ============================================================================
## DEFININDO AS VARIÁVEIS
""" definição das variáveis (x) """

def criar_malha (iniciocomprimento=0.0, totalcomprimento=0.1, numerodepontos=501 ):
    
    """ Cria os pontos ao longo do bocal.

    Parâmetros:
        iniciocomprimento: posição inicial do bocal (m);
        totalcomprimento: posição final do bocal (m);
        numerodepontos: quantidade de pontos da malha.
 
    Retorno:
        x: vetor de posições (m);
        A: vetor de áreas (m²). """

## Certificando o funcionamento da geometria: valores maiores que 0
    if totalcomprimento <= iniciocomprimento : 
    
        raise ValueError ("totalcomprimento deve ser maior que iniciocomprimento")

    if numerodepontos<2 or int(numerodepontos) != numerodepontos:
    
        raise ValueError ("são necessários pelo menos 2 pontos (entrada e saída)")
        
## Definindo quem é x
    x = np.linspace(iniciocomprimento, totalcomprimento, int(numerodepontos))
    
    
## Definindo quem é A
    A = area_bocal (x) 
      
## Retorno e cálculo ponto a ponto
    return x, A



## ============================================================================
## DEFININDO A GEOMETRIA DO BOCAL

def area_bocal(x):
    
    """     Calcula a área do bocal.
            Parâmetro: 
                x: posição ao longo do bocal (m); pode ser um vetor.
            Retorno:
                Área transversal (m²)   """
                
            
## Definindo a área
    A = 1e-4 * (1.5 + 0.5  * 
                np.cos(20* np.pi *x) 
                + 0.5 * np.cos(10 * np.pi *x))

    return A


##=============================================================================
## LOCALIZAÇÃO DA GARGANTA - pela condição dA/dx = 0

def localizar_garganta(x, A):
    """ Calcula a derivada dA/dx numericamente na malha e identifica a posição da garganta (menor área).
        Parâmetros:
            x: vetor de posições (m)
            A: vetor de áreas (m²)

        Retorno:
            x_garganta: posição estimada da garganta (m)
            A_garganta: área na garganta (m²)
            dA_dx: vetor da derivada dA/dx """

    ## Calcular a derivada (ponto a ponto)
    dA_dx = np.gradient(A, x)

    ## Agora: (i) procuramos onde dA/dx troca de sinal de (-) para (+), que é um MÍNIMO de A
    ## (garganta); as pontas ficam de fora porque lá não há troca de sinal;
    ## (ii) interpolamos linearmente o ponto exato em que dA/dx = 0, e (iii) calculamos a área
    ## nesse ponto com a própria função area_bocal.
    trocas = np.where((dA_dx[:-1] < 0) & (dA_dx[1:] >= 0))[0]
    if len(trocas) == 0:
        raise ValueError("não foi encontrada garganta (nenhum mínimo de A na malha)")

    ## Se houver mais de um mínimo local, a garganta é o de menor área
    i = trocas[np.argmin(A[trocas])]

    ## Zero de dA/dx entre os pontos i e i+1 (interpolação linear)
    x_garganta = x[i] - dA_dx[i] * (x[i+1] - x[i]) / (dA_dx[i+1] - dA_dx[i])
    A_garganta = area_bocal(x_garganta)

    return x_garganta, A_garganta, dA_dx


## EXPLICAÇÃO ATÉ AQUI: como precisamos achar onde fica a garganta, usamos a condição dA/dx = 0.
## Como na malha raramente existe um ponto onde a derivada é exatamente zero, procuramos a troca de sinal da derivada e interpolamos entre os dois pontos vizinhos.
## VANTAGEM DE DEFINIR A DERIVADA: se a geometria mudar, não precisamos calcular a derivada "na mão" para outra geometria; o próprio código já faz isso por nós.
##=============================================================================


##=============================================================================
## BLOCO DE TESTE: 

if __name__=="__main__" :
    
    print("TESTE DE GEOMETRIA DO BOCAL - ANALÍTICO")
    print("---------------------------------------")
        
    ## Área nas extremidades - se calcularmos, dará o mesmo valor nas condições dadas
    A_entrada = area_bocal(0.0)
    A_saida = area_bocal(0.1)
        
    print(f"área de entrada: {A_entrada:.6e} [m²]  (esperado 2.5e-4)")
    print(f"área de saída:   {A_saida:.6e} [m²]  (esperado 1.5e-4)")
        
    ## Malha definida e escolhida
    x, A = criar_malha (iniciocomprimento=0.0, totalcomprimento=0.1, numerodepontos=501)
        
    print(f"número de pontos na malha: {len(x)}")
    print(f"comprimento do bocal: {x[-1]:.6e} m")
        
    ## Garganta calculada pela derivada
    x_garganta, A_garganta, dA_dx = localizar_garganta(x, A)

    print("\n--- RESULTADO DA GARGANTA ---")
    print(f"Posição da garganta: {x_garganta:.6f} m")
    print(f"Área da garganta:    {A_garganta:.6e} m²")

    ## Verificação direta da área mínima
    indice_min = np.argmin(A)
        
    print(f"Menor área na malha:  {A[indice_min]:.6e} m²")
    print(f"Posição na malha:     {x[indice_min]:.6f} m")

    ## Verificação analítica: dA/dx = 0  <=>  cos(10πx) = -1/4
    x_analitico = np.arccos(-0.25) / (10 * np.pi)
    print(f"Garganta analítica:   {x_analitico:.6f} m  (A = {area_bocal(x_analitico):.6e} m²)")
