## iniciando o documento com permição para uso de caractéres do portugues acentos...
# -*- coding: utf-8 -*-

## definnindo nome do documento
""" geometria.py """


## importando bibliotecas a serem usadas
""" bibliotecas usadas """
import numpy as np


## ============================================================================
## DEFININDO AS VARIAVEIS - discretização do comprimento do bocal e area 
""" definição das variaveis (x) """

def criar_malha (iniciocomprimento=0.0, totalcomprimento=0.1, numerodepontos=501 ):
    
    """ Criando os pontos ao longo do bocal.

    Parâmetros:
        comprimento: comprimento do bocal(m);
        numerodepontos: quantidade de pontos da malha.
 
    Retorno:
        x: vetor de posições (m);
        A: vetor de áreas (m²). """

## certificando o funcionamento da geometria: valores maiores q 0
    if totalcomprimento<=0 : 
    
        raise ValueError ("totalcomprimento deve ser positivo")

    if numerodepontos<2 or int(numerodepontos) != numerodepontos:
    
        raise ValueError ("pelo menos 2 pontos (entrada e saida)")
        
##definindo quem é x
    x = np.linspace(iniciocomprimento, totalcomprimento, int(numerodepontos))
    
    
##definindo quem é A
    A = area_bocal (x) 
      
## retorno e cálculo ponto a ponto
    return x, A


## EXPLICAÇÃO ATE AQUI: para conseguir analizar um comprimento é necessário no mínimo 2 pontos; e esses pontos devem ser valores inteiros, ja que o ponto ex:499,9 não é bem um ponto. Ao aplicarmos o "retur" podemos calcular a área dado os pontos.
## ============================================================================


## ============================================================================
## DEFININDO GEOMETRIA DO BOCAL; após as variaveis ja estarem claras.

def area_bocal(x):
    
    """     Calcular a área do bocal.
            Parametro: 
                x: posição ao longo do bocal (m); pode ser um vetor.
            Retorno:
                Área transversal (m²)   """
                
            
## definindo a área
    A = 10e-4 * (1.5 + 0.5  * 
                np.cos(20* np.pi *x) 
                + 0.5 * np.cos(10 * np.pi *x))

    return A


##=============================================================================
## LOCALIZAÇÃO ANALÍTICA DA GARGANTA - pela condição dA/dx = 0

def localizar_garganta(x, A):
    """ Calcula a derivada dA/dx numericamente na malha e identifica a posição da garganta (menor área).
        Parâmetros:
            x: vetor de posições (m)
            A: vetor de áreas (m²)

        Retorno:
            x_garganta: posição estimada da garganta (m)
            A_garganta: área na garganta (m²)
            dA_dx: vetor da derivada dA/dx """

    ## calcular a derivada (ponto por ponto)
    dA_dx = np.gradient(A, x)

    ## calcular onde a derivada é zero (ou mais próximo)
    derivada_zero = np.argmin(np.abs(dA_dx))

    x_garganta = x[derivada_zero]
    A_garganta = A[derivada_zero]

    return x_garganta, A_garganta, dA_dx


##EXPLICAÇÃO ATE AQUI: como precisamos analizar analiticamente onde a onda de choque se encontra; usamos a condição da derivada ser igual a zero, como nem sempre será um valor absoluto, podemos deixar o mais proximo de zero que conseguirmos; e é isso que as bibliotecas  ##EXPLICAÇÃO ATE AQUI: como precisamos analizar analiticamente onde a onda de choque se encontra; usamos a condição da derivada ser igual a zero, como nem sempre será um valor absoluto, podemos deixar o mais proximo de zero que conseguirmos; e é isso que as bibliotecas argmin(menor valor) e abs ("modulo") fazem para encontrar o valor mais perto de zero
## VANTAGEM DE DEFINIR A DERIVADA:  se caso a geometria mudar, não precisamos fazer a derivada "na mão" para outra geometria, o próprio código ja faz isso por nós.
##=============================================================================


##=============================================================================
## BLOCO DE TESTE: 
if __name__=="__main__" :
    
    print("TESTE DE GEOMETRIA BOCAL - ANALITICO")
    print("------------------------------------")
        
## área nas extremidades (ja foram pré- definidas pelo professor) - se calcularmos dará o mesmmo valor na condições dadas
    
A_entrada = area_bocal(0.0)
A_saida = area_bocal(0.1)
    
print(f"area de entrada: {A_entrada:.6e} [m²]")
print(f"area de saida:   {A_saida:.6e} [m²]")
    
## malha definida - escolhida
x, A = criar_malha (iniciocomprimento=0.0, totalcomprimento=0.1, numerodepontos=501)
    
print(f"numero de pontos na malha: {len(x)}")
print(f"comprimento do bocal: {x[-1]:.6e} m")
    
## garganta calculada pela derivada
x_garganta, A_garganta, dA_dx = localizar_garganta(x, A)

print("\n--- RESULTADO DA GARGANTA  ---")
print(f"Posição da garganta: {x_garganta:.6f} m")
print(f"Área da garganta:    {A_garganta:.6e} m²")

## 4. Verificação direta da área mínima
derivada_zero = np.argmin(A)
    
print(f"Menor área na malha:  {A[derivada_zero]:.6e} m²")
print(f"Posição na malha:     {x[derivada_zero]:.6f} m")















          
