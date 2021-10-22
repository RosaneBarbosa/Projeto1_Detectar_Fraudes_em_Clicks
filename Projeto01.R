# Projeto 1 - Detecção de Fraudes no Tráfego de Cliques 
# em Propagandas de Aplicações Mobile

# Objetivo:
# Em resumo, neste projeto, o objetivo é construir um modelo de aprendizado de máquina
# para prever se um usuário fará o download de um aplicativo depois de clicar em um
# anúncio de aplicativo para dispositivos móveis.. 
#
#
# ***** Descrição das Variáveis *****
# ip -> endereço IP do click.
# app -> ID do aplicativo para o marketing.
# device -> ID do tipo de dispositivo do telefone celular do usuário (por exemplo,
# iphone 6 plus, iphone 7, huawei mate 7 etc.)
# os -> ID da versão do sistema operacional do telefone celular do usuário
# channel -> ID do canal do editor de anúncios para celular
# click_time ->: data/hora do clique (UTC)
# attributed_time -> momento do download do aplicativo pelo usuário
# is_attributed -> variável target, indica se o aplicativo foi baixado


# ****************** Carregando os Dados *******************
# ******* Utilizando o train_sample para as análises *******
# **********************************************************

library(data.table)

treino <- fread("train_sample.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Verificando as dimensões dos dados
dim(treino) 

# Visualizando os dados
View(treino)

# Verificando os tipos dos dados
str(treino)



# **********************************************************
# ************** Pré-processamento dos Dados ***************
# **********************************************************

# Formatando as variáveis de data (time) 

treino$click_time <- as.POSIXct(strptime(
  paste(treino$click_time, " ", ":00:00", sep = ""), 
  "%Y-%m-%d %H:%M:%S"))

treino$attributed_time <- as.POSIXct(strptime(
  paste(treino$attributed_time, " ", ":00:00", sep = ""), 
  "%Y-%m-%d %H:%M:%S"))  


# Verificando a quantidade de valores ausentes (missing) em cada coluna
sapply(treino, function (x) sum(is.na(x)))


# Criando uma nova variável para indicar qual o dia da semana do click
treino$click_dia_semana <- as.factor(weekdays(treino$click_time))

# Verificando a distribuição da nova variável dia da semana do click
table(treino$click_dia_semana)


# Criando uma nova variável para indicar a hora do click
treino$hora_click <- round(as.numeric(hour(treino$click_time)), digits = 0)

# Verificando a distribuição da nova variável hora do click
table(treino$hora_click)


# Convertendo a variável target para o tipo fator (categórica)
treino$is_attributed <- as.factor(treino$is_attributed)



# **********************************************************
# ****** Gráficos para Análise Exploratória de Dados *******
# **********************************************************

# Resumo das medidas de tendência central das variáveis 
summary(treino)

# Vetor com as variáveis ID
colunas_ID <- c("app", "device", "os", "channel")

# Transformando o tipo do dataset em dataframe
class(treino)
treino <- as.data.frame(treino)

library(ggplot2)

# Gráficos
lapply(colunas_ID, function (x) {
    ggplot(treino, aes_string(x)) +
      geom_bar() +
      facet_grid(. ~ is_attributed) +
      ggtitle(paste("Cliques sem/com Download por ", x))
})

# Comentários: 
# De acordo com os gráficos, a frequência de cliques da variável channel assume
# uma maior variedade de valores (não é tão concentrada em determinados valores
# como as demais). 
# Não é possível observar os cliques com download nos gráficos devido a baixíssima 
# representaividade de cliques com download em relação ao total de cliques.


# Vetor com determinados horários
horas <- c(0, 3, 6, 9, 12, 15, 18, 21)


lapply(horas, function(t) {
  ggplot(treino[treino$hora_click == t, ], 
         aes(ordered(click_dia_semana, levels = c("segunda-feira", 
                                                   "terça-feira", 
                                                   "quarta-feira", 
                                                   "quinta-feira", 
                                                   "sexta-feira", 
                                                   "sábado", 
                                                   "domingo")))) +
    geom_bar() +
    facet_grid(. ~ is_attributed) +
    ggtitle(paste("Cliques sem/com Download as ", as.character(t), "h", sep = ""))
})

# Comentário: De acordo com os gráficos, a frequência de cliques diminui às 18h e 21h
# (diferença de fuso horário da China em relação ao Brasil: + 11h)


# Subset dos registros 'is_attributed' igual a 1 para simples verificação
df_attrib <- subset(treino[treino$is_attributed == 1, ])
View(df_attrib)


lapply(colunas_ID, function (x) {
  ggplot(df_attrib, aes_string(x)) +
    geom_bar() +
    facet_grid(. ~ is_attributed) +
    ggtitle(paste("Cliques com Download por ", x))
})


# Verificando a distribuição dos valores da variável target
table(treino$is_attributed)
round(prop.table(table(treino$is_attributed))*100, digits = 1)

# Comentário: A variável target está muito desbalanceada 
# 99,8% de registros para o atributo 0 e apenas 0.2% para o atributo 1

# Visualizando a diferença (gráfico do desbalanceamento)
barplot(prop.table(table(treino$is_attributed)))



# **********************************************************
# ******** Divisão dos Dados em Treino e Validação *********
# **********************************************************

# Criando uma cópia dos dados e aplicando slice para eliminar algumas colunas. 

# Nota: A variável criada 'click_dia_semana' será descartada para o modelo, pois nos
# dados de teste a variável click_time é fixada em 2017-11-10 (único nível: sexta-feira). 
nomes_colunas <- append(colunas_ID, c("ip", "hora_click", "is_attributed"))
dados <- treino[ , nomes_colunas]

dim(dados)

# *NOTA: Como os dados de teste ('test.csv') não incluem a variável target, dividiremos
# os dados em conjuntos de treinamento e de validação a fim de avaliar o modelo.

set.seed(2020)

indice <- sample(1:nrow(dados), 0.8*nrow(dados))

dados_treino <- dados[indice, ]
dados_valida <- dados[-indice, ]

# Verificando as dimensões
dim(dados_treino)
dim(dados_valida)


# Removendo os datasets dados e treino para liberar a memória
rm(dados)
rm(treino)



# ************************************************************************
# *** Aplicando ROSE (Random OverSampling Example) aos dados de treino ***
# ************************************************************************

# Com ROSE conseguimos balancear a variável target usando a técnica de Oversampling, o  
# ideal é sempre aplicar o desbalanceamento após a divisão dos dados em treino e teste.
# Se fizermos antes, o padrão usado para aplicar o oversampling será o mesmo nos dados
# de treino e teste e, assim, a avaliação do modelo fica comprometida. 


library(ROSE)

# Aplicando ROSE em dados de treino e checando a proporção
rose_treino <- ROSE(is_attributed ~ ., data = dados_treino, seed = 1)$data
prop.table(table(rose_treino$is_attributed))

# Visualizando
View(rose_treino)

# Verificando o resumo estatístico após aplicar ROSE
summary(rose_treino)


# Convertendo os valores negativos das variáveis app, device, os, channel, ip e hora
#(gerados pela técnica de Oversampling) para valores absolutos
rose_treino[ , colunas_ID] <- lapply(colunas_ID, function (x) abs(round(rose_treino[ , x], digits = 0)))
rose_treino$hora_click <- abs(round(rose_treino$hora_click, digits = 0))


# Verificando a distribuição da variável hora_click após aplicar ROSE
table(rose_treino$hora_click)
sum(rose_treino$hora_click > 23)

# Zerando os valores da variável hora_click maiores do que 23 gerados pela técnica
# de Oversampling (mantendo assim uma distribuição de frequência similar a observada
# para esta variável originalmente)
for (i in 1:nrow(rose_treino)) {
  if(rose_treino$hora_click[i] > 23) rose_treino$hora_click[i] = 0
}

table(rose_treino$hora_click)



# *************** Seleção de variáveis para o modelo ***************
# ****** Primeiro verificaremos as variáveis mais relevantes ******* 
# ******************************************************************

# Aplicando o modelo randomForest para gerar um plot de importância das variáveis

library(randomForest)

modelo <- randomForest(is_attributed ~ .,
                       data = rose_treino,
                       ntree = 100,
                       nodesize = 10,
                       importance = TRUE)

varImpPlot(modelo)



# ********************** Construção do Modelo **********************
# **** Avaliando 2 Modelos: Random Forest e Regressão Logística ****
# ******************************************************************

# *** Primeiro Modelo: Random Forest

### modelo sem seleção de variáveis

modelo_rf <- randomForest(is_attributed ~ .,
                          data = rose_treino,
                          ntree = 100,
                          nodesize = 10)

print(modelo_rf)


### modelo com seleção de variáveis

modelo_rf2 <- randomForest(is_attributed ~ app + ip + channel + os,
                           data = rose_treino,
                           ntree = 100,
                           nodesize = 10)

print(modelo_rf2)


# Comentário:
# 1) O modelo Random Forest com todas as variáveis apresentou uma taxa de erro 
# menor do que o modelo com a seleção de variáveis.

# 2) Como os dados de teste ('test.csv') não incluem a variável target, as previsões
# para avaliar o modelo serão realizadas com o conjunto de dados que foi separado 
# para a validação do modelo.


# Previsão com os dados de validação
fitted.results_rf <- predict(modelo_rf, dados_valida, type='response')
View(fitted.results_rf)

# Avaliando a performance do modelo com os dados de validação
caret::confusionMatrix(table(fitted.results_rf, reference = dados_valida$is_attributed), positive = '1')

# Calculando o score AUC do modelo de random forest
roc.curve(dados_valida$is_attributed, fitted.results_rf, plotit = TRUE, col = "green")



# *** Segundo Modelo: Regressão Logística

library(caret)

### modelo sem seleção de variáveis

modelo_logReg <- glm(is_attributed ~ .,
                 family = binomial (link = "logit"),
                 data = rose_treino)

summary(modelo_logReg)


### modelo com seleção de variáveis

modelo_logReg2 <- glm(is_attributed ~ app +
                        ip +
                        channel +
                        os,
                     family = binomial (link = "logit"),
                     data = rose_treino)

summary(modelo_logReg2)


# Comentário:
# O AIC (Critério de Informação de Akaike) do modelo de Regressão Logística
# com todas as variáveis resultou menor do que o AIC do modelo com a seleção de
# variáveis, indicando que o modelo sem seleção de variáveis apresentou um melhor
# ajuste aos dados do que o modelo com seleção de variáveis.


# Previsão com os dados de validação

fitted.results_logReg <- predict(modelo_logReg, dados_valida, type='response')
View(fitted.results_logReg)
fitted.results_logReg <- round(fitted.results_logReg)

# Avaliando a performance do modelo com os dados de validação
confusionMatrix(table(fitted.results_logReg, reference = dados_valida$is_attributed), positive = '1')

# Calculando o score AUC para o modelo de Regressão Logística
roc.curve(dados_valida$is_attributed, fitted.results_logReg, plotit = TRUE, col = "blue", add.roc = TRUE)


# Considerações Finais: 
# Entre os dois modelos avaliados verifica-se que o modelo de random forest apresentou
# uma performance um pouco melhor do que a regressão logística

# Resultados obtidos:
# Acurácia Random Forest: 0.8834
# AUC Random Forest: 0.758
# Acurácia Regressão Logística: 0.8192
# AUC Regressão Logística: 0.736



# ******************************************************************
# ****************** Carregando os Dados de Teste ******************
# ******************************************************************

teste <- fread("test.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Visualizando os dados de teste e a dimensão
dim(teste)
View(teste)

# Verificando os tipos dos dados
str(teste)



# ************** Pré-processamento dos Dados de Teste **************

# Formatando a variável de data (click_time) 

teste$click_time <- as.POSIXct(strptime(
  paste(teste$click_time, " ", ":00:00", sep = ""), 
  "%Y-%m-%d %H:%M:%S"))

# Criando a variável hora do click para o arquivo de teste
teste$hora_click <- as.numeric(hour(teste$click_time))

# Verificando a distribuição da variável
table(teste$hora_click)

# Resumo das medidas de tendência central das variáveis 
summary(teste)

# Removendo a variável click_time
teste$click_time = NULL



# ******* Previsão do Modelo Selecionado com Dados de Teste ********
# *** Modelo selecionado: Random Forest sem seleção de variáveis ***
# ******************************************************************

previsao <- predict(modelo_rf, newdata = teste)
df_submission <- data.frame(teste$click_id, previsao)
View(df_submission)

round(prop.table(table(df_submission$previsao))*100, digits = 1)

# Gravando o arquivo de submissão contendo as previsões
write.csv(df_submission, "submission.csv")
