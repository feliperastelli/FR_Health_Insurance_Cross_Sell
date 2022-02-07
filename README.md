

### :pushpin: [__Read in English__](https://github.com/feliperastelli/FR-Rossmann_Sales_Model/blob/main/README-en.md)

# Health Insurance - Cross Sell
![image](Images/hp1.png)

O objetivo desse projeto é fornecer para o CFO da Rossmann Drug Stores, um **modelo de previsão de vendas** para as próximas seis semanas para que ele possa definir um orçamento específico para reformas nas lojas. O modelo de previsão atualmente utilizado não atende as necessidades da empresa, portanto, o modelo de machine learning desenvolvido nesse projeto veio como uma solução exata para esse problema de negócio.

O projeto foi desenvolvido através da técnica CRISP-DM, e ao final do primeiro ciclo de desenvolvimento foi possível produzir um modelo de previsão com indíce **MAPE Error de 9%** utilizando o algoritmo **XGBoost**.

Em termos de negócio, o resultado desse modelo de previsão pode ser resumido com os números abaixo:

| __Scenarios__ | __Values__ |
| ------------- | -----------|
| predictions	| US$ 282,662,848.00 |
| worst scenario | US$ 281,907,880.11 |
| best scenario	| US$ 283,417,771.65 |

*O "worst scenario" considera o erro calculado do modelo (MAE) negativamente e o "best scenario", positivamente.

Para visualização do resultado da previsão de cada loja, foi construindo um bot no aplicativo Telegram, onde o usuário pode inserir o número da loja e terá o retorno da previsão calculada pelo modelo que foi colocado em produção no Heroku. Ou seja, foi realizado o deploy em produção do modelo e bot para que possam ser acessados de qualquer lugar.

Para acessar basta apenas ter o aplicativo instalado no smartphone ou PC, criar uma conta, e solicitar para o contato Bot o número da loja. Ex: '/22', '/50'. Faça o teste:

[<img alt="Telegram" src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"/>]( http://t.me/vs_rossmannbot)

## 1. Sobre a Insurance All

### 1.1 Contexto do negócio:

A Insurance All é uma empresa fictícia que fornece como produto principal, seguro de saúde para seus clientes. Nesse contexto, time de produtos está analisando a possibilidade de oferecer aos segurados, um novo produto: Um seguro de automóveis.Assim como o seguro de saúde, os clientes desse novo plano de seguro de automóveis precisam pagar um valor anualmente à Insurance All para obter um valor assegurado pela empresa, destinado aos custos de um eventual acidente ou dano ao veículo.

A Insurance All fez uma pesquisa com cerca de 380 mil clientes sobre o interesse em aderir a um novo produto de seguro de automóveis, no ano passado. Todos os clientes demonstraram interesse ou não em adquirir o seguro de automóvel e essas respostas ficaram salvas em um banco de dados junto com outros atributos dos clientes.

O time de produtos selecionou 127 mil novos clientes que não responderam a pesquisa para participar de uma campanha, no qual receberão a oferta do novo produto de seguro de automóveis. A oferta será feita pelo time de vendas através de ligações telefônicas.

### 1.2 Questão do negócio:

Dado o contexto acima, sabe-se que time de vendas tem uma capacidade de realizar 20 mil ligações dentro do período da campanha. Ou seja, para realizar essa campanha, a empresa possui recursos limitados e precisa alcançar com prioridade, às pessoas que de supostamente estarão interessadas no novo produto.

O time de negócios definiu as seguintes questões à serem avaliadas e respondidas dentro desse projeto:

- Principais Insights sobre os atributos mais relevantes de clientes interessados em adquirir um seguro de automóvel.
- Qual a porcentagem de clientes interessados em adquirir um seguro de automóvel, o time de vendas conseguirá contatar fazendo 20.000 ligações?
- E se a capacidade do time de vendas aumentar para 40.000 ligações, qual a porcentagem de clientes interessados em adquirir um seguro de automóvel o time de vendas conseguirá contatar? 
- Quantas ligações o time de vendas precisa fazer para contatar 80% dos clientes interessados em adquirir um seguro de automóvel? 

### 1.3 Sobre os dados:

Os dados foram disponibilizados pela empresa na plataforma do Kaggle: https://www.kaggle.com/c/rossmann-store-sales/data

|***Atributo*** | ***Descrição*** |
| -------- | --------- |
|**Id** | identificador único do cliente |
|**Gender** | gênero do cliente |
|**Age** | idade do cliente |
|**Driving License** |  0, o cliente não tem permissão para dirigir e 1, o cliente tem para dirigir ( CNH – Carteira Nacional de Habilitação ) |
|**Region Code** | código da região do cliente |
|**Previously Insured** |  0, o cliente não tem seguro de automóvel e 1, o cliente já tem seguro de automóvel |
|**Vehicle Age** |  idade do veículo |
|**Vehicle Damage** | 0, cliente nunca teve seu veículo danificado no passado e 1, cliente já teve seu veículo danificado no passado | 
|**Anual Premium** | quantidade que o cliente pagou à empresa pelo seguro de saúde anual | 
|**Policy sales channel** |  código anônimo para o canal de contato com o cliente | 
|**Vintage** | número de dias que o cliente se associou à empresa através da compra do seguro de saúde | 
|**Response** | 0, o cliente não tem interesse e 1, o cliente tem interesse | 

### 1.4 Premissas do negócio:

- Os dados originais, com a variável resposta, foram utilizados para treino e teste.
- Para analisar e responder as questões propostas, foram utilizados os dados de produção (127k).

## 2. Planejamento da solução:

O projeto foi desenvolvido através do método CRISP-DM, aplicando os seguintes passos:

**Passo 01 - Descrição dos dados:** Nessa etapa, o objetivo foi conhecer os dados, seus tipos, usar métricas estatísticas para identificar outliers no escopo do negócio e também analisar métricas estatísticas básicas como: média, mediana, máximo, mínimo, range, skew, kurtosis e desvio padrão. Nessa etapa foi observado um grande desbalanceamento dos dados, mas não foi aplicado o balanceamento nesse ciclo do projeto.

**Passo 02 - Feature Engineering:** Nessa etapa, foi desenvolvido um mapa mental para analisar o fenômeno, suas variáveis e os principais aspectos que impactam cada variável. Nesse ciclo, não foi realizada nenhuma derivação de features, apenas a alteração de algumas.

**Passo 03 - Filtragem dos dados:** Não houve necessidade de realizar filtragem dos dados.

**Passo 04 - Análise Exploratória dos dados:** O objetivo desta etapa foi explorar os dados para encontrar insights, entender melhor a relevância das variáveis no aprendizado do modelo. Foram feitas analises univariadas, biváriadas e multivariadas, utilizandos os dados numéricos e categóricos do conjunto.

**Passo 05 - Preparação dos dados:** Nessa etapa,  os dados foram preparados para o inicio das aplicações de modelos de machine learning. Foram utilizadas técnicas como Standardization, Rescaling e Encoder, para reescalar e padronizar algumas features.

**Passo 06 - Seleção de Features:** O objetivo desta etapa foi selecionar os melhores atributos para treinar o modelo. Foram utilizados o algoritmo Boruta e a técnica de Features Importance, para fazer a seleção das variáveis, destacando as que tinham maior relevância para o fenômeno.

**Passo 07 - Modelagem de Machine Learning:** Nessa etapa foram feitos os testes e treinamento de alguns modelos de machine learning, onde foi possível comparar suas respectivas performances e feita a escolha do modelo ideal para o projeto. Para todos, foi utilizada também a técnica de Cross Validation para garantir a performance real sobre os dados selecionados.

**Passo 08 - Hyperparameter Fine Tunning e Modelo Final:** Tendo a escolha do algorotimo LightGBM na etapa anterior, foi feita uma randomização para escolher os melhores valores para cada um dos parâmetros do modelo. Após isso, os dados de treino e validação foram unificados para treinamento do modelo final, que foi avaliado sob os dados de teste para verificar o poder de generalização.

**Passo 09 - Performance de Negócio:** O objetivo dessa etapa foi de fato demonstrar o resultado do projeto, aplicando o modelo treinado sobre os dados de produção. Os objetivos finais do projeto foram então desenvolvidos, onde as questões do negócio foram analisadas e respondidas.

**Passo 10 - Deploy do modelo em produção:** Após execução bem sucedida do modelo, o objetivo foi publica-lo em um ambiente de nuvem para que outras pessoas ou serviços possam acessá-lo. A plataforma para hospedagem em nuvem escolhida foi o Heroku.

**Passo 11 - Planilha Google:** Como um extra, foi desenvolvida uma planilha na plataforma do Google, que permite ao usuário listar os novos clientes, e ao solicitar a predição, a planilha utilizará do modelo em produção e fará o ranking dos clientes através do seu resultado de "score".

## 3. Principais insights:

**Hipótese 1:** A maioria dos clientes com carros mais novos já tem seguro.
  **Verdadeira:** Clientes que possuem carros mais novos, já possuem seguro veicular, logo não teriam interesse em um novo produto.

![image](Images/hp1.png)

**Hipótese 2:** Clientes que pagam mais pelo seguro de saúde anualmente estão menos interessados em comprar outro.
  **Verdadeira:** Há uma concentração maior de clientes interessados dentro dos que pagam quantias menores de seguro de saúde.

![image](Images/hp2.png)

**Hipótese 3:** Clientes com veículos já danificados, sem sua maioria, já possuem seguro.
  **Falsa:** A maior parte dos clientes que já sofreram danos em seus veículos, não possuem seguro de automóvel.
  
  ![image](Images/hp3.png)

*Demais insights podem ser consultados no notebook do projeto.*

## 4. Performance dos Modelos de Machine Learning:

Se tratando de um problema de *Learning to Rank*, foram utilizados algoritmos de classificação para calcular a cada cliente, sua propensão em aceitar o novo produto. Com os valores de "score" determinados, o objetivo foi ordená-los do maior para o menor, avaliando a capacidade que o Modelo teve de ranquear as maiores propensões no topo. Para esse projeto, foram selecionados os seguintes algorítmos: 

**- Modelos Utilizados:**

   - K Neighbors Classifier - KNN
   - Logistic Regression 
   - Random Forest Classifier
   - Gradient Boosting Classifier - XGBoost
   - Light Gradient Boosting Machine Classifier - LGBM

**Comparação da performance dos modelos:**

***Model Name*** | ***MAE CV*** | ***MAPE CV*** | ***RMSE CV*** |
| ---------------- | ---------- | --------- | ---------- |
|Random Forest Regressor | 842.56 +/- 220.07 | 0.12 +/- 0.02	 | 1264.33 +/- 323.29 |
|XGBoost Regressor | 1048.45 +/- 172.04 | 0.14 +/- 0.02	 | 1513.27 +/- 234.33 |
|Average Model | 1354.80 | 0.45	 | 1835.13 |
|Linear Regression | 2081.73 +/- 295.63 | 0.3 +/- 0.02	 | 2952.52 +/- 468.37 |
|Lasso | 2116.38 +/- 341.5 | 0.29 +/- 0.01	 | 3057.75 +/- 504.26 |

**Performance final do modelo escolhido após Hyperparameter Fine Tuning:**

***Model Name*** | ***MAE*** | ***MAPE*** | ***RMSE*** |
| -------- | --------- | --------- | --------- |
|XGBoost Regressor | 673.394631 | 0.097298	 | 965.731681 |

## 5. Resultado final - Model performance vs Business Values

O resultado final do projeto foi satisfatório para a maior parte das lojas abrangidas nos dados, conforme gráfico abaixo (Essas lojas em específico podem conter particularidades e possivelmente num segundo ciclo desse projeto, algo poderia ser feito para melhor a performance e predição para elas).

![image](https://user-images.githubusercontent.com/77105763/143149982-0e6c1f18-3874-412a-a82f-01ff03b13c85.png)

A maior parte das lojas tiveram o erro MAPE muito próximo do erro performado no modelo - **MAPE Error de 9%**

Como indicado no resumo prévio do projeto, o resultado que pode ser obtido utilizando-se do modelo, considerando o melhor e pior cenário, é o seguinte:

| __Scenarios__ | __Values__ |
| ------------- | -----------|
| predictions	| US$ 282,662,848.00 |
| worst scenario | US$ 281,907,880.11 |
| best scenario	| US$ 283,417,771.65 |



Podemos observar o performance do modelo, avaliando a relação entre as vendas (dados de teste) e as predições:

![image](https://user-images.githubusercontent.com/77105763/143151060-c9ef9bcd-a266-4a1a-9457-e99a203d77d6.png)

## 6. Conclusão

O projeto desenvolvido foi concluído com êxito, onde foi possível projetar as vendas das próximas semanas para que o CFO tenha informações reais para criar o budget das lojas, podendo consultar em tempo real cada predição.

- O deploy do modelo desenvolvido e da aplicação do Bot do Telegram foram construídos no ambiente em nuvem do **Heroku** e estão em funcionamento.

- Toda documentação do projeto pode ser consultada no repositório, incluindo os notebooks desenvolvidos e todos os scritps finais para as aplicações web.
