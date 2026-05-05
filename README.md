# AM Heart Failure Prediction
---
# Projeto: Previsão de Insuficiência Cardíaca - Machine Learning

## Descrição do Projeto
Este projeto consiste no desenvolvimento e comparação de modelos de **Machine Learning** para prever a ocorrência de insuficiência cardíaca com base em indicadores clínicos. O trabalho abrange desde a análise exploratória de dados (EDA) até o treinamento e avaliação de múltiplos algoritmos de classificação.

## Contexto e Objetivo

A insuficiência cardíaca é uma condição crítica onde o coração não bombeia sangue de forma eficiente. A detecção precoce através de padrões clínicos pode salvar vidas e otimizar intervenções médicas.

**Objetivo:** Desenvolver um modelo de classificação binária capaz de identificar se um paciente possui ou não doença cardíaca utilizando o dataset *Heart Failure Prediction*.

## 📊 O Conjunto de Dados

O dataset contém **11 variáveis preditoras** e **1 variável alvo (target)**:

**Variáveis (Features)**

- **Idade (Age):** Numérica.

- **Sexo (Sex):** M (Masculino), F (Feminino).

- **Tipo de Dor no Peito (ChestPainType):** TA, ATA, NAP, ASY.

- **Pressão Arterial em Repouso (RestingBP):** mm Hg.

- **Colesterol (Cholesterol):** Nível sérico (mm/dl).

- **Açúcar no Sangue em Jejum (FastingBS):** Binária (1 se > 120 mg/dl).

- **Eletrocardiograma em Repouso (RestingECG):** Normal, ST, LVH.

- **Frequência Cardíaca Máxima (MaxHR):** Batimentos por minuto.

- **Angina Induzida por Exercício (ExerciseAngina):** Binária (1 = Sim).

- **Oldpeak:** Depressão do segmento ST induzida pelo exercício.

- **Inclinação do Pico do Exercício ST (ST_Slope):** Ascendente, Plana, Descendente.

**Alvo (Target)**

- **HeartDisease:** Diagnóstico de insuficiência cardíaca (1: Sim, 0: Não).

## ⚙️ Como Executar
1.  Clone o repositório.
2. Certifique-se de ter as bibliotecas instaladas:

       pip install pandas numpy matplotlib seaborn scikit-learn kagglehub

3. Abra o arquivo `Trabalho_AM_Heart_Failure_Prediction.ipynb` no **Google Colab** ou **Jupyter Notebook**.
4. Execute as células sequencialmente para reproduzir a análise e o treinamento dos modelos.

## 📂 Estrutura dos Arquivos
Os arquivos se encontram na pasta raiz
- `trabalho_am_heart_failure_prediction.py`: Código principal do projeto.
- `heart.csv`: Dataset utilizado no projeto.
- `README.md`: Documentação do projeto.
- `Trabalho_AM_Heart_Failure_Prediction.ipynb`: Notebook com a implementação.
  
## 🛠️ **Tecnologias Utilizadas**

- **Linguagem:** Python 3.x

- **Manipulação de Dados:** `pandas`, `numpy`

- **Visualização:** `matplotlib`, `seaborn`

- **Machine Learning:** `scikit-learn`

- **Dataset:** Obtido via `kagglehub`

## 🚀 Fluxo do Projeto
**1. Análise Exploratória (EDA)**

- **Identificação de Padrões:** Observou-se forte correlação entre a variável alvo e as variáveis `ST_Slope`, `ExerciseAngina` e `MaxHR`.

- **Limpeza:** Detecção de valores zerados em `Cholesterol` e análise de outliers.

- **Visualização:** Gráficos de barras, matrizes de correlação 3D e boxplots para entender a distribuição dos dados.

**2. Pré-processamento**

- **Tratamento de Outliers:** Utilização do método IQR para suavizar anomalias na variável `RestingBP`.

- **Transformação de Dados:** * `StandardScaler` para variáveis numéricas.

- `OneHotEncoder` para variáveis categóricas.

- **Pipeline:** Uso de `ColumnTransformer` e `Pipeline` do scikit-learn para garantir a integridade dos dados entre treino e teste.

**3. Modelagem e Treinamento**

Foram testados e comparados cinco algoritmos diferentes:

1. **Random Forest** (Modelo principal escolhido)

2. **Decision Tree**

3. **KNN** (K-Nearest Neighbors)

4. **Perceptron**

5. **MLP** (Multi-Layer Perceptron)







## 👥 Autores
- **Jefferson Antônio T. Silva**
- **João Paulo Sandes Brito**

**Aviso**: Este projeto tem fins educacionais e de estudo de Machine Learning. Diagnósticos médicos reais devem ser realizados por profissionais de saúde qualificados.
