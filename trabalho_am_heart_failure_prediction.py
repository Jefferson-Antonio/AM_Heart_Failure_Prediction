# Importando os pacaotes que serão utilizados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Se houver mais campos no conjunto de dados, vai aparece o scroll para pode visualizar tudo
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format #Sempre formatar os números em duas casas decimais

# Download do Heart Failure Prediction Dataset
import kagglehub

# Download latest version
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")

print("Path to dataset files:", path)

# Carregando o arquivo CSV
# Caminho para o arquivo CSV (geralmente é esse o nome)
csv_path = os.path.join(path, "heart.csv")

# Carrega os dados com pandas
df_original = pd.read_csv(csv_path)

# Verificando as primeiras linhas do Dataframe
df_original.head()

#df_original['Oldpeak'] = -df_original['Oldpeak']

#order_mapping = {
#    'Y': 0,
#    'N': 1,
#}
#df_original['ExerciseAngina'] = df_original['ExerciseAngina'].map(order_mapping)

#df_original.head()

# Veridicando as dimensões do arquivo.

df_original.shape

# Verificando informações do arquivo

df_original.info()

# Verificando se há valores nulos (dados missing)

df_original.isnull().sum()

# Total de valores únicos de cada variável

valores_unicos = []
for i in df_original.columns[0:12].tolist():
    print(i, ':', len(df_original[i].astype(str).value_counts()))
    valores_unicos.append(len(df_original[i].astype(str).value_counts()))

qtd_zeros = (df_original['FastingBS'] == 0).sum()

print(f"Quantidade de zeros: {qtd_zeros}")

# Visualizando algumas medidas estatísticas.

df_original.describe()

# Quantidade de observações por Sexo
df_original.groupby(['Sex']).size()

# Visualização através do gráfico
plt.rcParams["figure.figsize"] = [6.00, 4.00]
plt.rcParams["figure.autolayout"] = True

df_original.Sex.value_counts().plot(kind='bar', title='Sexo', color = ['#219ebc', '#023047']);

# Quantidade de observações por Tipo de Dor Toráxica
df_original.groupby(['ChestPainType']).size()

# Visualizando através do gráfico
df_original.ChestPainType.value_counts().plot(kind='bar', title='ChestPainType', color = ['#219ebc', '#023047', '#8ecae6', '#d62728']);

# Quantidade de observações por Glicemia em Jejum
df_original.groupby(['FastingBS']).size()

# Visualizando através do gráfico
df_original.FastingBS.value_counts().plot(kind='bar', title='Glicemia em Jejum', color = ['#219ebc', '#023047']);

# Quantidade de observações por tipo de Eletrocardiograma em Repouso
df_original.groupby(['RestingECG']).size()

# Visualizando através do gráfico
df_original.RestingECG.value_counts().plot(kind='bar', title='Eletrocardiograma em Repouso', color = ['#219ebc', '#023047', '#8ecae6']);

# Quantidade de observações por Angina Induzida
df_original.groupby(['ExerciseAngina']).size()

# Visualizando através do gráfico
df_original.ExerciseAngina.value_counts().plot(kind='bar', title='Angina Induzida', color = ['#219ebc', '#023047']);

# Quantidade de observações Inclinação Pico Exercício
df_original.groupby(['ST_Slope']).size()

# Vizualização através do gráfico
df_original.ST_Slope.value_counts().plot(kind='bar', title='Inclinação Pico Exercício', color = ['#219ebc', '#023047', '#8ecae6']);

# Quantidade de observações Doença Cardíaca (Variavel TARGET)
df_original.groupby(['HeartDisease']).size()

df_original.HeartDisease.value_counts().plot(kind='bar', title='HeartDisease', color = ['#219ebc', '#023047']);

# Formatando o tamanho do plot
plt.rcParams["figure.figsize"] = [6.00, 4.00]
plt.rcParams["figure.autolayout"] = True

# Visualizando a Variável Sex x HeartDisease
sns.countplot(data=df_original, x='Sex', hue='HeartDisease');
plt.show()

# Visualizando a variável ChestPainType x HeartDisease
sns.countplot(data=df_original, x='ChestPainType', hue='HeartDisease');
plt.show()

# Visualizando a variável FastingBS x HeartDisease
sns.countplot(data=df_original, x='FastingBS', hue='HeartDisease');
plt.show()

# Visualizando a variável RestingECG x HeartDisease
sns.countplot(data=df_original, x='RestingECG', hue='HeartDisease');
plt.show()

# Visualizando a variável ExerciseAngina x HeartDisease
sns.countplot(data=df_original, x='ExerciseAngina', hue='HeartDisease');
plt.show()

# Visualizando a variável ST_Slope x HeartDisease
sns.countplot(data=df_original, x='ST_Slope', hue='HeartDisease');
plt.show()

# Utilize um catplot do tipo 'strip' para visualizar a distribuição das despesas médicas de acordo com o status de fumante
import seaborn as sns

# Criar catplot do tipo strip
g = sns.catplot(
    data=df_original,
    x='HeartDisease',
    y='MaxHR',
    kind='strip',
    palette=['#3274A1', '#E1812C'],
    height=6,
    aspect=1.2,
    jitter=0.25  # Controla a dispersão dos pontos
)

# Personalizar o gráfico
g.set_axis_labels('Doença Cardiaca', 'Frequência Cardíaca Máxima')
g.fig.suptitle('Distribuição das Doença Cardiaca por Frequência Cardíaca Máxima', y=1.05)
g.set_xticklabels(['Não Doente', 'Doente'])

# Adicionar grid
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Mostrar o gráfico
plt.show()

# Utilize um catplot do tipo 'strip' para visualizar a distribuição das despesas médicas de acordo com o status de fumante
import seaborn as sns

# Criar catplot do tipo strip
g = sns.catplot(
    data=df_original,
    x='HeartDisease',
    y='Oldpeak',
    kind='strip',
    palette=['#3274A1', '#E1812C'],
    height=6,
    aspect=1.2,
    jitter=0.25  # Controla a dispersão dos pontos
)

# Personalizar o gráfico
g.set_axis_labels('Doença Cardiaca', 'Depressão do segmento ST')
g.fig.suptitle('Distribuição das Doença Cardiaca por Depressão do segmento ST', y=1.05)
g.set_xticklabels(['Não Doente', 'Doente'])

# Adicionar grid
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Mostrar o gráfico
plt.show()

from sklearn.preprocessing import LabelEncoder

# Identificar colunas categóricas
categorical_cols = df_original.select_dtypes(include=['object', 'category']).columns
print("Colunas categóricas:", categorical_cols)

# Copiar o DataFrame original
df_encoded = df_original.copy()

# Aplicar LabelEncoder em todas as colunas categóricas
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Garantir que todas as colunas agora são numéricas
df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')

# Remover colunas completamente não numéricas ou inválidas
df_encoded = df_encoded.dropna(axis=1, how='all')

# Gerar matriz de correlação
plt.figure(figsize=(10, 8))
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Mapa de Calor de Correlação entre Variáveis')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Gerar matriz de correlação
correlation_matrix = df_encoded.corr()
labels = correlation_matrix.columns

# Preparar dados para o gráfico 3D
xpos, ypos = np.meshgrid(range(len(labels)), range(len(labels)))
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

# Tamanho das barras
dx = dy = 0.5
dz = correlation_matrix.values.flatten()

# Criar figura 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Definir cores baseadas nos valores de correlação
colors = plt.cm.coolwarm((dz - dz.min()) / (dz.max() - dz.min()))

# Plotar as barras 3D
bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

# Configurar eixos e rótulos
ax.set_xticks(np.arange(len(labels)) + 0.5)
ax.set_yticks(np.arange(len(labels)) + 0.5)
ax.set_xticklabels(labels, rotation=90)
ax.set_yticklabels(labels)
ax.set_zlabel('Correlação')

# Adicionar barra de cores
mappable = plt.cm.ScalarMappable(cmap='coolwarm')
mappable.set_array(dz)
plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, label='Valor de Correlação')

plt.title('Matriz de Correlação em 3D')
plt.tight_layout()
plt.show()

#carregar variáveis para plot
variaveis_numericas = []
for i in df_original.columns[0:11].tolist():
    if df_original.dtypes[i] == 'int64' or df_original.dtypes[i] == 'float64':
        variaveis_numericas.append(i)
variaveis_numericas

Q1 = df_original['RestingBP'].quantile(0.25)
Q3 = df_original['RestingBP'].quantile(0.75)
IQR = Q3 - Q1

# Limites para outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrar outliers
outliers = df_original[(df_original['RestingBP'] < limite_inferior) | (df_original['RestingBP'] > limite_superior)]
print(f"Número de outliers encontrados: {outliers.shape[0]}")

#Podemos observar nos boxplots abaixo que as variáveis númericas apresentam uma grande quantidade de "possíveis" outliers
#Precisamos avaliar cada uma dessas variaveis dentro do contexto dos dados para saber se realmente iremos trata-las como outlier

plt.rcParams["figure.figsize"] = [14.00, 20.00]
plt.rcParams["figure.autolayout"] = True
f, axes = plt.subplots(3, 3) #3 linhas e 2 colunas

linha = 0
coluna = 0
for i in variaveis_numericas:
    sns.boxplot(data = df_original, y=i, ax=axes[linha][coluna])
    coluna += 1
    if coluna == 2:
        linha += 1
        coluna = 0

plt.show()

#Podemos observar nos boxplots que as variáveis númericas apresentam uma grande quantidade de "possíveis" outliers
#Precisamos avaliar cada uma dessas variaveis dentro do contexto dos dados para saber se realmente iremos trata-las como outlier

plt.rcParams["figure.figsize"] = [14.00, 20.00]
plt.rcParams["figure.autolayout"] = True
f, axes = plt.subplots(4, 2) #3 linhas e 2 colunas

linha = 0
coluna = 0
for i in variaveis_numericas:
    sns.histplot(data = df_original, x=i, ax=axes[linha][coluna])
    coluna += 1
    if coluna == 2:
        linha += 1
        coluna = 0

plt.show()

df_tratado = df_original.copy()

Q1 = df_original['RestingBP'].quantile(0.25)
Q3 = df_original['RestingBP'].quantile(0.75)
IQR = Q3 - Q1

# Limites para outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Filtrar outliers
outliers = df_original[(df_original['RestingBP'] < limite_inferior) | (df_original['RestingBP'] > limite_superior)]
print(f"Número de outliers encontrados: {outliers.shape[0]}")

# Substituir outliers por valores como a mediana
mediana = df_original['RestingBP'].median()
df_tratado['RestingBP'] = df_original['RestingBP'].apply(lambda x: mediana if x < limite_inferior or x > limite_superior else x)

# Comparando estatísticas
print("Estatísticas antes do tratamento:")
print(df_original['RestingBP'].describe())

print("\nEstatísticas após o tratamento:")
print(df_tratado['RestingBP'].describe())

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
#from imblearn.over_sampling import SMOTE # usado no balanceamento
from sklearn.metrics import ConfusionMatrixDisplay

# Definir colunas por tipo
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'FastingBS', 'ExerciseAngina', 'ST_Slope']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# Pré-processamento
X = df_tratado.drop('HeartDisease', axis=1)
y = df_tratado['HeartDisease']

# Transformador para pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Criar modelos
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced',  random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Perceptron": Perceptron(eta0=0.1, class_weight='balanced', random_state=42, max_iter=1000, tol=1e-3),
    "MLP": MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu',
                        solver='adam', max_iter=1000, random_state=42)
}

# Treinar e avaliar cada modelo
results = {}
for name, model in models.items():
    # Criar pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Treinar modelo
    pipeline.fit(X_train, y_train)

    # Fazer previsões
    y_pred = pipeline.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Validação cruzada
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

    # Armazenar resultados
    results[name] = {
        'model': pipeline,
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'confusion_matrix': conf_matrix,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores)
    }
    # Imprimir resultados dos diferentes modelos
    print(f"\n{'='*50}")
    print(f"Modelo: {name}")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {results[name]['precision']:.4f}")
    print(f"Recall: {results[name]['recall']:.4f}")
    print(f"F1-Score: {results[name]['f1']:.4f}")
    print(f"Validação Cruzada (5-fold): {results[name]['cv_mean']:.4f} ± {results[name]['cv_std']:.4f}\n")

    # Plotar matriz de confusão
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Doente', 'Doente'],
            yticklabels=['Não Doente', 'Doente'])
    plt.title(f'Matriz de Confusão - {name}', fontsize=14)
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()

# Comparação visual dos modelos
plt.figure(figsize=(12, 8))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
cv_means = [results[name]['cv_mean'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Teste')
plt.bar(x + width/2, cv_means, width, label='Val. Cruzada')
plt.xticks(x, model_names)
plt.ylabel('Acurácia')
plt.title('Comparação de Desempenho dos Modelos')
plt.legend()
plt.ylim(0.7, 0.95)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Reexecutar apenas a validação cruzada para cada modelo
cv_all_scores = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    cv_all_scores[name] = scores

# Transformar em DataFrame para visualização
cv_df = pd.DataFrame(cv_all_scores)

# Plotar boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=cv_df, orient='h', palette='Set2')
plt.title('Validação Cruzada (5-Folds) - Acurácia dos Modelos', fontsize=14)
plt.xlabel('Acurácia')
plt.ylabel('Modelos')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
