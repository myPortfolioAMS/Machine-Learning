# check kaggle for charts
#https://www.kaggle.com/code/mg22024/notebook89c5ff0b25/notebook


#Data Treatment 
import numpy as np
import pandas as pd
import warnings

# Vizualization
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

#Hypothesis Tests
from scipy.stats import ranksums
from scipy.stats import normaltest
from statsmodels.stats.weightstats import ztest

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Stratification strategy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold

# Assessment metrics
from sklearn.metrics import accuracy_score
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# Customizing views
matplotlib.rcParams['figure.figsize'] = [14, 8]
sns.set_theme(style='whitegrid')
warnings.simplefilter(action='ignore', category=FutureWarning)


def personaliza_grafico(titulo:str, x_label: str, y_label: str):
  '''
  Defines the descriptive information of the chart

  Keywords arguments
  titulo: Chart Title
  x_label: Description of the X-axis of the graph
  y_label: Description of the Y-axis of the graph
  '''
  plt.title(titulo, loc='left', fontsize=24)
  plt.xlabel(x_label, fontsize=18)
  plt.ylabel(y_label, fontsize=18)
  

def tabela_frequencia(dados: pd.DataFrame, coluna: str, titulo: str):
  '''
  Calculate data series frequency table

  Keywords arguments
  dados: DataFrame with data for calculation
  coluna: Dataframe columns with desired data
  titulo: Descriptive title of the axis/table

  return: DataFrame with frequency and percentage information
  '''
  frequencia = dados[coluna].value_counts(ascending=False)
  percentual = round(dados[coluna].value_counts(normalize=True, ascending=False) * 100, 1)
  _ = pd.DataFrame(
      {'Frequencia': frequencia,
      'Percentual %': percentual})
  return _.rename_axis(titulo)


def grafico_pizza(dados: pd.DataFrame, valores: str, nomes: str, titulo: str):
  '''
  Plots pizza charts

  Keywords arguments
  dados: DataFrame with data for calculation
  valores: Coluna do DataFrame with the values in series
  nomes: Variables descriptions
  titulo: Chart descriptive title
  '''
  px.pie(dados, values=valores, names=nomes,
       title=titulo)


def preenche_tabela(dados):
  '''
  functions fills lines with NaN values with the lines antecedents and subsequent ones referring to the same patients considering the line
   that ICU is equal to 1


  Keywords arguments:
  dados: DataFrame with data for calculations

  return: Columns filled with data from the preceding or subsequent rows of the NaN value
  '''
  features_continuas_colunas = dados.iloc[:, 13:-2].columns
  features_continuas = dados.query('ICU == 0').groupby("PATIENT_VISIT_IDENTIFIER", as_index=False)[features_continuas_colunas].fillna(method='bfill').fillna(method='ffill')
  features_categoricas = dados.iloc[:, :13]
  saida = dados.iloc[:, -2:]
  dados_finais = pd.concat([features_categoricas, features_continuas, saida], ignore_index=True,axis=1)
  dados_finais.columns = dados.columns
  return dados_finais


def prepare_window(rows):
  '''
  function scans rows and identify in which row the ICU (UTI) column is equal to 1, and assign that value to the line WINDOW = 0-2
  '''
  if np.any(rows['ICU']):
    rows.loc[rows['WINDOW'] == '0-2', 'ICU'] = 1
  return rows.loc[rows['WINDOW'] == '0-2']


def selecao_janela_1(dados: pd.DataFrame,
                    identificacao_paciente='PATIENT_VISIT_IDENTIFIER',
                    janela='WINDOW'):
  '''
    Excludes patients with ICU = 1 and WINDOW = 0-2, assigns the value 1 to the first window of patients who went to the ICU in subsequent windows, and excludes those unnecessary columns and indexes.

  Keywords arguments
  dados: DataFrame with NaN values filled
  identificacao_paciente: Column that identifies the patient (visitor)
  janela: Column that identifies the window for collecting information about the patient

  return: fist window (WINDOW == 0-2), assigning the value 1 to patients who went to the ICU in subsequent windows
  '''
  dados.drop(dados.query('ICU == 1 and WINDOW == "0-2"')['PATIENT_VISIT_IDENTIFIER'].values,
             inplace=True)
  dados = dados.groupby(identificacao_paciente, as_index=False).apply(prepare_window)
  dados.drop([identificacao_paciente, janela], inplace=True, axis=1)
  dados = dados.droplevel(1)
  return dados


def teste_normalidade(dados: pd.DataFrame, coluna: pd.Series, descricao_coluna: str):
  '''
  Returns outcome of Normality Test

  Keywords arguments
  dados: DataFrame with data for testing
  coluna: Testing DataFrame column 
  descricao_coluna: Description of the variable being tested
  '''
  _, p_value = normaltest(dados[coluna],
                        nan_policy='omit')
  if p_value < 0.05:
    print(f'''
P-value of the variable's normality test {descricao_coluna}: {p_value:.2f}
P-value < 0.05 = {p_value < 0.05}
This is NOT a normal distribution.
''')
  else:
    print(f'''
P-value of the variable's normality test {descricao_coluna}: {p_value:.2f}
P-value < 0.05 = {p_value < 0.05}
This is a normal distribution.
''')

    
def teste_ranksums(dados: pd.Series, coluna: str, descricao_coluna):
  '''
  Rank sums test of the group that went and did not go to the ICU

  Keywords arguments:
  dados: DataFrarme with patient data that went and did not go to the ICU
  coluna: Testing DataFrame columns
  descricao_coluna: Description of the variable being tested
  '''
  _, p_value = ranksums(dados.query('ICU == 0')[coluna],
         dados.query('ICU == 1')[coluna])

  if p_value < 0.05:
    print(f'''
Column: {descricao_coluna}
P-value: {p_value:.2f}
P-value < 0.05 = {p_value < 0.05}
Result: Same Distribution = NO.
''')
  else:
    print(f'''
Column: {descricao_coluna}
P-value: {p_value:.2f}
P-value < 0.05 = {p_value < 0.05}
Result: Same Distribution = YES.
''')

    
def altera_valores_index(dados: pd.DataFrame, antigo_index: str,
                         dicionario_valores: dict):
  '''
  Altera os valores do index para um formato descritivo

  Keyword arguments
  dados: DataFrame com os dados para alteração do index
  antigo_index: Coluna com os valores do antigo index
  dicionario_valores: Dicionário com o mapeamento dos valores

  return: Novo DataFrame com o index descritivo
  '''
  dados.reset_index(inplace=True)
  dados[antigo_index] = _[antigo_index].map(dicionario_valores)
  _.set_index(antigo_index, inplace=True)
  return _


def teste_ranksums_colunas(dados: pd.DataFrame, colunas: list,
                           variacao_colunas_1=False, variacao_colunas_2=False,
                           distribuicoes='Foi para UTI x Não foi para UTI'):
  '''
  Realiza o teste Ranksums em um bloco de colunas

  Keywords arguments
  dados: DataFrame com os dados para teste
  colunas: Lista com parte da descrição da coluna que se repete nas demais
  colunas
  variacao_colunas_1: Variação em cada descrição geral das colunas
  referente aos sinais vitais
  variacao_colunas_2: Descrição geral das colunas referente
  aos exames de sangue
  distribuições: Descrição das distribuições que estão sendo testadas
  '''
  print('                    TESTES RANKSUMS')
  print('')
  print(distribuicoes)
  print('')
  if variacao_colunas_1:
    for i in colunas:
      coluna = i + '_' + variacao_colunas_1
      teste_ranksums(dados, coluna, coluna)
  elif variacao_colunas_2:
    for i in colunas:
      coluna = variacao_colunas_2 + '_' +  i
      teste_ranksums(dados, coluna, coluna)

      
def mesma_distribuicao_colunas(dados: pd.DataFrame):
  '''
  Realiza o teste de normalidade e se a distribuição for ou não normal a função
  verifica a iguldade da distribuição referente aos pacientes que foram ou não
  para UTI.

  Keywords arguments
  dados: DataFrame com os dados para teste

  return: Lista com as colunas que possuem a mesma distribuição referente
  aos pacintes que foram e não foram para UTI
  '''
  mesma_distribuicao = list()
  for i in dados.columns:
    _, p_value = normaltest(dados[i])
    # Se não for uma distribuição normal
    if p_value < 0.05:
      _, p_value = ranksums(dados.query('ICU == 0')[i], dados.query('ICU == 1')[i])
      # Se as distribuições forem iguais
      if p_value >= 0.05:
        mesma_distribuicao.append(i)
    # Se for uma distribuição normal
    elif p_value > 0.05:
      _, p_value = ztest(dados.query('ICU == 0')[i], dados.query('ICU == 1')[i])
      # Se as distribuições forem iguais
      if p_value >= 0.05:
        mesma_distribuicao.append(i)
  return mesma_distribuicao 


def roda_modelo_cv(dados: pd.DataFrame, y='ICU',
                x_drop=['ICU'],
                modelo=DecisionTreeClassifier(), n_splits=5, n_repeats=10,
                descricao_modelo='Decision Tree Classifier'):
  '''
  Realiza a validação cruzada

  Keyword arguments
  dados: DataFrame com as informações para treino e teste
  y: Coluna do DataFrame atribuída a variável y
  x_drop: Colunas do DataFrame que devem ser desconsidradas na variável x
  modelo: Modelo para classificação
  n_splits: Número de partes que o DataFrame será divido para validação
  n_repeats: Número de vezes que o modelo será treinado
  descricao_modelo: Descrição do modelo utilizado

  return: Média do auc de teste e de treino
  '''

  np.random.seed(2933)
  dados = dados.sample(frac=1).reset_index(drop=True)
  y = dados[y]
  x = dados.drop(x_drop, axis=1)

  modelo = modelo

  cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
  resultado = cross_validate(modelo, x, y, cv=cv, scoring='roc_auc',
                             return_train_score=True)

  media_teste = np.mean(resultado['test_score'])
  media_treino = np.mean(resultado['train_score'])

  print(f'AUC teste/ treino: {media_teste:.2f} - {media_treino:.2f}')
  return media_teste, media_treino


def curva_auc(media_teste: list(), media_treino: list(), descricao_modelo: str,
              n_repeats_x=10):
  '''
  Plota o gráfico com o resultado de AUC de treino e teste

  Keyword arguments
  n_repeats_x: Número de repetições do modelo
  media_treino: Resultado médio do AUC de treino
  descricao_modelo: Descrição do modelo utilizado
  '''
  x = range(1, n_repeats_x)
  plt.figure(figsize=(16, 8))
  plt.plot(x, media_teste, label='AUC Teste', )
  plt.plot(x, media_treino, label='AUC Treino')
  plt.title(descricao_modelo, loc='left', fontsize=24)
  plt.legend();
  
  
def modelo_treino_teste(dados: pd.DataFrame, y='ICU', x_drop=['ICU'],
                        modelo=RandomForestClassifier()):
  '''
  Realiza a estratificação dos dados de treino e teste e retorna o modelo
  ajusdado aos dados de treino

  Keyword arguments
  dados: DataFrame com as informações para treino e teste
  y: Coluna do DataFrame atribuída a variável y
  x_drop: Colunas do DataFrame que devem ser desconsidradas na variável x
  modelo: Modelo para classificação

  return: Modelo de ML treinado, x de treino e teste, y de treino e y previsto
  '''

  seed = np.random.seed(2933)
  dados = dados.sample(frac=1).reset_index(drop=True)
  y = dados[y]
  x = dados.drop(x_drop, axis=1)

  x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=.3,
                                                      random_state=seed,
                                                      stratify=y)
  modelo.fit(x_treino, y_treino)
  y_predito = modelo.predict(x_teste)

  return modelo, x_treino, x_teste, y_treino, y_teste, y_predito


# Reading and cleaning Data


dados = pd.read_excel('./Data/Dataset.xlsx', thousands='.', decimal=',')
dados.head()

'''
dados_limpos: Informações disponibilizadas pelo hospital Sírio Libanês com os
seguintes tratamentos;

1. Preenchimento dos valores NaN considerando as linhas antecedentes e
subsequentes do histórico do pacinete (visitante), as informações das linhas com
ICU = 1 não foram utilizadas;
2. Seleção da primeira janela (WINDOW == 0-2) e atribuição do valor 1 aos
pacientes que foram para UTI nas janelas subsequentes
3. Codificação da coluna AGE_PERCENTIL
4. Exclusão das linhas com valores NaN
'''
dados_limpos = preenche_tabela(dados)
dados_limpos = selecao_janela_1(dados_limpos)
dados_limpos.dropna(inplace=True)
dados_limpos.head()


# Exploratory Analysis

'''
Linhas: Número de pacientes (visitantes)
Colunas: Informações sobre os pacientes (visitantes)
'''
print(f'Número de linhas: {dados_limpos.shape[0]}')
print(f'Número de colunas: {dados_limpos.shape[1]}')

tabela_frequencia(dados_limpos, 'ICU', 'UTI')



_ = pd.DataFrame({'Pacientes': [161, 189]}, index=['Foram para a UTI', 'Não forma para a UTI'])
px.pie(_, values='Pacientes', names=_.index,
       title='Pacientes(visitantes) que foram ou não para a UTI')

# Demographic Data


# Colunas referente as informações demográficas do paciente
dados.columns[1:4]

'''
dados_demograficos: Contêm as informações demográficas dos pacientes
'''
# Seleção das colunas
dados_demograficos = dados_limpos[['AGE_ABOVE65',
                            'AGE_PERCENTIL', 'GENDER', 'ICU']]
dados_demograficos.head()

'''
nao_sim: Dicionário com o significado dos valores 0 e 1
'''
nao_sim = {
    0: 'Não',
    1: 'Sim'
}


_ = tabela_frequencia(dados_demograficos, 'AGE_ABOVE65',
                      'Idade superior a 65 anos')
altera_valores_index(_, 'Idade superior a 65 anos', nao_sim)


_ = pd.DataFrame({
    'Foi para UTI': round(dados_demograficos.query('ICU == 1')['AGE_ABOVE65'].value_counts(normalize=True) * 100, 1),
    'Não foi para UTI': round(dados_demograficos.query('ICU == 0')['AGE_ABOVE65'].value_counts(normalize=True) * 100, 1)
})
_.rename_axis('Idade superior a 65 anos', inplace=True)
altera_valores_index(_, 'Idade superior a 65 anos', nao_sim)

'''
Variável: Formato categórico da coluna AGE_PERCENTIL
'''
percentil_idade = {
    '10th': 0,
    '20th': 1,
    '30th': 2,
    '40th': 3,
    '50th': 4,
    '60th': 5,
    '70th': 6,
    '80th': 7,
    '90th': 8,
    'Above 90th': 9}


tabela_frequencia(dados_demograficos, 'AGE_PERCENTIL', 'Percentil das idades')

print(f'''
Média do número de pacientes por percentil de idade: {round(dados_demograficos['AGE_PERCENTIL'].value_counts().values.mean())}
Desvio padrão do número de pacientes por percentil das idades: {round(dados_demograficos['AGE_PERCENTIL'].value_counts().values.std())}
''')


plt.hist(x=dados_demograficos.query('ICU == 0')['AGE_PERCENTIL'].map(percentil_idade))
personaliza_grafico('Percentil das idades do grupo que não foi para UTI',
                    'Percentil', 'Frequência')


plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['10th', '20th', '30th', '40th', '50th',
                                            '60th', '70th', '80th', '90th',
                                            'Above 90th']);


tabela_frequencia(dados_demograficos.query('ICU == 0'), 'AGE_PERCENTIL',
                  'Percentil da idade/ não foi para UTI')


plt.hist(x=dados_demograficos.query('ICU == 1')['AGE_PERCENTIL'].map(percentil_idade))
personaliza_grafico('Percentil das idades do grupo que foi para UTI',
                    'Percentil', 'Frequência');


plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['10th', '20th', '30th', '40th', '50th',
                                            '60th', '70th', '80th', '90th',
                                            'Above 90th']);


tabela_frequencia(dados_demograficos.query('ICU == 1'), 'AGE_PERCENTIL',
                  'Percentil idade/ foi para UTI')


_ = ranksums(dados_demograficos.query('ICU == 1')['AGE_PERCENTIL'],
             dados_demograficos.query('ICU == 0')['AGE_PERCENTIL'])
print(f'''
P-value: {_[1]:.2f}
P-value < 0.05 = {_[1] < 0.05}
P-value menor do que 0.05 rejeita a hitótese nula e confirma a hipótese alternativa,
como já observado graficamente as distribuições referente ao percentil das idades
são diferentes.''')


tabela_frequencia(dados_limpos, 'GENDER', 'Gênero')


tabela_frequencia(dados_limpos.query('ICU == 0'), 'GENDER',
                  'Não foram para UTI por gênero')

tabela_frequencia(dados_demograficos.query('ICU == 1'), 'GENDER', 'Foram para UTI por gênero')


# Colunas referente aos grupos prévios de doênças do paciente
dados.columns[4:13]


'''
dados_doencas: Dados referente ao grupo de doêncas dos pacientes
'''

dados_doencas = dados_limpos[['DISEASE GROUPING 1', 'DISEASE GROUPING 2',
                             'DISEASE GROUPING 3', 'DISEASE GROUPING 4',
                             'DISEASE GROUPING 5', 'DISEASE GROUPING 6', 'HTN',
                             'IMMUNOCOMPROMISED', 'OTHER', 'ICU']]


dados_doencas.head()


sns.barplot(x='variable', y='value', data=dados_doencas.melt(id_vars='ICU'),
            hue='ICU')
plt.title('Pacintes por grupo de doênças', fontsize=24, loc='left')
plt.xlabel('Grupo de doênças', fontsize=15)
plt.ylabel('Frequência', fontsize=15)
plt.xticks(rotation=45);


'''
Calculo do número médio de grupo de doênça considerando os pacientes que estão
em pelo menos um grupo
'''
_ = round(pd.Series(dados.query('ICU == 0').drop_duplicates('PATIENT_VISIT_IDENTIFIER').iloc[:, 4:13].sum(axis=1).values).replace({0: None}).mean())
print(f'''
Os pacientes que não foram para a UTI e que estão em algum grupo de doênça em
média estão em {_} grupos.
''')


'''
Calculo do número médio de grupo de doênça considerando os pacientes que estão
em pelo menos um grupo
'''
_ = round(pd.Series(dados.query('ICU == 0').drop_duplicates('PATIENT_VISIT_IDENTIFIER').iloc[:, 4:13].sum(axis=1).values).replace({0: None}).mean())
print(f'''
Os pacientes que foram para a UTI e que estão em algum grupo de doênça em
média estão em {_} grupos.
''')


'''
sinais_vitais_colunas: Identificação geral das colunas referente aos sinais vitais
'''
sinais_vitais_colunas = ['BLOODPRESSURE_DIASTOLIC', 'BLOODPRESSURE_SISTOLIC',
                         'HEART_RATE', 'RESPIRATORY_RATE', 'TEMPERATURE',
                         'OXYGEN_SATURATION']


'''
sinais_vitais: DataFrame com as colunas referente aos sinais vitais dos pacientes,
considerando os dados da primeira janela, os registros com valores NaN serão
preenchidos com os dados da linha subsequente
'''

sinais_vitais = dados_limpos.iloc[:, 192:]
sinais_vitais.head()


# Número de colunas
sinais_vitais.shape[1]


# Número de linhas com o método melt
sinais_vitais.melt().shape[0]


# Número de linhas para cada coluna com o método melt
sinais_vitais.melt().shape[0] / sinais_vitais.shape[1]


'''
linhas_melt: Número de linhas que abrange o grupo das seis colunas referente a
MEAN, MEDIAN, MIN, MAX, DIFF e DIFF_RELL após aplicar o método melt
'''


linhas_colunas_melt = int((sinais_vitais.melt().shape[0] / sinais_vitais.shape[1]) * 6)
linhas_colunas_melt


'''
sinais_vitais_melt: Alteração do formato dos dados referente aos sinais vitais para
visualização dos dados
'''


sinais_vitais_melt = sinais_vitais.melt(id_vars='ICU')
sinais_vitais_melt.head()


sns.violinplot(x='variable', y='value',
               data=sinais_vitais_melt.iloc[0:linhas_colunas_melt],
               hue='ICU', split=True)
plt.xticks(rotation=45)
plt.title('Sinais vitais (Média)', loc='left', fontsize=24);


# Teste de normalidade da variável que visualmente está mais próxima de uma
# distribuição normal no gráfico acima
teste_normalidade(sinais_vitais, 'HEART_RATE_MEAN', 'HEART_RATE_MEAN')


teste_ranksums_colunas(sinais_vitais, sinais_vitais_colunas, 'MEAN')


'''
exames_sangue: DataFrame com as colunas referente aos exames de sangue dos pacientes.
'''
exames_sangue = dados_limpos.iloc[:, 12:]
exames_sangue.drop(exames_sangue.iloc[:, 180:-1], axis=1, inplace=True)
exames_sangue.head()


'''
exames_sangue_melt: Colunas referente aos exames de sangue ordenados pela linha
'''
exames_sangue_melt = exames_sangue.melt(id_vars='ICU')


'''
exames_sangue_colunas: Informações que se repetem nas colunas referente aos exames
de sangue
'''
exames_sangue_colunas = ['MEDIAN', 'MEAN', 'MIN', 'MAX', 'DIFF']


# Número de colunas
exames_sangue.shape[1]


# Número de linhas com o método melt
exames_sangue.melt().shape[0]


# Número de linhas ocupadas por cada coluna ao utilizar o método melt
# Essa informação será utilizada para plotagem dos gráficos
exames_sangue.melt(id_vars='ICU').shape[0] / ((exames_sangue.shape[1]) - 1)


'''
linhas_colunas_melt_2: Número de linhas para plotagem dos gráficos a partir do
método melt
'''
linhas_colunas_melt_2 = int((exames_sangue.melt(id_vars='ICU').shape[0] / ((exames_sangue.shape[1]) - 1)) * 5)
linhas_colunas_melt_2


sns.violinplot(x='variable', y='value',
               data=exames_sangue_melt.iloc[0:linhas_colunas_melt_2],
               hue='ICU', split=True)
plt.xticks(rotation=45)
plt.title('Exames de sangue (ALBUMIN)', loc='left', fontsize=24);


teste_normalidade(exames_sangue, 'ALBUMIN_DIFF', 'ALBUMIN_DIFF')


# Binarizando a coluna AGE_PERCENTIL
dados_limpos = pd.get_dummies(dados_limpos['AGE_PERCENTIL']).merge(dados_limpos, right_index=True,
                                                    left_index=True)


# Excluíndo a coluna original
dados_limpos.drop('AGE_PERCENTIL', axis=1, inplace=True)


'''
colunas_mesma_distribuição: Lista com todas as colunas que apresentam a mesma
distribuição nos grupos que foram e não foram para UTI
'''
colunas_mesma_distribuicao = mesma_distribuicao_colunas(dados_limpos)


# teste 1 | Sinais vitais
for i in ['BLOODPRESSURE_SISTOLIC_MEAN', 'HEART_RATE_MEAN', 'TEMPERATURE_MEAN']:
 print(i in colunas_mesma_distribuicao)


# teste 2 | Exames de sangue
for i in ['ALBUMIN_MEDIAN', 'ALBUMIN_MEAN', 'ALBUMIN_MIN', 'ALBUMIN_MAX',
          'ALBUMIN_DIFF']:
  print(i in colunas_mesma_distribuicao)
  

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Modelos de Machine Learning
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


roda_modelo_cv(dados_limpos, modelo=DummyClassifier(),
                 descricao_modelo='Dummy Classifier')


roda_modelo_cv(dados_limpos, modelo=LogisticRegression(max_iter=300),
                 descricao_modelo='Logistic Regression')


roda_modelo_cv(dados_limpos)


'''
auc_teste_tree: Lista com os resultados do AUC médio de teste para cada nó do modelo
de árvore
auc_treino_tree: Lista com os resultados do AUC médio de treino para cada nó do
modelo de árvore
'''
auc_teste_tree = list()
auc_treino_tree = list()
for i in range(1, 10):
  teste, treino = roda_modelo_cv(dados_limpos, modelo=DecisionTreeClassifier(max_depth=i),
                 descricao_modelo='Decision Tree Classifier')
  auc_teste_tree.append(teste)
  auc_treino_tree.append(treino)


curva_auc(auc_teste_tree, auc_treino_tree, 'Decison Tree Classifier')


roda_modelo_cv(dados_limpos, modelo=RandomForestClassifier())


'''
auc_teste_forest: Lista com os resultados do AUC médio de teste para cada nó do
modelo Random Forest Classifier
auc_treino_forest: Lista com os resultados do AUC médio de treino para cada nó do
modelo Random Forest Cassifier
'''


auc_teste_forest = list()
auc_treino_forest = list()
for i in range(1, 10):
  teste, treino = roda_modelo_cv(dados_limpos, modelo=RandomForestClassifier(max_depth=i),
                 descricao_modelo='Random Forest Classifier')
  auc_teste_forest.append(teste)
  auc_treino_forest.append(treino)
  

curva_auc(auc_teste_forest, auc_treino_forest, 'Random Forest Classifier')


'''
dados_limpos_2: Cópia dos dados limpos desconsiderando as colunas que possuem a
mesma distribuição para os pacientes que foram e não forma para UTI
'''
dados_limpos_2 = dados_limpos.copy()
dados_limpos_2.drop(colunas_mesma_distribuicao, axis=1, inplace=True)

dados_limpos_2.head()



# Linhas e colunas
dados_limpos_2.shape

# Proporção após desconsiderar colunas com a mesma distribuição nos grupos de
# paciente que foram e não formam para UTI
print(f'''
Proporção de colunas desconsiderando aquelas que apresentam a mesma distribuição
para os pacientes que foram e não foram para UTI: {round(dados_limpos_2.shape[1] / dados_limpos.shape[1] * 100)}%
''')

roda_modelo_cv(dados_limpos_2, modelo=DummyClassifier(),
                 descricao_modelo='Dummy Classifier')


roda_modelo_cv(dados_limpos_2, modelo=LogisticRegression(max_iter=300),
                 descricao_modelo='Logistic Regression')


roda_modelo_cv(dados_limpos_2)

'''
auc_teste_tree: Lista com os resultados do AUC médio de teste para cada nó do modelo
de árvore
auc_treino_tree: Lista com os resultados do AUC médio de treino para cada nó do
modelo de árvore
'''
auc_teste_tree = list()
auc_treino_tree = list()
for i in range(1, 10):
  teste, treino = roda_modelo_cv(dados_limpos_2, modelo=DecisionTreeClassifier(max_depth=i),
                 descricao_modelo='Decision Tree Classifier')
  auc_teste_tree.append(teste)
  auc_treino_tree.append(treino)
  
  
roda_modelo_cv(dados_limpos_2, modelo=RandomForestClassifier())

'''
auc_teste_forest: Lista com os resultados do AUC médio de teste para cada nó do
modelo Random Forest Classifier
auc_treino_forest: Lista com os resultados do AUC médio de treino para cada nó do
modelo Random Forest Cassifier
'''
auc_teste_forest = list()
auc_treino_forest = list()
for i in range(1, 10):
  teste, treino = roda_modelo_cv(dados_limpos_2, modelo=RandomForestClassifier(max_depth=i),
                 descricao_modelo='Random Forest Classifier')
  auc_teste_forest.append(teste)
  auc_treino_forest.append(treino)
  
curva_auc(auc_teste_forest, auc_treino_forest, 'Random Forest Classifier')

roda_modelo_cv(dados_limpos_2, modelo=SVC())



'''
modelo_random_forest_1: Modelo de Machine Learning com os dados limpos para
facilitar a visualização das featurer importances
x_treino_1: Dados de treino do modelo
x_teste_1: Dados de teste da variável x
y_teste_1: Dads de teste da variável y
y_predito_1: Valores previsto para y
'''
modelo_random_forest_1, x_treino_1, x_teste_1, y_treino_1, y_teste_1, y_predito_1 = modelo_treino_teste(dados_limpos,
                                                                                              modelo=RandomForestClassifier(max_depth=6))

'''
modelo_random_forest_2: Modelo de Machine Learning com os dados limpos 2 para
facilitar a visualização das featurer importances
x_treino_2: Dados de treino do modelo
x_teste_2: Dados de teste da variável x
y_teste_2: Dads de teste da variável y
y_predito_2: Valores previsto para y
'''
modelo_random_forest_2, x_treino_2, x_teste_2, y_treino_2, y_teste_2, y_predito_2 = modelo_treino_teste(dados_limpos_2,
                                                                                              modelo=RandomForestClassifier(max_depth=7))


# Visualização sem validação cruzada
ConfusionMatrixDisplay.from_estimator(modelo_random_forest_1, x_teste_1, y_teste_1)
personaliza_grafico('Matriz de Confusão (Dados Limpos 1)', 'Predicted Label', 'True Label')


print(classification_report(y_teste_1, y_predito_1))


_ = pd.DataFrame({'feature_importance': modelo_random_forest_1.feature_importances_,
                  'variavel': x_treino_1.columns}).sort_values('feature_importance', ascending=False)
plt.figure(figsize=(16, 12))
sns.barplot(data=_[:20], y='variavel', x='feature_importance', orient='h')
personaliza_grafico('Featurer importances Random Forest Classifier (Dados Limpos)',
                    'Featurer importances', '')


_ = pd.DataFrame({'feature_importance': modelo_random_forest_2.feature_importances_,
                  'variavel': x_treino_2.columns}).sort_values('feature_importance', ascending=False)
plt.figure(figsize=(16, 12))
sns.barplot(data=_[:20], y='variavel', x='feature_importance', orient='h')
personaliza_grafico('Featurer importances Random Forest Classifier (Dados Limpos 2)',
                    'Featurer importances', '')


sns.violinplot(x='variable', y='value', hue='ICU',
               data=dados_limpos_2[['BLOODPRESSURE_DIASTOLIC_MIN', 'ICU']].melt(id_vars='ICU'),
               split=True)


personaliza_grafico('Pressão Arterial Diastólica Mínima', 'Grupos de pacientes', 'Frequência');


sns.violinplot(x='variable', y='value', hue='ICU',
               data=dados_limpos_2[['BLOODPRESSURE_DIASTOLIC_MEAN', 'ICU']].melt(id_vars='ICU'),
               split=True)


personaliza_grafico('Pressão Arterial Diastólica Média', 'Grupos de pacientes', 'Frequência');


sns.violinplot(x='variable', y='value', hue='ICU',
               data=dados_limpos_2[['PCR_MIN', 'ICU']].melt(id_vars='ICU'),
               split=True)


personaliza_grafico('PCR Mínimo', 'Grupos de pacientes', 'Frequência');


sns.violinplot(x='variable', y='value', hue='ICU',
               data=dados_limpos_2[['CREATININ_MEDIAN', 'ICU']].melt(id_vars='ICU'),
               split=True)


personaliza_grafico('Creatina Mediana', 'Grupos de pacientes', 'Frequência');










  