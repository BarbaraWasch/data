import pandas as pd  # Biblioteca para manipulação de dados (DataFrames)
import numpy as np  # Manipulação de arrays e operações matemáticas
import matplotlib.pyplot as plt  # Visualização de dados com gráficos
import seaborn as sns  # Visualização avançada e estilizada de gráficos

# Definindo a URL do arquivo CSV
url = 'https://raw.githubusercontent.com/rafaelrdias/DS_Examples/refs/heads/main/Arquivos/Plano_de_Saude.csv'

# Carregando os dados do CSV diretamente da URL para um DataFrame 'data'
data = pd.read_csv(url)

# Limpeza dos dados
data = data.dropna()  # Remove linhas com valores nulos
data = data.drop_duplicates()  # Remove linhas duplicadas

# Verificando os tipos dos dados
print("Tipos de dados das colunas:")
print(data.dtypes)

# Definindo colunas para análise
numerical_columns = ['Idade', 'Indice_Massa_Corporal', 'Valor_Pago_PS']
categorical_columns = ['Genero_Biologico', 'Fumante', 'Filhos', 'Regiao']

# Função para plotar histogramas para colunas numéricas
def plot_numerical_distributions(data, columns):
    for column in columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data[column], bins=30, kde=True)
        plt.title(f'Distribuição de {column}', fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

# Plotando distribuições para colunas numéricas
plot_numerical_distributions(data, numerical_columns)

# Função para plotar gráficos de contagem para colunas categóricas
def plot_categorical_distributions(data, columns):
    for column in columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=data, x=column, palette='Set2')
        plt.title(f'Distribuição de {column}', fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Contagem', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

# Plotando distribuições para colunas categóricas
plot_categorical_distributions(data, categorical_columns)

# Gráficos de violino para variáveis numéricas categorizadas
def plot_violin(data, numeric_col, categorical_col):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x=categorical_col, y=numeric_col, palette='Set2')
    plt.title(f'Distribuição de {numeric_col} por {categorical_col}', fontsize=16)
    plt.xlabel(categorical_col, fontsize=12)
    plt.ylabel(numeric_col, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Gráficos de violino para 'Valor_Pago_PS' por 'Regiao' e 'Fumante'
plot_violin(data, 'Valor_Pago_PS', 'Regiao')
plot_violin(data, 'Valor_Pago_PS', 'Fumante')

# Boxplots para comparação
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Regiao', y='Valor_Pago_PS', palette='Set2')
plt.title('Valor Pago por Plano de Saúde por Região', fontsize=16)
plt.xlabel('Região', fontsize=12)
plt.ylabel('Valor Pago', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='Fumante', y='Valor_Pago_PS', palette='Set1')
plt.title('Impacto de Fumar no Valor do Plano de Saúde', fontsize=16)
plt.xlabel('Fumante', fontsize=12)
plt.ylabel('Valor Pago', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Gráficos de dispersão para observar relações entre variáveis numéricas
def plot_scatter(data, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_col, y=y_col, hue='Fumante', palette='Set1', alpha=0.7)
    plt.title(f'Relação entre {x_col} e {y_col}', fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid()
    plt.show()

# Plotando gráficos de dispersão
plot_scatter(data, 'Idade', 'Valor_Pago_PS')
plot_scatter(data, 'Indice_Massa_Corporal', 'Valor_Pago_PS')

# Matriz de correlação com um gráfico de calor
plt.figure(figsize=(10, 8))
correlation_matrix = data[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Matriz de Correlação', fontsize=16)
plt.show()

# Exibir as primeiras linhas para identificar as colunas
print("Primeiras 5 linhas do DataFrame:")
print(data.head())

# Verificar a distribuição das regiões
print("\nDistribuição de Planos de Saúde por Região:")
print(data['Regiao'].value_counts())

# Criar o gráfico de contagem da distribuição das regiões
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Regiao', palette='Set2')
plt.title('Distribuição de Planos de Saúde por Região', fontsize=16)
plt.xlabel('Região', fontsize=12)
plt.ylabel('Contagem', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Função para calcular e exibir estatísticas descritivas
def display_statistics(data, columns):
    for column in columns:
        print(f"\nEstatísticas para {column}:")
        media = data[column].mean()
        moda = data[column].mode()[0]
        quartis = data[column].quantile([0.25, 0.5, 0.75])
        
        print(f"Média: {media:.2f}")
        print(f"Moda: {moda}")
        print(f"Quartil 1 (25%): {quartis[0.25]:.2f}")
        print(f"Mediana (50%): {quartis[0.50]:.2f}")
        print(f"Quartil 3 (75%): {quartis[0.75]:.2f}")

# Exibir estatísticas descritivas para colunas numéricas
display_statistics(data, numerical_columns)
