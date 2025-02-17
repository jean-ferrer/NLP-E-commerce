# Natural Language Processing (NLP) e Unsupervised Machine Learning em E-Commerce #

##### Este projeto aplica técnicas de Natural Language Processing
##### (Processamento de Linguagem Natural) e Unsupervised Machine
##### Learning (Aprendizado de Máquina Não Supervisionado) para
##### identificar padrões em comentários negativos de plataformas
##### de vendas brasileiras, agrupando textos semelhantes e extraindo
##### tópicos relevantes.

##### Links para os modelos de machine learning e data set:
##### - K-Means: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
##### - Latent Dirichlet Allocation (LDA): https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
##### - b2w Data Set: https://www.kaggle.com/code/abnerfreitas/nlp-buscape-data-ptbr-sentiment-analysis/input





## Extração e Representação do Corpus ##


#### Imports ####

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA, PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import spacy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



#### Criação do DataFrame e Separação do Corpus ####

# Carrega o dataset e remove linhas com valores NaN em qualquer coluna
df = pd.read_csv('b2w.csv').dropna()

# Filtra o dataframe para obter apenas os comentários classificados como negativos (polarity == 0)
corpus_negative = df.loc[df['polarity'] == 0]

# Seleciona apenas a coluna 'review_text', contendo os textos das avaliações negativas
corpus = corpus_negative['review_text']



#### Pré-Processamento do Corpus ####

# Carrega o modelo da língua portuguesa 'pt_core_news_lg' 
nlp = spacy.load('pt_core_news_lg', disable=["morphologizer", "senter", "attribute_ruler", "ner"]) # 'disable' desativa componentes que não serão utilizados no processamento de texto

# Lista de palavras irrelevantes para a análise, mas frequentes nos textos – nomes de lojas, verbos genéricos e palavras comuns que não agregam valor na criação de clusters e tópicos
additional_stop_words = {
    "produto", "comprei", "dia", "loja", "pra", "comprar", "compra", "hoje", "empresa", "americana", 
    "americanas", "polishop", "casas bahia", "casa bahia", "mercado livre", "ml", "site", "fastshop", 
    "amazon", "magazine luiza", "magalu", "recomendar", "conseguir", "ter", "já", "ja"
}

# Função de pré-processamento do corpus: lematiza os textos e remove stop-words
def preprocess(corpus):
  result = []

  for text in corpus.tolist():
    doc = nlp(text.lower()) # Converte para minúsculas
    tokens = [
      token.lemma_ # Obtém a forma lematizada da palavra
      for token in doc
      if token.is_alpha # Mantém apenas palavras (remove números e pontuações)
      and not token.is_stop # Garante que a palavra não está na lista padrão de stop-words de português do spaCy
      and token.lemma_ not in additional_stop_words # Garante que a palavra não está na lista adicional de stop-words
      ]
    
    result.append(' '.join(tokens).strip()) # Junta tokens em um único texto processado

  # Remove textos vazios antes de retornar
  return [text for text in result if text]

# Aplica o pré-processamento ao corpus original
corpus_preprocessed = preprocess(corpus)

# Mostra o tamanho e os primeiros exemplos do corpus original
print(len(corpus))
print(corpus[:3])

# Mostra o tamanho e os primeiros exemplos do corpus pré-processado
print(len(corpus_preprocessed))
print(corpus_preprocessed[:3])



#### Representação de Texto com TF-IDF ####

# Instancia um vetorizador TF-IDF que converte textos em representações numéricas com pesos que refletem a importância dos termos (dentro de cada documento e em relação ao corpus inteiro)
vectorizer = TfidfVectorizer()

# Ajusta o vetorizador ao corpus pré-processado (aprende o vocabulário) e o transforma em uma matriz esparsa documento-termo
X_tfidf = vectorizer.fit_transform(corpus_preprocessed)



#### Representação de Texto com Bag-of-Words (BoW) ####

# Instancia um vetorizador Bag of Words (BoW) que converte textos em uma matriz baseada na contagem de palavras
extractor = CountVectorizer()

# Ajusta o vetorizador ao corpus pré-processado (aprende o vocabulário) e o transforma em uma matriz esparsa documento-termo
X_bow = extractor.fit_transform(corpus_preprocessed)



#### Exibição das Palavras Mais Frequentes ####

# Obtém as palavras do vocabulário aprendido pelo CountVectorizer
words = extractor.get_feature_names_out()

# Conta a frequência total de cada palavra no corpus
word_counts = np.asarray(X_bow.sum(axis=0)).flatten()

# Cria um DataFrame associando cada palavra à sua respectiva frequência
df_words = pd.DataFrame({'word': words, 'count': word_counts})

# Define a quantidade de palavras mais frequentes a serem exibidas
words_amount = 50

# Ordena o DataFrame em ordem decrescente de frequência e seleciona as 'words_amount' palavras mais comuns
top_words = df_words.sort_values(by='count', ascending=False).head(words_amount)

# Exibe as palavras mais frequentes e suas contagens
print(top_words)



#### Criação e Pré-Processamento do Texto para Teste ####

# Define um texto de teste para processamento
test_text = 'Ainda não recebi o produto. Vou cancelar a compra se não o receber até amanhã!'

# Converte a string para pd.Series, aplica a função 'preprocess' e reconverte para uma única string unindo os tokens processados
test_text = ' '.join(preprocess(pd.Series([test_text])))
print(test_text)





## Unsupervised Machine Learning ##

#### Clusterização de Documentos com K-Means ####

# Instancia o modelo K-Means para agrupamento de textos
kmeans = KMeans(n_clusters=3, max_iter=1000, random_state=42)
# n_clusters: define o número de clusters que o algoritmo deve encontrar
# max_iter: define o número máximo de iterações para convergência
# random_state: garante reprodutibilidade dos resultados

# Ajusta o modelo aos dados vetorizados pelo TF-IDF, aprendendo os centros dos clusters
kmeans.fit(X_tfidf)

# Transforma o texto de teste para a representação vetorizada TF-IDF
X_test = vectorizer.transform([test_text])

# Obtém o cluster atribuído ao documento pelo modelo K-Means
cluster_label = kmeans.predict(X_test)[0]

# Exibe o resultado do cluster atribuído ao documento de teste
print(f'O documento "{test_text}" pertence ao cluster {cluster_label}.\n')

# Exibe os primeiros documentos pertencentes ao mesmo cluster
print(f'Os primeiros documentos do cluster {cluster_label} são:')
count = 0
for i, text in enumerate(corpus):
  if kmeans.labels_[i] == cluster_label:
    print(f'- {text}')
    count += 1
    if count == 5:
      break

print("Documentos de outros clusters:\n")

# Número total de clusters no modelo K-Means
num_clusters = kmeans.n_clusters

# Percorre todos os clusters e imprime exemplos de cada, exceto do cluster atribuído ao texto de teste
for c in range(num_clusters):
    if c == cluster_label:
        continue  # Pula o cluster do documento testado
    print(f'Cluster {c}:')
    count = 0
    for i, text in enumerate(corpus):
        if kmeans.labels_[i] == c:
            print(f'- {text}')
            count += 1
            if count == 5:
                break
    print()



#### Visualização do K-Means por Meio de Redução de Dimensionalidade com Principal Component Analysis (PCA) ####

# Instancia o modelo PCA para reduzir a dimensionalidade dos vetores TF-IDF para 2D
pca = PCA(n_components=2)
# n_components: define a quantidade de componentes principais a serem mantidos

# Aplica PCA aos vetores TF-IDF convertidos em array
X_reduced = pca.fit_transform(X_tfidf.toarray())
# fit_transform: ajusta o PCA aos dados e os transforma para a nova dimensionalidade

# Plota os clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.25)

# Adiciona título e rótulos
plt.title("Clusters K-Means com PCA (2D)", fontsize=16)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label="Cluster")

# Exibe o gráfico
plt.show()
# Fecha o gráfico para liberar memória
plt.clf()



#### Extração de Tópicos com Latent Dirichlet Allocation (LDA) ####

# Instancia o modelo LDA para identificação de tópicos
lda = LDA(n_components=3, max_iter=20, random_state=42)
# n_components: define o número de tópicos que o modelo deve identificar no corpus
# max_iter: define o número máximo de iterações para a convergência do modelo
# random_state: garante reprodutibilidade dos resultados

# Ajusta o modelo aos dados vetorizados pelo BoW, aprendendo a distribuição dos tópicos
lda.fit(X_bow)

# Transforma o documento de teste usando o vetorizador BoW
X_test = extractor.transform([test_text])

# Obtém a distribuição de probabilidade do documento em relação aos tópicos extraídos pelo LDA
topic_probabilities = lda.transform(X_test)[0]

# Obtém os índices dos tópicos mais relevantes
top_topics = topic_probabilities.argsort()[::-1][-5:] # [-5:] mostra até 5 tópicos e [::-1] organiza em ordem decrescente

# Obtém o vocabulário aprendido pelo vetorizador BoW
word_index = extractor.get_feature_names_out()

print('Os tópicos que possuem maior compatibilidade com o documento de teste são:')
for topic in top_topics:

  # Seleciona as palavras mais importantes para o tópico (ordenadas por peso no modelo)
  frequent_words = lda.components_[topic].argsort()[-10:][::-1] # [-10:] mostra até 10 palavras e [::-1] organiza em ordem decrescente
  words = ', '.join([word_index[i] for i in frequent_words])

  # Exibe o tópico, sua probabilidade e suas palavras mais representativas
  print(f'- Tópico {topic} ({round((topic_probabilities[topic])*100, 1)}%) | Palavras principais do tópico: {words}')



#### Visualização do LDA por Meio de Redução de Dimensionalidade com t-Distributed Stochastic Neighbor Embedding (t-SNE) ####

# Obtém as distribuições de tópicos para todos os documentos
lda_topic_distributions = lda.transform(X_bow)

# Instancia o modelo t-SNE para reduzir a dimensionalidade dos tópicos para 3D
tsne = TSNE(n_components=3, random_state=42)
# n_components: reduz os dados para a dimensão inserida
# random_state: garante reprodutibilidade dos resultados

# Aplica t-SNE às distribuições de tópicos
X_tsne = tsne.fit_transform(lda_topic_distributions)

# Cria a figura para o gráfico 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Define o ângulo de visualização
ax.view_init(elev=45, azim=45)  # 45° de elevação e 45° de rotação horizontal

# Plota os documentos em 3D com base nos tópicos LDA
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=lda_topic_distributions.argmax(axis=1), cmap='viridis', alpha=0.75)

# Adiciona título e rótulos
ax.set_title("Distribuição dos Tópicos LDA com t-SNE (3D)", fontsize=16)
ax.set_xlabel("Componente t-SNE 1")
ax.set_ylabel("Componente t-SNE 2")
ax.set_zlabel("Componente t-SNE 3")

# Exibe o gráfico
plt.show()
# Fecha o gráfico para liberar memória
plt.clf()



#### Criação de DataFrame com Dados do K-Means e LDA ####

# Lista de clusters atribuídos a cada texto preprocessado do corpus pelo K-Means
cluster_kmeans = [kmeans.predict(vectorizer.transform([text]))[0] for text in corpus_preprocessed]

# Lista dos tópicos mais relevantes atribuídos a cada texto preprocessado do corpus pelo LDA
topics_lda = [lda.transform(extractor.transform([text]))[0].argmax() for text in corpus_preprocessed]

data = {
    "corpus_preprocessed": corpus_preprocessed,  # Lista de textos pré-processados
    "cluster_kmeans": cluster_kmeans,  # Número do cluster K-Means para cada texto
    "x_kmeans": X_reduced[:, 0],  # Primeira componente principal (PCA -> K-Means)
    "y_kmeans": X_reduced[:, 1],  # Segunda componente principal (PCA -> K-Means)
    "topic_lda": topics_lda,  # Número do tópico LDA para cada texto
    "x_tsne": X_tsne[:, 0],  # Primeira componente t-SNE (LDA)
    "y_tsne": X_tsne[:, 1],  # Segunda componente t-SNE (LDA)
    "z_tsne": X_tsne[:, 2],  # Terceira componente t-SNE (LDA)
}

# Cria o dataframe com as variáveis de interesse
df_kmeans_lda = pd.DataFrame(data)

# Exibe as primeiras linhas do dataframe
print(df_kmeans_lda.head())

# Salva o dataframe
df_kmeans_lda.to_csv("df_kmeans_lda.csv", index=False, encoding="utf-8")