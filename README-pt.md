# Rede Neural Totalmente Conectada para Classificação de Emoções (Português)

## Visão Geral
Este projeto implementa uma rede neural totalmente conectada (FCNN) para classificação de emoções em textos em português, utilizando o dataset [Portuguese Tweets](https://huggingface.co/datasets/fpaulino/portuguese-tweets) da Hugging Face. O modelo classifica textos em três categorias de sentimentos: tristeza, alegria e neutro.

## 📑 Índice
- [Rede Neural Totalmente Conectada para Classificação de Emoções (Português)](#rede-neural-totalmente-conectada-para-classificação-de-emoções-português)
  - [Visão Geral](#visão-geral)
  - [📑 Índice](#-índice)
  - [📁 Estrutura do Projeto](#-estrutura-do-projeto)
  - [📊 Conjunto de Dados](#-conjunto-de-dados)
  - [🔍 Pré-processamento de Texto](#-pré-processamento-de-texto)
  - [🧠 Arquitetura do Modelo](#-arquitetura-do-modelo)
  - [⚙️ Detalhes do Treinamento](#️-detalhes-do-treinamento)
  - [🚀 Instalação](#-instalação)
    - [Pré-requisitos](#pré-requisitos)
    - [Passos](#passos)
  - [🛠️ Uso](#️-uso)
    - [📥 Preparação dos Dados](#-preparação-dos-dados)
    - [🏋️ Treinar o Modelo](#️-treinar-o-modelo)
    - [🔍 Testar o Modelo](#-testar-o-modelo)
  - [📈 Desempenho](#-desempenho)
  - [📋 Requisitos](#-requisitos)
  - [👥 Contribuindo](#-contribuindo)
  - [📄 Licença](#-licença)
  - [🙏 Agradecimentos](#-agradecimentos)

## 📁 Estrutura do Projeto
```
fcnn_pt_BR_emotion_classification/
│
├── data/                  # Diretório de dados
│   ├── train_data.parquet # Dados de treinamento
│   ├── test_data.parquet  # Dados de teste
│   └── validation_data.parquet # Dados de validação
│
├── model/                 # Diretório de artefatos do modelo
│   ├── fcnn.keras         # Modelo Keras salvo
│   ├── label_encoder.joblib # Codificador de rótulos serializado
│   └── vectorizer.joblib  # Vetorizador TF-IDF serializado
│
├── utils/                 # Módulos utilitários
│   ├── __init__.py
│   ├── evaluate_and_save.py # Utilitários para avaliação e salvamento do modelo
│   ├── preprocessing.py   # Funções de pré-processamento de texto
│   └── training_plot.py   # Utilitários de visualização de treinamento
│
├── explore_data.ipynb     # Notebook para Análise Exploratória de Dados (EDA)
├── data_preparation.py    # Script para download e pré-processamento de dados
├── model_trainer.py       # Script para treinamento e avaliação do modelo
├── test_model.py          # Script para testar o modelo com novas entradas (a ser criado)
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação do projeto
```

## 📊 Conjunto de Dados
Este projeto utiliza o dataset [Portuguese Tweets](https://huggingface.co/datasets/fpaulino/portuguese-tweets) da Hugging Face, que contém amostras de texto em português rotuladas com três sentimentos:
- 😢 Tristeza
- 😄 Alegria
- 😐 Neutro

O dataset contém um grande número de tweets rotulados, divididos em conjuntos de treinamento e teste. Uma parte do conjunto de treinamento é separada para validação.

## 🔍 Pré-processamento de Texto
O pipeline de pré-processamento inclui:
- Lematização: Reduzir palavras à sua forma base usando spaCy (`pt_core_news_md`)
- Conversão para minúsculas: Converter todo o texto para letras minúsculas
- Remoção de stopwords: Remover palavras comuns em português que carregam pouca informação significativa
- Normalização de espaços: Padronizar os espaços entre palavras

O texto processado é então vetorizado usando TF-IDF (Term Frequency-Inverse Document Frequency), que converte o texto em recursos numéricos com base na importância das palavras.

## 🧠 Arquitetura do Modelo
O modelo é uma rede neural totalmente conectada com a seguinte arquitetura:

![FCNN Architecture Placeholder](https://via.placeholder.com/1192x400?text=Arquitetura+da+FCNN+(3+Classes))

- Camada de entrada: Texto vetorizado com TF-IDF (matriz esparsa convertida para densa)
- Camada oculta 1: 1024 neurônios com ativação SELU, inicializador `lecun_normal` e regularização L2 (0.01)
- Camada oculta 2: 512 neurônios com ativação SELU, inicializador `lecun_normal` e regularização L2 (0.01)
- Camada oculta 3: 256 neurônios com ativação SELU, inicializador `lecun_normal` e regularização L2 (0.01)
- Camada oculta 4: 64 neurônios com ativação SELU
- Camada de saída: 3 neurônios com ativação softmax (um para cada sentimento)

O modelo utiliza a função de ativação SELU (Scaled Exponential Linear Unit), que auxilia nas propriedades de auto-normalização e pode levar a uma melhor convergência durante o treinamento.

## ⚙️ Detalhes do Treinamento
- Função de perda: Categorical Crossentropy (`CategoricalCrossentropy`)
- Otimizador: Adam
- Taxa de aprendizado: Começa em 0.001 com decaimento por etapas (reduzida pela metade a cada 10 épocas)
- Pesos das classes: Balanceados usando `compute_class_weight` para lidar com o desequilíbrio de classes
- Parada antecipada: Monitorada na perda de validação (`val_loss`) com paciência de 3 épocas
- Tamanho do lote: 256
- Máximo de épocas: 20

## 🚀 Instalação

### Pré-requisitos
- Python 3.8+
- Gerenciador de pacotes pip

### Passos
1. Clone o repositório:
```bash
git clone https://github.com/yourusername/fcnn_pt_BR_emotion_classification.git # Substitua pela URL correta
cd fcnn_pt_BR_emotion_classification
```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows, use: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Baixe o modelo spaCy para português (feito automaticamente em `preprocessing.py`, mas pode ser executado separadamente):
```bash
python -m spacy download pt_core_news_md
```

## 🛠️ Uso

### 📥 Preparação dos Dados
Baixe e pré-processe o conjunto de dados:
```bash
python data_preparation.py
```
Este script irá:
- Baixar o dataset Portuguese Tweets da Hugging Face
- Processar os dados de texto (lematização, remoção de stopwords, etc.)
- Salvar os dados processados como arquivos parquet no diretório `data/`

### 🏋️ Treinar o Modelo
Treine o modelo FCNN com os dados pré-processados:
```bash
python model_trainer.py
```
Este script irá:
- Carregar os dados pré-processados
- Aplicar a vetorização TF-IDF
- Treinar o modelo com a arquitetura e hiperparâmetros especificados
- Exibir gráficos de treinamento mostrando as curvas de perda
- Avaliar o modelo no conjunto de teste
- Salvar o modelo treinado em `model/fcnn.keras`

### 🔍 Testar o Modelo
Teste o modelo treinado com novas entradas de texto (criar `test_model.py`):
```bash
python test_model.py
```
Este script (a ser criado) permitirá testar o modelo com frases em português.

## 📈 Desempenho
As métricas de desempenho do modelo (acurácia, precisão, recall, F1-score) são avaliadas no conjunto de teste e exibidas após o treinamento. O desempenho típico para esta arquitetura apresenta:

- Acurácia geral: ~65-70%
- Precisão e recall variam por classe de emoção

![Resultados da Classificação FCNN](model/fcnn_model.png)

A matriz de confusão ajuda a visualizar quais emoções são mais frequentemente classificadas incorretamente e quais o modelo prevê com maior confiança.

## 📋 Requisitos
- Python 3.8+
- TensorFlow
- scikit-learn
- pandas
- numpy
- spaCy (com o modelo `pt_core_news_md`)
- matplotlib
- joblib
- nltk (para stopwords)
- pyarrow (para leitura de parquet)
- emoji

*(Consulte `requirements.txt` para versões específicas)*

## 👥 Contribuindo
Contribuições para melhorar a arquitetura do modelo, o pré-processamento ou adicionar novos recursos são bem-vindas! Sinta-se à vontade para enviar um pull request.

1. Faça um fork do repositório
2. Crie sua branch de recurso (`git checkout -b feature/novo-recurso`)
3. Faça commit das suas alterações (`git commit -m 'Adicionar novo recurso'`)
4. Envie para a branch (`git push origin feature/novo-recurso`)
5. Abra um Pull Request

## 📄 Licença
[Licença MIT](LICENSE)

## 🙏 Agradecimentos
- [fpaulino](https://huggingface.co/fpaulino) por fornecer o dataset Portuguese Tweets
- [SpaCy](https://spacy.io/) por utilitários de processamento de linguagem natural
- [TensorFlow](https://www.tensorflow.org/) e [Keras](https://keras.io/) pelo framework de deep learning
- [Scikit-learn](https://scikit-learn.org/) por utilitários de machine learning
- [NLTK](https://www.nltk.org/) por stopwords em português
- [Hugging Face](https://huggingface.co/) pela plataforma de datasets
- [Netron](https://github.com/lutzroeder/netron) pela visualização de modelos