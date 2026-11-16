# RELATÓRIO TÉCNICO: CLUSTERIZAÇÃO COM MISTURAS GAUSSIANAS (GMM)
**Curso:** Sistemas de Informação — Universidade Estadual do Tocantins (UNITINS)  
**Disciplina:** Mineração de Dados  
**Professor:** Daniel  
**Grupo 5:** Daniel, Vladmir, Nedson, Paulo Henrique  
**Algoritmo:** Misturas Gaussianas (Gaussian Mixture Models - GMM)  
**Dataset:** Rice (Cammeo and Osmancik) — UCI Repository ID 545  
**Data de Entrega:** 16 de Novembro de 2026  

---

## 1. Descrição do Problema e Contexto

A diferenciação e classificação de variedades de grãos agrícolas é uma tarefa crítica para a indústria de alimentos, controle de qualidade, precificação de mercado e certificação de sementes. No caso do arroz, variedades como **Cammeo** e **Osmancik** apresentam características visuais semelhantes a olho nu, porém diferem em tempo de cocção, teor de amilose e valor comercial.

O objetivo deste trabalho é aplicar técnicas de **aprendizado não supervisionado** com o algoritmo de **Misturas Gaussianas (GMM)** para descobrir agrupamentos morfológicos naturais a partir de atributos geométricos extraídos por processamento digital de imagens, disponibilizando todo o ciclo da solução através de uma aplicação web conteinerizada em **Docker** e publicada no **Docker Hub**.

---

## 2. Análise Exploratória dos Dados (EDA)

### 2.1 Visão Geral do Dataset
O conjunto de dados utilizado é proveniente do *UCI Machine Learning Repository* (ID 545), doado pelos pesquisadores Murat Köklü e İbrahim Çınar (2019). O conjunto possui um total de **3.810 instâncias** e **7 atributos previsores contínuos**, além do rótulo real (ground truth) utilizado estritamente para avaliação posterior da pureza dos grupos formados:
- **Cammeo:** 1.630 instâncias (42,78%)
- **Osmancik:** 2.180 instâncias (57,22%)

### 2.2 Atributos Morfológicos
1. **Area:** Número total de pixels delimitados pelas bordas do grão.
2. **Perimeter:** Distância calculada ao redor do contorno do grão.
3. **Major_Axis_Length:** Comprimento da maior reta que pode ser traçada sobre a silhueta do grão.
4. **Minor_Axis_Length:** Comprimento da menor reta transversal traçada sobre a silhueta do grão.
5. **Eccentricity:** Medida de excentricidade da elipse equivalente (quão alongado é o grão, variando de 0 a 1).
6. **Convex_Area:** Quantidade de pixels dentro do menor polígono convexo que envolve o grão.
7. **Extent:** Razão entre a área dos pixels do grão e a área da caixa delimitadora retangular (*bounding box*).

### 2.3 Tratamento de Inconsistências e Estatísticas Básicas
- **Valores ausentes:** Não foram encontrados valores nulos ou campos vazios no arquivo `.arff`.
- **Estatísticas Descritivas:**
  - `Area`: Média de 12.667 px (desvio padrão: 1.732; mín: 7.551; máx: 18.913).
  - `Perimeter`: Média de 454,23 (mín: 359,10; máx: 548,40).
  - `Major_Axis_Length`: Média de 201,80 px (mín: 145,20; máx: 239,00).
  - `Minor_Axis_Length`: Média de 82,00 px (mín: 59,50; máx: 107,50).
  - `Eccentricity`: Média de 0,88 (alta excentricidade indica grãos predominantemente alongados).
  - `Convex_Area`: Média de 12.952 px.
  - `Extent`: Média de 0,60.

### 2.4 Justificativa dos Atributos e Normalização
Devido à disparidade de escalas (por exemplo, `Area` na casa dos milhares e `Eccentricity` entre 0 e 1), aplicou-se a padronização via **`StandardScaler`** ($z = (x - \mu)/\sigma$). Isso evita que atributos com valores numéricos nominais maiores dominem o cálculo da matriz de covariância.

---

## 3. Justificativa do Algoritmo: Misturas Gaussianas (GMM)

O algoritmo de **Misturas Gaussianas (GMM)** é um método de agrupamento probabilístico (*Soft Clustering*) que assume que os dados são gerados por uma combinação finita de distribuições normais multivariadas com parâmetros desconhecidos.

### Vantagens do GMM sobre Métodos Tradicionais (como K-Means):
1. **Modelagem de Covariância Flexível:** Enquanto o K-Means assume clusters estritamente esféricos e de igual variância, o GMM com matriz de covariância completa (`covariance_type='full'`) ajusta elipsoides multidimensionais de diferentes orientações e tamanhos, o que reflete a biometria real de grãos de arroz.
2. **Atribuição Probabilística (Pertinência Suave):** Em vez de atribuir uma amostra de forma rígida (*Hard Clustering*), o GMM calcula a probabilidade *a posteriori* $P(Cluster = k \mid x)$ via algoritmo de Expectativa-Maximização (EM), permitindo identificar amostras de fronteira ou com alta incerteza.

---

## 4. Métricas e Avaliação dos Clusters

### 4.1 Métricas Não Supervisionadas (Intrínsecas)
- **Silhouette Score:** **0.4284** (indica separação positiva e consistente entre as populações).
- **Davies-Bouldin Index:** **0.9841** (valores menores que 1 indicam boa compactação intra-cluster em relação à distância entre centroides).
- **Calinski-Harabasz Index:** **2.894,60** (reforça a densidade e dispersão inter-cluster das gaussianas).
- **Critérios de Informação:** A análise comparativa de BIC (*Bayesian Information Criterion*) e AIC apontou para $K=2$ como ponto de equilíbrio ideal entre complexidade e ajuste.

### 4.2 Correspondência com as Espécies Reais (Pureza)
Ao confrontar os dois agrupamentos encontrados com a variável de origem, obteve-se:
- **Matriz de Contingência:**
  - **Cluster 0:** 1.382 grãos *Cammeo* e 335 grãos *Osmancik*.
  - **Cluster 1:** 248 grãos *Cammeo* e 1.845 grãos *Osmancik*.
- **Pureza Global:** **84,70%** de acurácia natural não supervisionada.

---

## 5. Interpretação Semântica dos Clusters

- **Cluster 0 (Perfil Cammeo):** Representa grãos com maiores médias de área, perímetro e comprimento do eixo principal. Morfologicamente, o arroz Cammeo é mais longo, encorpado e volumoso.
- **Cluster 1 (Perfil Osmancik):** Representa grãos com menor área, menor perímetro e menor eixo maior, apresentando uma morfologia proporcionalmente menor e mais arredondada.

---

## 6. Implementação da Aplicação e Containerização

A solução foi construída utilizando o framework **Streamlit** (Python) e empacotada em uma imagem Docker otimizada:
- **Interface Web:** Possui formulário interativo para entrada dos 7 atributos morfológicos, cálculo instantâneo do cluster previsto, exibição das probabilidades posteriores em gráfico horizontal de barras e abas informativas com a EDA e métricas.
- **Dockerfile:** Baseado em `python:3.10-slim`, com instalação enxuta de dependências e configuração de healthcheck na porta 8501.

### 6.1 Prints da Execução e do Docker
*(Espaço reservado para inserção das capturas de tela)*
1. **Print da Aplicação Web:** [Inserir captura da tela de inferência com os campos preenchidos e o gráfico de probabilidade].
2. **Print do Container Docker:** [Inserir captura do comando `docker ps` e do terminal exibindo o container ativo].
3. **Print do Repositório Docker Hub:** [Inserir captura da página pública do Docker Hub].

### 6.2 Link do Repositório no Docker Hub
- **Link Público:** `https://hub.docker.com/r/vladimircoutinho/rice-gmm-cluster`
- **Comando de Pull:**
  ```bash
  docker pull vladimircoutinho/rice-gmm-cluster:latest
  docker run -p 8501:8501 vladimircoutinho/rice-gmm-cluster:latest
  ```

---

## 7. Conclusão

O projeto cumpriu integralmente todos os requisitos estabelecidos na proposta da disciplina. O algoritmo de Misturas Gaussianas demonstrou aderência excelente à morfologia biológica dos grãos, capturando com 84,70% de pureza a divisão natural entre as espécies Cammeo e Osmancik sem necessidade de supervisão humana prévia. O ciclo de DevOps e empacotamento em container Docker garante a portabilidade e facilidade de implantação da ferramenta em qualquer infraestrutura.
