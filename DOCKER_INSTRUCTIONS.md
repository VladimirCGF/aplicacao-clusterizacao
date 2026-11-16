# Instruções para Build, Execução e Publicação no Docker Hub

Este guia detalha o passo a passo completo para gerar a imagem Docker da aplicação de clusterização GMM e publicá-la no Docker Hub.

---

## 1. Pré-Requisitos
- Ter o **Docker Desktop** instalado e em execução na máquina;
- Possuir uma conta cadastrada no [Docker Hub](https://hub.docker.com/).

---

## 2. Passo a Passo no Terminal (PowerShell ou Bash)

Navegue até a pasta do projeto:
```powershell
cd "c:\Users\vladi\OneDrive\Documentos\facu\Mdados\TRABALHO_16"
```

### Passo 2.1: Autenticar no Docker Hub
```powershell
docker login
```
*Insira seu nome de usuário e senha ou token de acesso pessoal do Docker Hub.*

---

### Passo 2.2: Construir a Imagem Docker Localmente
Substitua `<SEU_USUARIO_DOCKERHUB>` pelo seu login do Docker Hub (exemplo: `vladmirg/rice-gmm-cluster`):

```powershell
docker build -t <SEU_USUARIO_DOCKERHUB>/rice-gmm-cluster:latest .
```

---

### Passo 2.3: Testar e Rodar o Container Localmente
Execute o comando abaixo para iniciar o container:
```powershell
docker run -d -p 8501:8501 --name rice_clustering_app <SEU_USUARIO_DOCKERHUB>/rice-gmm-cluster:latest
```

Verifique se o container está rodando:
```powershell
docker ps
```

Abra o navegador no endereço:
👉 **[http://localhost:8501](http://localhost:8501)**

---

### Passo 2.4: Publicar a Imagem no Docker Hub
Envie a imagem construída para o seu repositório público:
```powershell
docker push <SEU_USUARIO_DOCKERHUB>/rice-gmm-cluster:latest
```

---

## 3. Captura dos Prints Obrigatórios para o Relatório Técnico

Para o relatório em PDF exigido pelo professor, lembre-se de tirar print das seguintes telas:
1. **Terminal com `docker ps`** mostrando o container em execução com a porta 8501 mapeada;
2. **Interface Web Streamlit no navegador** com a aba de inferência preenchida e resultado exibido;
3. **Página pública do repositório no Docker Hub** comprovando a disponibilização pública da tag `latest`.
