import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib

def run_pipeline():
    print("Iniciando pipeline de treinamento e avaliação GMM...")
    arff_path = "Rice_Cammeo_Osmancik.arff"
    
    # Leitura do ARFF
    columns = []
    records = []
    data_started = False
    with open(arff_path, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip()
            if not l or l.startswith("%"):
                continue
            if l.lower().startswith("@attribute"):
                parts = l.split()
                columns.append(parts[1])
            elif l.lower().startswith("@data"):
                data_started = True
            elif data_started:
                row = [x.strip() for x in l.split(",")]
                if len(row) == len(columns):
                    records.append(row)
                    
    df = pd.DataFrame(records, columns=columns)
    feature_cols = columns[:-1]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    print(f"Total de registros carregados: {len(df)}")
    
    # 1. Estatísticas descritivas
    stats = {}
    for c in feature_cols:
        stats[c] = {
            "mean": float(df[c].mean()),
            "std": float(df[c].std()),
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "median": float(df[c].median())
        }
        
    # 2. Normalização
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Modelagem GMM (K=2)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    gmm.fit(X_scaled)
    
    preds = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    
    # 4. Métricas Intrínsecas
    sil = float(silhouette_score(X_scaled, preds))
    db = float(davies_bouldin_score(X_scaled, preds))
    ch = float(calinski_harabasz_score(X_scaled, preds))
    bic = float(gmm.bic(X_scaled))
    aic = float(gmm.aic(X_scaled))
    
    # 5. Avaliação contra Ground Truth
    df["Cluster"] = preds
    contingency = pd.crosstab(df["Cluster"], df["Class"])
    print("Tabela de Contingência:")
    print(contingency)
    
    # Mapeamento do perfil de cluster
    cluster_mapping = {}
    for cl in [0, 1]:
        predom = df[df["Cluster"] == cl]["Class"].mode()[0]
        cluster_mapping[cl] = predom
        
    aligned_classes = [cluster_mapping[p] for p in preds]
    purity = float((df["Class"] == aligned_classes).mean())
    
    # 6. Salvar modelos e parâmetros
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(gmm, "gmm_model.joblib")
    
    meta = {
        "features": feature_cols,
        "feature_stats": stats,
        "cluster_mapping": {int(k): v for k, v in cluster_mapping.items()},
        "metrics": {
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "bic": bic,
            "aic": aic,
            "purity": purity
        },
        "contingency_matrix": {
            f"Cluster_{k}": {cls_name: int(count) for cls_name, count in row.items()}
            for k, row in contingency.iterrows()
        },
        "means": gmm.means_.tolist(),
        "weights": gmm.weights_.tolist()
    }
    
    with open("model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    print("Resultados:")
    print(f"- Silhouette: {sil:.4f}")
    print(f"- Davies-Bouldin: {db:.4f}")
    print(f"- Calinski-Harabasz: {ch:.4f}")
    print(f"- Pureza em relação às classes: {purity*100:.2f}%")
    print("Modelos 'scaler.joblib', 'gmm_model.joblib' e 'model_metadata.json' gerados com sucesso!")

if __name__ == "__main__":
    run_pipeline()
