import hydra
from omegaconf import DictConfig
import ast
import mlflow
import logging
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import asyncio
import re
import os

# Imports de Scikit-learn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

# La intercepción de Ollama se movió aquí para asegurar que se configure antes de usar GraphRAG
from interceptor import install_ollama_interceptor
interceptor = install_ollama_interceptor()

import graphrag.api as api
from graphrag.config.load_config import load_config

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_test_data(task_cfg: DictConfig):
    """Carga solo el split de 'test' del dataset."""
    data_path = os.path.join(hydra.utils.get_original_cwd(), task_cfg.data_path)
    dataset = load_dataset('json', data_files=data_path, split='train')
    test_ds = dataset.filter(lambda x: x['split'] == 'test')
    log.info(f"Cargados {len(test_ds)} elementos del test set para evaluación.")
    return test_ds

def parse_llm_output(output: str, possible_labels: list):
    """Parsea la salida del LLM, buscando una lista de Python o etiquetas conocidas."""
    try:
        predicted_labels = ast.literal_eval(output.strip())
        if isinstance(predicted_labels, list):
            return [label for label in predicted_labels if label in possible_labels]
    except (ValueError, SyntaxError, NameError):
        log.warning(f"No se pudo parsear la salida con ast: '{output}'. Buscando etiquetas conocidas en el texto.")
    
    matches = []
    for label in possible_labels:
        if re.search(rf'\b{re.escape(label)}\b', output):
            matches.append(label)
    return matches

async def run_graphrag_classification(cfg: DictConfig):
    """Ejecuta una única corrida de evaluación usando un grafo de GraphRAG."""
    # --- 1. Cargar Configuración --- 
    task_cfg = cfg.task
    strategy_cfg = cfg.strategy.graphrag
    task_name_key = task_cfg.name.split(' - ')[-1].lower()

    # --- 2. Configuración de MLflow ---
    mlflow.set_tracking_uri(cfg.run_config.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.run_config.mlflow_experiment_name)
    run_name = f"GraphRAG_{task_name_key}_{cfg.models.llm}"
    
    with mlflow.start_run(run_name=run_name) as run:
        if "parent_run_id" in cfg.run_config and cfg.run_config.parent_run_id != "placeholder":
            mlflow.set_tag("mlflow.parentRunId", cfg.run_config.parent_run_id)

        # Logueo de parámetros
        params_to_log = {
            "task_name": task_cfg.name,
            "model_name": cfg.models.llm,
            "embedding_model": cfg.db.naive.embedding_model, # Asumimos que el embedder es el naive
            "worker_type": "GraphRAG_LocalSearch",
            "graph_project_path": strategy_cfg.project_path
        }
        mlflow.log_params(params_to_log)

        # --- 3. Cargar Artefactos de GraphRAG ---
        project_dir = os.path.join(hydra.utils.get_original_cwd(), strategy_cfg.project_path)
        log.info(f"Cargando artefactos del grafo desde: {project_dir}")
        
        # Carga la configuración de GraphRAG. La librería busca automáticamente
        # los archivos settings.yaml y .env dentro del root_dir.
        # Se debe pasar un objeto Path, no un string.
        graphrag_config = load_config(root_dir=Path(project_dir))
                
        output_dir = os.path.join(project_dir, "output_definitivo")
        entities = pd.read_parquet(os.path.join(output_dir, "entities.parquet"))
        communities = pd.read_parquet(os.path.join(output_dir, "communities.parquet"))
        community_reports = pd.read_parquet(os.path.join(output_dir, "community_reports.parquet"))
        text_units = pd.read_parquet(os.path.join(output_dir, "text_units.parquet"))
        relationships = pd.read_parquet(os.path.join(output_dir, "relationships.parquet"))
        
        # --- 4. Cargar Datos de Evaluación ---
        test_dataset = load_test_data(task_cfg)
        possible_options = list(task_cfg.options.keys())
        
        # --- 5. Bucle de Inferencia Asíncrono ---
        true_labels, pred_labels = [], []
        
        for item in test_dataset:
            query_text = item['text']
            prompt = f"Clasifica esta clausula: {query_text}"

            result, _ = await api.local_search(
                config=graphrag_config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                query=prompt,
                community_level=1,
                response_type="Multiple Paragraphs",
                covariates=None,
            )
            
            detected_labels = parse_llm_output(result, possible_options)
            log.info(f"Respuesta LLM: {result} -> Detectado: {detected_labels}")
            
            true_labels.append(item['human_readable_labels'])
            pred_labels.append(detected_labels)

        # --- 6. Evaluación y Métricas ---
        log.info("--- Evaluación Final ---")
        mlb = MultiLabelBinarizer(classes=possible_options)
        true_bin = mlb.fit_transform(true_labels)
        pred_bin = mlb.transform(pred_labels)
        
        metrics = {
            "micro_f1": f1_score(true_bin, pred_bin, average='micro', zero_division=0),
            "macro_f1": f1_score(true_bin, pred_bin, average='macro', zero_division=0)
        }
        mlflow.log_metrics(metrics)
        log.info(f"Resultados GraphRAG -> {metrics}")

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Función síncrona que inicia el bucle de eventos de asyncio."""
    asyncio.run(run_graphrag_classification(cfg))
    
if __name__ == "__main__":
    main()