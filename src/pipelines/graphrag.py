import hydra
from omegaconf import DictConfig
import ast
import mlflow
import logging
import pandas as pd
from pathlib import Path
import asyncio
import re
import os
import sys

# --- Inicio: Modificación del Path para Imports Absolutos ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fin: Modificación del Path ---

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from src.utils.data_loader import load_test_set
from src.pipelines.common.base_worker import BaseExperimentWorker

from interceptor import install_ollama_interceptor
interceptor = install_ollama_interceptor()

import graphrag.api as api
from graphrag.config.load_config import load_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_llm_output(output: str, possible_labels: list):
    try:
        predicted_labels = ast.literal_eval(output.strip())
        if isinstance(predicted_labels, list):
            return [label for label in predicted_labels if label in possible_labels]
    except (ValueError, SyntaxError, NameError):
        log.warning(f"No se pudo parsear la salida con ast: '{output}'. Buscando etiquetas conocidas.")
    
    matches = []
    for label in possible_labels:
        if re.search(rf'\b{re.escape(label)}\b', output):
            matches.append(label)
    return matches

class GraphRAGWorker(BaseExperimentWorker):
    """Worker para experimentos con GraphRAG."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.strategy_cfg = self.cfg.strategy.graphrag
        self.project_dir = os.path.join(hydra.utils.get_original_cwd(), self.strategy_cfg.project_path)
        self.graphrag_config = load_config(root_dir=Path(self.project_dir))
        self.actual_model_name = self.graphrag_config.get("models", {}).get("default_chat_model", {}).get("model", "unknown")

    def _get_run_name(self) -> str:
        return f"GraphRAG_{self.task_name_key}_{self.actual_model_name}"

    def _get_parameters_to_log(self) -> dict:
        return {
            "task_name": self.task_cfg.name,
            "model_name": self.actual_model_name,
            "embedding_model": self.graphrag_config.get("models", {}).get("default_embedding_model", {}).get("model", "unknown"),
            "worker_type": "GraphRAG_LocalSearch",
            "graph_project_path": self.strategy_cfg.project_path
        }

    def _execute_task(self) -> None:
        # El bucle de eventos de asyncio debe ser manejado aquí
        asyncio.run(self._execute_async_task())

    async def _execute_async_task(self):
        log.info(f"Cargando artefactos del grafo desde: {self.project_dir}")
        output_dir = os.path.join(self.project_dir, "output_definitivo")
        entities = pd.read_parquet(os.path.join(output_dir, "entities.parquet"))
        communities = pd.read_parquet(os.path.join(output_dir, "communities.parquet"))
        community_reports = pd.read_parquet(os.path.join(output_dir, "community_reports.parquet"))
        text_units = pd.read_parquet(os.path.join(output_dir, "text_units.parquet"))
        relationships = pd.read_parquet(os.path.join(output_dir, "relationships.parquet"))
        
        test_dataset = load_test_set(self.task_cfg)
        
        true_labels, pred_labels = await self._run_inference_loop(
            entities, communities, community_reports, text_units, relationships, test_dataset
        )
        
        self._evaluate_and_log(true_labels, pred_labels)

    async def _run_inference_loop(self, entities, communities, community_reports, text_units, relationships, test_dataset):
        true_labels, pred_labels = [], []
        possible_options = list(self.task_cfg.options.keys())

        for item in test_dataset:
            query_text = item['text']
            prompt = f"Clasifica esta clausula: {query_text}"

            result, _ = await api.local_search(
                config=self.graphrag_config,
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
        return true_labels, pred_labels

    def _evaluate_and_log(self, true_labels, pred_labels):
        log.info("--- Evaluación Final ---")
        possible_options = list(self.task_cfg.options.keys())
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
    worker = GraphRAGWorker(cfg)
    worker.run()
    
if __name__ == "__main__":
    main()