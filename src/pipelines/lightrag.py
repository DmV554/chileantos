import hydra
from omegaconf import DictConfig
import logging
import os
import sys
import asyncio
from tqdm import tqdm
import mlflow

# --- Modificación del Path para Imports Absolutos ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---

from lightrag.lightrag import LightRAG, QueryParam
from lightrag.components.model_client import ollama_model_complete, ollama_embed, EmbeddingFunc
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from src.utils.data_loader import load_test_set
from src.pipelines.common.base_worker import BaseExperimentWorker
from src.utils.parsing import parse_llm_output

log = logging.getLogger(__name__)

class LightRAGWorker(BaseExperimentWorker):
    """Worker para experimentos con la librería LightRAG."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.strategy_cfg = self.cfg.strategy.lightrag
        self.model_cfg = self.cfg.models
        self.db_cfg = self.cfg.db.naive

    def _get_run_name(self) -> str:
        # El nombre del modelo real se leerá de forma asíncrona más tarde
        return f"LightRAG_{self.task_name_key}_{self.model_cfg.llm}"

    def _get_parameters_to_log(self) -> dict:
        # Parámetros iniciales, el nombre real del modelo se puede añadir después
        return {
            "task_name": self.task_cfg.name,
            "model_name": self.model_cfg.llm, # Modelo de síntesis
            "embedding_model": self.db_cfg.embedding_model,
            "worker_type": "LightRAG",
            "index_path": self.strategy_cfg.index_path
        }

    def _execute_task(self) -> None:
        """Punto de entrada que ejecuta la lógica asíncrona principal."""
        asyncio.run(self._execute_async_task())

    async def _execute_async_task(self):
        """Contiene la lógica asíncrona para configurar y ejecutar LightRAG."""
        workspace_path = os.path.join(hydra.utils.get_original_cwd(), self.strategy_cfg.index_path)
        
        async def llm_wrapper(prompt, **kwargs):
            return await ollama_model_complete(prompt, model=self.model_cfg.llm, **kwargs)

        async def embedding_wrapper(texts: list[str]):
            return await ollama_embed(texts, model=self.db_cfg.embedding_model)

        rag_app = None
        try:
            rag_app = LightRAG(
                working_dir=workspace_path,
                llm_model_func=llm_wrapper,
                embedding_func=EmbeddingFunc(
                    embedding_dim=1024, # Ajustar según el modelo
                    func=embedding_wrapper
                )
            )
            await rag_app.initialize_storages()
            # await initialize_pipeline_status()

            test_ds = load_test_set(self.task_cfg)
            true_labels, pred_labels = await self._run_inference_loop(rag_app, test_ds)
            self._evaluate_and_log(true_labels, pred_labels)

        finally:
            if rag_app:
                await rag_app.finalize_storages()

    async def _run_inference_loop(self, rag_app: LightRAG, test_ds):
        true_labels, pred_labels = [], []
        possible_options = list(self.task_cfg.options.keys())

        log.info(f"Iniciando inferencia con LightRAG en {len(test_ds)} cláusulas...")
        for item in tqdm(test_ds, desc=f"Evaluando LightRAG en '{self.task_name_key}'"):
            query_text = item['text']
            
            response = await rag_app.query(
                query=query_text, 
                param=QueryParam(mode="hybrid")
            )
            
            response_text = response.get("answer", "")
            detected_labels = parse_llm_output(response_text, possible_options)
            
            true_labels.append(item['human_readable_labels'])
            pred_labels.append(detected_labels)
            
        return true_labels, pred_labels

    def _evaluate_and_log(self, true_labels, pred_labels):
        possible_options = list(self.task_cfg.options.keys())
        mlb = MultiLabelBinarizer(classes=possible_options)
        true_bin = mlb.fit_transform(true_labels)
        pred_bin = mlb.transform(pred_labels)
        
        metrics = { 
            "micro_f1": f1_score(true_bin, pred_bin, average='micro', zero_division=0),
            "macro_f1": f1_score(true_bin, pred_bin, average='macro', zero_division=0)
        }
        mlflow.log_metrics(metrics)
        log.info(f"Resultados LightRAG -> {metrics}")

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    worker = LightRAGWorker(cfg)
    worker.run()

if __name__ == "__main__":
    main()