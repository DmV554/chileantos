
import hydra
from omegaconf import DictConfig
import logging
import os
import pandas as pd
from tqdm import tqdm
import mlflow
import ast
import sys

# --- Inicio: Modificación del Path para Imports Absolutos ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fin: Modificación del Path ---

from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from src.utils.data_loader import load_test_set
from src.pipelines.common.base_worker import BaseExperimentWorker

log = logging.getLogger(__name__)

# --- Funciones de utilidad específicas para este worker ---

def parse_model_output(output: str):
    try:
        return ast.literal_eval(output.strip())
    except (ValueError, SyntaxError):
        return []

def load_prompt_template(path: str):
    full_path = os.path.join(hydra.utils.get_original_cwd(), path)
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()

class HybridRAGWorker(BaseExperimentWorker):
    """Worker para experimentos de RAG Híbrido con Re-ranking."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.db_cfg = self.cfg.db.hybrid
        self.strategy_cfg = self.cfg.strategy.hybrid_rag

    def _get_run_name(self) -> str:
        return f"HybridRAG_{self.task_name_key}_{self.cfg.models.llm}"

    def _get_parameters_to_log(self) -> dict:
        return {
            "task_name": self.task_cfg.name,
            "llm_model": self.cfg.models.llm,
            "dense_model": self.db_cfg.embedding_model,
            "sparse_model": self.strategy_cfg.sparse_embedding_model,
            "ranker_model": self.strategy_cfg.ranker_model,
            "prompt_path": self.cfg.prompts.prompt,
            "top_k_retriever": self.strategy_cfg.top_k_retriever,
            "top_k_ranker": self.strategy_cfg.top_k_ranker,
            "worker_type": "HybridRAG_Haystack"
        }

    def _execute_task(self) -> None:
        storage_dir = self.db_cfg.storage_dir.format(
            task_key=self.task_name_key,
            embed_short=self.db_cfg.embed_model_name_short
        )
        full_storage_path = os.path.join(hydra.utils.get_original_cwd(), storage_dir)
        
        document_store = QdrantDocumentStore(
            path=full_storage_path,
            index=self.task_name_key,
            use_sparse_embeddings=True,
            embedding_dim=self.db_cfg.embedding_dim
        )
        
        test_dataset = load_test_set(self.task_cfg)
        pipeline = self._build_pipeline(document_store)
        
        true_labels, pred_labels = self._run_inference_loop(pipeline, test_dataset)
        
        self._evaluate_and_log(true_labels, pred_labels, test_dataset)

    def _build_pipeline(self, document_store):
        log.info("Construyendo el pipeline de RAG Híbrido...")
        prompt_template = load_prompt_template(self.cfg.prompts.prompt)

        hybrid_pipeline = Pipeline()
        hybrid_pipeline.add_component("sparse_embedder", FastembedSparseTextEmbedder(model=self.strategy_cfg.sparse_embedding_model))
        hybrid_pipeline.add_component("dense_embedder", SentenceTransformersTextEmbedder(model=self.db_cfg.embedding_model))
        hybrid_pipeline.add_component("retriever", QdrantHybridRetriever(document_store=document_store, top_k=self.strategy_cfg.top_k_retriever))
        hybrid_pipeline.add_component("ranker", SentenceTransformersSimilarityRanker(model=self.strategy_cfg.ranker_model, top_k=self.strategy_cfg.top_k_ranker))
        hybrid_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
        hybrid_pipeline.add_component("llm", OllamaGenerator(model=self.cfg.models.llm, timeout=3600, url=self.run_config.base_url_model))

        hybrid_pipeline.connect("sparse_embedder.sparse_embedding", "retriever.query_sparse_embedding")
        hybrid_pipeline.connect("dense_embedder.embedding", "retriever.query_embedding")
        hybrid_pipeline.connect("retriever.documents", "ranker.documents")
        hybrid_pipeline.connect("ranker.documents", "prompt_builder.documents")
        hybrid_pipeline.connect("prompt_builder", "llm")
        return hybrid_pipeline

    def _run_inference_loop(self, pipeline: Pipeline, test_ds):
        true_labels, pred_labels = [], []
        options_desc = "\n".join([f"- {key}: {desc}" for key, desc in self.task_cfg.options.items()])

        for item in tqdm(test_ds, desc=f"Evaluando pipeline híbrido en '{self.task_name_key}'"):
            query_text = item['text']
            result = pipeline.run({
                "dense_embedder": {"text": query_text},
                "sparse_embedder": {"text": query_text},
                "ranker": {"query": query_text},
                "prompt_builder": {"query": query_text, "template_variables": {"options_descriptions": options_desc}}
            })
            
            response_text = result["llm"]["replies"][0]
            detected = parse_model_output(response_text)

            true_labels.append(item['human_readable_labels'])
            pred_labels.append(detected)
        return true_labels, pred_labels

    def _evaluate_and_log(self, true_labels, pred_labels, test_dataset):
        possible_options = list(self.task_cfg.options.keys())
        mlb = MultiLabelBinarizer(classes=possible_options)
        true_bin = mlb.fit_transform(true_labels)
        pred_bin = mlb.transform(pred_labels)
        
        metrics = {
            "micro_f1": f1_score(true_bin, pred_bin, average='micro', zero_division=0),
            "macro_f1": f1_score(true_bin, pred_bin, average='macro', zero_division=0)
        }
        mlflow.log_metrics(metrics)
        log.info(f"Resultados Finales (Clasificación): {metrics}")

        df_results = pd.DataFrame({
            'text': [item['text'] for item in test_dataset],
            'true_labels': true_labels,
            'predicted_labels': pred_labels
        })
        df_results.to_csv("hybrid_rag_evaluation_details.csv", index=False)
        mlflow.log_artifact("hybrid_rag_evaluation_details.csv")

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    worker = HybridRAGWorker(cfg)
    worker.run()

if __name__ == "__main__":
    main()


