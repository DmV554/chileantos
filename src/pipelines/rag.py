import hydra
from omegaconf import DictConfig
import ast
import mlflow
import logging
import os
from tqdm import tqdm
import sys

# --- Inicio: Modificación del Path para Imports Absolutos ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fin: Modificación del Path ---

from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from src.utils.data_loader import load_test_set
from src.pipelines.common.base_worker import BaseExperimentWorker

log = logging.getLogger(__name__)

# --- Funciones de utilidad específicas para este worker ---

def load_haystack_knowledge_base(storage_path: str, collection_name: str):
    full_storage_path = os.path.join(hydra.utils.get_original_cwd(), storage_path)
    if not os.path.exists(full_storage_path):
        raise FileNotFoundError(f"El directorio de la KB de ChromaDB no fue encontrado: {full_storage_path}")
    
    log.info(f"Cargando KB de ChromaDB desde: {full_storage_path} con la colección '{collection_name}'")
    return ChromaDocumentStore(collection_name=collection_name, persist_path=full_storage_path)

def parse_model_output(output: str):
    try:
        return ast.literal_eval(output.strip())
    except (ValueError, SyntaxError):
        return []

def load_prompt_template(path: str):
    full_path = os.path.join(hydra.utils.get_original_cwd(), path)
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()

class RAGWorker(BaseExperimentWorker):
    """Worker para experimentos de RAG Naive."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.db_cfg = self.cfg.db.naive
        self.strategy_cfg = self.cfg.strategy.naive_rag

    def _get_run_name(self) -> str:
        return f"RAG_{self.task_name_key}_{self.cfg.models.llm}_k{self.strategy_cfg.top_k}"

    def _get_parameters_to_log(self) -> dict:
        return {
            "task_name": self.task_cfg.name,
            "model_name": self.cfg.models.llm,
            "embedding_model": self.db_cfg.embedding_model,
            "similarity_top_k": self.strategy_cfg.top_k,
            "worker_type": "RAG_Haystack"
        }

    def _execute_task(self) -> None:
        storage_dir = self.db_cfg.storage_dir.format(
            task_key=self.task_name_key, 
            embed_short=self.db_cfg.embed_model_name_short
        )
        document_store = load_haystack_knowledge_base(storage_dir, collection_name=self.task_name_key)
        test_ds = load_test_set(self.task_cfg)
        
        pipeline = self._build_pipeline(document_store)
        
        true_labels, pred_labels = self._run_inference_loop(pipeline, test_ds)
        
        self._evaluate_and_log(true_labels, pred_labels)

    def _build_pipeline(self, document_store):
        log.info("Construyendo el pipeline de RAG con Haystack...")
        text_embedder = SentenceTransformersTextEmbedder(model=self.db_cfg.embedding_model)
        retriever = ChromaEmbeddingRetriever(document_store=document_store)
        prompt_template = load_prompt_template(self.cfg.prompts.prompt)
        prompt_builder = PromptBuilder(template=prompt_template)
        llm = OllamaGenerator(model=self.cfg.models.llm, timeout=3600, url=self.run_config.base_url_model)

        rag_pipeline = Pipeline()
        rag_pipeline.add_component("query_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)
        
        rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")
        return rag_pipeline

    def _run_inference_loop(self, pipeline: Pipeline, test_ds):
        true_labels, pred_labels = [], []
        options_desc = "\n".join([f"- {key}: {desc}" for key, desc in self.task_cfg.options.items()])
        
        for item in tqdm(test_ds, desc=f"Evaluando RAG en tarea '{self.task_name_key}'"):
            query_text = item['text']
            pipeline_output = pipeline.run({
                "query_embedder": {"text": query_text},
                "retriever": {"top_k": self.strategy_cfg.top_k},
                "prompt_builder": {"query": query_text, "template_variables": {"options_descriptions": options_desc}},
            })
            
            response_text = pipeline_output["llm"]["replies"][0]
            detected_labels = parse_model_output(response_text)
            
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
        log.info(f"Resultados RAG con Haystack -> {metrics}")

@hydra.main(config_path="../../config", config_name="config",  version_base=None)
def main(cfg: DictConfig):
    worker = RAGWorker(cfg)
    worker.run()

if __name__ == "__main__":
    main()