
import hydra
from omegaconf import DictConfig
import logging
import os
import pandas as pd
from tqdm import tqdm
import mlflow
import ast

# Imports de Haystack y Componentes
from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.embedders import SentenceTransformersTextEmbedder
from datasets import load_dataset

# --- Configuración del Logging ---
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_model_output(output: str):
    """Parsea la salida del modelo, esperando una lista de Python."""
    try:
        predicted_labels = ast.literal_eval(output.strip())
        return predicted_labels if isinstance(predicted_labels, list) else [str(predicted_labels)]
    except (ValueError, SyntaxError, NameError):
        log.warning(f"No se pudo parsear la salida: '{output}'. Se devuelve una lista vacía.")
        return []

def load_prompt_template(path: str):
    """Carga la plantilla de prompt desde una ruta absoluta o relativa al CWD original."""
    full_path = os.path.join(hydra.utils.get_original_cwd(), path)
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Script principal para evaluar un pipeline de RAG Híbrido con Re-ranking.
    """
    log.info("--- Iniciando Evaluación de RAG Híbrido con Re-ranking ---")

    # --- 1. Cargar Configuración desde Hydra ---
    task_cfg = cfg.task
    db_cfg = cfg.db.hybrid
    strategy_cfg = cfg.strategy.hybrid_rag
    prompt_path = cfg.prompts.prompt
    llm_model = cfg.models.llm
    task_name_key = task_cfg.name.split(' - ')[-1].lower()

    # --- 2. Configuración de MLflow ---
    mlflow.set_tracking_uri(cfg.run_config.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.run_config.mlflow_experiment_name)
    run_name = f"HybridRAG_{task_name_key}_{llm_model.replace('/', '_')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log de parámetros usando las rutas de config correctas
        params_to_log = {
            "task_name": task_cfg.name,
            "llm_model": llm_model,
            "dense_model": db_cfg.embedding_model,
            "sparse_model": strategy_cfg.sparse_embedding_model,
            "ranker_model": strategy_cfg.ranker_model,
            "prompt_path": prompt_path,
            "top_k_retriever": strategy_cfg.top_k_retriever,
            "top_k_ranker": strategy_cfg.top_k_ranker,
            "worker_type": "HybridRAG_Haystack"
        }
        mlflow.log_params(params_to_log)
        log.info(f"Parámetros de la ejecución: {params_to_log}")

        # --- 3. Cargar Componentes ---
        storage_dir = db_cfg.storage_dir.format(
            task_key=task_name_key,
            embed_short=db_cfg.embed_model_name_short
        )
        full_storage_path = os.path.join(hydra.utils.get_original_cwd(), storage_dir)
        
        document_store = QdrantDocumentStore(
            path=full_storage_path,
            index=task_name_key,
            use_sparse_embeddings=True,
            embedding_dim=db_cfg.embedding_dim
        )
        
        test_dataset = load_dataset('json', data_files=os.path.join(hydra.utils.get_original_cwd(), task_cfg.data_path), split='train').filter(lambda x: x['split'] == 'test')
        prompt_template = load_prompt_template(prompt_path)

        # --- 4. Construir Pipeline Híbrido ---
        hybrid_pipeline = Pipeline()
        hybrid_pipeline.add_component("sparse_embedder", FastembedSparseTextEmbedder(model=strategy_cfg.sparse_embedding_model))
        hybrid_pipeline.add_component("dense_embedder", SentenceTransformersTextEmbedder(model=db_cfg.embedding_model))
        hybrid_pipeline.add_component("retriever", QdrantHybridRetriever(document_store=document_store, top_k=strategy_cfg.top_k_retriever))
        hybrid_pipeline.add_component("ranker", SentenceTransformersSimilarityRanker(model=strategy_cfg.ranker_model, top_k=strategy_cfg.top_k_ranker))
        
        prompt_builder = PromptBuilder(template=prompt_template)
        hybrid_pipeline.add_component("prompt_builder", prompt_builder)
        hybrid_pipeline.add_component("llm", OllamaGenerator(model=llm_model, timeout=3600, url=cfg.run_config.base_url_model))

        # Conectar componentes
        hybrid_pipeline.connect("sparse_embedder.sparse_embedding", "retriever.query_sparse_embedding")
        hybrid_pipeline.connect("dense_embedder.embedding", "retriever.query_embedding")
        hybrid_pipeline.connect("retriever.documents", "ranker.documents")
        hybrid_pipeline.connect("ranker.documents", "prompt_builder.documents")
        hybrid_pipeline.connect("prompt_builder", "llm")

        # --- 5. Bucle de Evaluación ---
        true_labels, pred_labels = [], []
        options_desc = "\n".join([f"- {key}: {desc}" for key, desc in task_cfg.options.items()])
        possible_options = list(task_cfg.options.keys())

        for item in tqdm(test_dataset, desc=f"Evaluando pipeline híbrido en '{task_name_key}'"):
            query_text = item['text']
            
            result = hybrid_pipeline.run({
                "dense_embedder": {"text": query_text},
                "sparse_embedder": {"text": query_text},
                "ranker": {"query": query_text},
                "prompt_builder": {"query": query_text, "template_variables": {"options_descriptions": options_desc}}
            })
            
            response_text = result["llm"]["replies"][0]
            detected = parse_model_output(response_text)

            true_labels.append(item['human_readable_labels'])
            pred_labels.append(detected)

        # --- 6. Calcular y Registrar Métricas ---
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import f1_score

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

        log.info("--- Evaluación de RAG Híbrido Finalizada ---")

if __name__ == "__main__":
    main()

