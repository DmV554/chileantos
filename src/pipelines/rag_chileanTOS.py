import hydra
from omegaconf import DictConfig
import ast
import mlflow
import logging
from datasets import load_dataset
import os
from tqdm import tqdm

# Imports de Haystack
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

# Imports de Scikit-learn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)

def load_haystack_knowledge_base(storage_path: str, collection_name: str):
    """Carga un DocumentStore de ChromaDB persistente."""
    full_storage_path = os.path.join(hydra.utils.get_original_cwd(), storage_path)
    if not os.path.exists(full_storage_path):
        raise FileNotFoundError(f"El directorio de la KB de ChromaDB no fue encontrado: {full_storage_path}")
    
    log.info(f"Cargando KB de ChromaDB desde: {full_storage_path} con la colección '{collection_name}'")
    document_store = ChromaDocumentStore(
        collection_name=collection_name,
        persist_path=full_storage_path
    )
    return document_store

def load_test_data(task_cfg: DictConfig):
    """Carga solo el split de 'test' para la evaluación."""
    data_path = os.path.join(hydra.utils.get_original_cwd(), task_cfg.data_path)
    dataset = load_dataset('json', data_files=data_path, split='train')
    test_ds = dataset.filter(lambda x: x['split'] == 'test')
    log.info(f"Cargados {len(test_ds)} elementos del test set para evaluación.")
    return test_ds

def parse_model_output(output: str):
    """Parsea la salida del modelo, manejando respuestas con y sin razonamiento."""
    closing_tag = "</think>"
    clean_output = output.strip()
    if closing_tag in clean_output:
        try:
            clean_output = clean_output.split(closing_tag, 1)[1].strip()
        except IndexError: return []
    try:
        predicted_labels = ast.literal_eval(clean_output)
        return predicted_labels if isinstance(predicted_labels, list) else [str(predicted_labels)]
    except (ValueError, SyntaxError, NameError):
        return []

def load_prompt_template(path: str):
    """Carga la plantilla de prompt desde una ruta absoluta o relativa al CWD original."""
    # Hydra cambia el directorio de trabajo, así que usamos get_original_cwd() para rutas relativas
    full_path = os.path.join(hydra.utils.get_original_cwd(), path)
    with open(full_path, 'r', encoding='utf-8') as f:
        return f.read()


@hydra.main(config_path="../../config", config_name="config",  version_base=None)
def main(cfg: DictConfig):
    # --- 1. Carga de Configuración --- 
    # La configuración de la tarea se carga directamente en cfg.task
    task_cfg = cfg.task
    # Derivamos el nombre de la tarea para usarlo como clave (ej: 'illegal')
    task_name_key = task_cfg.name.split(' - ')[-1].lower()

    # --- 2. Configuración de MLflow --- 
    mlflow.set_tracking_uri(cfg.run_config.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.run_config.mlflow_experiment_name)
    
    # Construye el nombre del run con las rutas de config correctas
    run_name = f"RAG_{task_name_key}_{cfg.models.llm}_k{cfg.strategy.naive_rag.top_k}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Usa la ruta correcta para el parent_run_id
        if "parent_run_id" in cfg.run_config and cfg.run_config.parent_run_id != "placeholder":
            mlflow.set_tag("mlflow.parentRunId", cfg.run_config.parent_run_id)

        # Log de parámetros usando las rutas de config correctas y limpias
        params_to_log = { 
            "task_name": task_cfg.name,
            "model_name": cfg.models.llm,
            "embedding_model": cfg.db.naive.embedding_model,
            "similarity_top_k": cfg.strategy.naive_rag.top_k,
            "worker_type": "RAG_Haystack"
        }
        mlflow.log_params(params_to_log)

        # --- 3. Pipeline de Haystack --- 
        # Cargar la KB de ChromaDB usando la configuración de db.naive
        storage_dir = cfg.db.naive.storage_dir.format(
            task_key=task_name_key, 
            embed_short=cfg.db.naive.embed_model_name_short
        )
        document_store = load_haystack_knowledge_base(storage_dir, collection_name=task_name_key)

        # Cargar el dataset para las consultas
        query_dataset = load_test_data(task_cfg)

        # Inicializar los componentes del pipeline de RAG
        log.info("Construyendo el pipeline de RAG con Haystack...")
        text_embedder = SentenceTransformersTextEmbedder(model=cfg.db.naive.embedding_model)
        retriever = ChromaEmbeddingRetriever(document_store=document_store)
        
        # Cargar el prompt desde la sección central de prompts
        prompt_template = load_prompt_template(cfg.prompts.prompt)
        prompt_builder = PromptBuilder(template=prompt_template)
        
        llm = OllamaGenerator(
            model=cfg.models.llm,
            timeout=3600,
            url=cfg.run_config.base_url_model
        )

        # Construir el pipeline conectando los componentes
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("query_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)
        
        rag_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        # --- 4. Bucle de Inferencia --- 
        true_labels, pred_labels = [], []
        options_desc = "\n".join([f"- {key}: {desc}" for key, desc in task_cfg.options.items()])
        
        for item in tqdm(query_dataset, desc=f"Evaluando RAG en tarea '{task_name_key}'"):
            query_text = item['text']

            # Ejecutar el pipeline usando la config de la estrategia
            pipeline_output = rag_pipeline.run({
                "query_embedder": {"text": query_text},
                "retriever": {"top_k": cfg.strategy.naive_rag.top_k},
                "prompt_builder": {"query": query_text, "template_variables": {"options_descriptions": options_desc}},
            })
            
            response_text = pipeline_output["llm"]["replies"][0]
            detected_labels = parse_model_output(response_text)
            
            true_labels.append(item['human_readable_labels'])
            pred_labels.append(detected_labels)
        
        # --- 5. Evaluación y Métricas --- 
        possible_options = list(task_cfg.options.keys())
        mlb = MultiLabelBinarizer(classes=possible_options)
        true_bin = mlb.fit_transform(true_labels)
        pred_bin = mlb.transform(pred_labels)
        
        metrics = { 
            "micro_f1": f1_score(true_bin, pred_bin, average='micro', zero_division=0),
            "macro_f1": f1_score(true_bin, pred_bin, average='macro', zero_division=0)
        }
        mlflow.log_metrics(metrics)
        log.info(f"Resultados RAG con Haystack -> {metrics}")

if __name__ == "__main__":
    main()
