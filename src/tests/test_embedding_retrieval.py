import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

# Imports de Haystack
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder

# --- Configuración del Logging ---
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_knowledge_base(storage_path: str, collection_name: str):
    """Carga un DocumentStore de ChromaDB existente."""
    if not os.path.exists(storage_path):
        log.error(f"El directorio de la base de datos vectorial no existe: {storage_path}")
        raise FileNotFoundError(f"No se encontró el directorio de ChromaDB: {storage_path}")
    
    log.info(f"Cargando la base de datos vectorial desde: {storage_path} (Colección: {collection_name})")
    return ChromaDocumentStore(
        collection_name=collection_name,
        persist_path=storage_path
    )

@hydra.main(config_path=".", config_name="config_chile", version_base=None)
def main(cfg: DictConfig):
    """
    Script principal para probar la recuperación de embeddings.
    """
    log.info("--- Iniciando Test de Recuperación de Embeddings ---")
    
    # --- Validar configuración ---
    if "test_retrieval" not in cfg:
        log.error("La configuración 'test_retrieval' no se encuentra en el archivo de configuración.")
        log.info("Por favor, añade una sección 'test_retrieval' a tu config_chile.yaml con 'task_key', 'query_clause', 'top_k' y 'embedding_model'.")
        return

    test_cfg = cfg.test_retrieval
    task_key = test_cfg.get("task_key")
    query_clause = test_cfg.get("query_clause")
    top_k = test_cfg.get("top_k", 5)
    embedding_model = test_cfg.get("embedding_model")

    if not all([task_key, query_clause, embedding_model]):
        log.error("Faltan parámetros en la configuración 'test_retrieval'. Asegúrate de definir 'task_key', 'query_clause' y 'embedding_model'.")
        return

    log.info(f"Tarea seleccionada: {task_key}")
    log.info(f"Modelo de embedding: {embedding_model}")
    log.info(f"Top K: {top_k}")
    log.info(f"Cláusula de consulta: {query_clause}")

    # --- Cargar la Base de Datos Vectorial ---
    embed_short = embedding_model.split('/')[-1] # Extrae el nombre corto del modelo
    storage_dir = cfg.embedding_config.storage_dir_template.format(task_key=task_key, embed_short=embed_short)
    
    try:
        document_store = load_knowledge_base(storage_dir, collection_name=task_key)
    except FileNotFoundError:
        return # El error ya se ha logueado en la función

    # --- Construir el Pipeline de Recuperación de Haystack ---
    retrieval_pipeline = Pipeline()
    retrieval_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=embedding_model))
    retrieval_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store))
    retrieval_pipeline.connect("embedder.embedding", "retriever.query_embedding")

    # --- Ejecutar la Búsqueda ---
    log.info("Ejecutando la búsqueda de similitud...")
    result = retrieval_pipeline.run({
        "embedder": {"text": query_clause},
        "retriever": {"top_k": top_k}
    })

    # --- Mostrar Resultados ---
    #print(result["retriever"])
    retrieved_docs = result["retriever"]["documents"]
    if not retrieved_docs:
        log.warning("No se encontraron documentos similares.")
        return

    log.info(f"--- Top {len(retrieved_docs)} Cláusulas Similares Encontradas ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Resultado {i+1} | Puntuación de Similitud: {doc.score:.4f} ---")
        print(f"\n--- Resultado {i+1} | Metadata: {doc.meta} ---")
        print(doc.content)
    log.info("--- Fin del Test ---")


if __name__ == "__main__":
    main()
