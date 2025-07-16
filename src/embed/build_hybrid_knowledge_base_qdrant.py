
import hydra
from omegaconf import DictConfig
import logging
import os
from tqdm import tqdm
import sys

# --- Inicio: Modificación del Path para Imports Absolutos ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fin: Modificación del Path ---
# Imports de Haystack y Datasets
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.dataclasses import Document

# Importamos nuestro nuevo data loader
from src.utils.data_loader import load_dataset_for_indexing

# --- Configuración del Logging ---
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Script para construir y guardar una base de conocimiento híbrida (BM25 + Vectores).
    """
    log.info("--- Iniciando la construcción de la Base de Conocimiento Híbrida con Qdrant ---")

    # --- Cargar configuración desde las rutas correctas ---
    task_cfg = cfg.task
    db_cfg = cfg.db.hybrid
    task_name_key = task_cfg.name.split(' - ')[-1].lower()

    log.info(f"Tarea seleccionada: {task_cfg.name}")
    log.info(f"Modelo de embedding: {db_cfg.embedding_model}")

    # Construir la ruta de almacenamiento
    storage_dir = db_cfg.storage_dir.format(
        task_key=task_name_key,
        embed_short=db_cfg.embed_model_name_short
    )
    full_storage_path = os.path.join(hydra.utils.get_original_cwd(), storage_dir)

    # --- Preparar Componentes de Haystack ---
    log.info(f"Inicializando QdrantDocumentStore en: {full_storage_path}")
    document_store = QdrantDocumentStore(
        path=full_storage_path,
        index=task_name_key, # Usamos el task_name_key como nombre de la colección
        recreate_index=True,
        embedding_dim=db_cfg.embedding_dim,
        use_sparse_embeddings=True,
        sparse_idf=True
    )

    doc_embedder = SentenceTransformersDocumentEmbedder(model=db_cfg.embedding_model)
    doc_embedder.warm_up()

    # --- Cargar y Procesar Datos ---
    all_docs_for_indexing = load_dataset_for_indexing(task_cfg)

    # Deduplicar documentos por contenido para evitar errores de Haystack
    log.info("Deduplicando documentos por contenido...")
    unique_docs = []
    seen_content = set()
    for item in all_docs_for_indexing:
        content = item['text']
        if content not in seen_content:
            unique_docs.append(item)
            seen_content.add(content)
    
    log.info(f"Documentos únicos encontrados: {len(unique_docs)} (de {len(all_docs_for_indexing)} totales)")

    haystack_docs = []
    for item in tqdm(unique_docs, desc="Preparando documentos de Haystack"):
        haystack_docs.append(Document(
            content=item['text'],
            meta={
                'human_readable_labels': item['human_readable_labels'],
                'source_split': item['split']
            }
        ))
    
    log.info(f"Total de documentos a indexar: {len(haystack_docs)}")

    # --- Ejecutar Indexación ---
    log.info(f"Generando embeddings con el modelo: {db_cfg.embedding_model}...")
    embedding_result = doc_embedder.run(documents=haystack_docs)
    embedded_docs = embedding_result["documents"]

    log.info("Escribiendo documentos en QdrantDocumentStore (BM25/BM42 + Vectores)...")
    document_store.write_documents(embedded_docs)
    
    log.info(f"✅ Base de conocimiento híbrida en Qdrant creada/actualizada exitosamente en: {full_storage_path}")

if __name__ == "__main__":
    main()
