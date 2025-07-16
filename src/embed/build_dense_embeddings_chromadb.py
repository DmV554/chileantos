import hydra
from omegaconf import DictConfig
import os
import logging
import hashlib
import sys
# --- Inicio: Modificación del Path para Imports Absolutos ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fin: Modificación del Path ---

# Imports de Haystack
from haystack.dataclasses import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# Importamos nuestro nuevo data loader
from src.utils.data_loader import load_dataset_for_indexing

log = logging.getLogger(__name__)

def create_haystack_documents(dataset):
    """Convierte el dataset en una lista de Documentos Haystack con metadatos."""
    log.info("Creando Documentos Haystack con doc_id y metadatos compatibles...")
    haystack_documents = []
    
    for row in dataset:
        clause_text = str(row['text'])
        doc_id = hashlib.sha256(clause_text.encode()).hexdigest()
        
        label_keys_list = row['human_readable_labels']
        labels_as_string = ", ".join(label_keys_list) if label_keys_list else "N/A"
        
        metadata = {
            "doc_id": doc_id,
            "source_split": str(row.get('split')),
            "label_keys": labels_as_string,
        }
        
        doc = Document(content=clause_text, meta=metadata)
        haystack_documents.append(doc)
        
    log.info(f"Se crearon {len(haystack_documents)} Documentos Haystack.")
    return haystack_documents

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    task_cfg = cfg.task
    task_name_key = task_cfg.name.split(' - ')[-1].lower()
    log.info(f"--- Iniciando creación de KB con Haystack/ChromaDB para: '{task_cfg.name}' ---")

    # Usamos la nueva función centralizada para cargar los datos
    full_dataset = load_dataset_for_indexing(task_cfg)
    haystack_documents = create_haystack_documents(full_dataset)
    
    storage_dir = cfg.db.naive.storage_dir.format(
        task_key=task_name_key, 
        embed_short=cfg.db.naive.embed_model_name_short
    )
    full_storage_path = os.path.join(hydra.utils.get_original_cwd(), storage_dir)

    document_store = ChromaDocumentStore(
        collection_name=task_name_key,
        persist_path=full_storage_path
    )
    log.info(f"Usando ChromaDB en la ruta: {full_storage_path}")
    
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model=cfg.db.naive.embedding_model
    )
    writer = DocumentWriter(document_store=document_store)
    
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    log.info(f"Ejecutando pipeline de indexación para {len(haystack_documents)} documentos...")
    indexing_pipeline.run({"embedder": {"documents": haystack_documents}})
    log.info("Indexación en ChromaDB completada.")
    
    log.info(f"--- ✅ KB de ChromaDB para '{task_cfg.name}' creada en: {full_storage_path} ---")

if __name__ == "__main__":
    main()