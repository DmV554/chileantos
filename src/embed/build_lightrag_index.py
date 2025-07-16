import hydra
from omegaconf import DictConfig
import logging
import os
import sys
import asyncio

# --- Modificación del Path para Imports Absolutos ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---

from lightrag.lightrag import LightRAG
from lightrag.components.model_client import ollama_model_complete, ollama_embed, EmbeddingFunc
from src.utils.data_loader import load_dataset_for_indexing

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def main_async(cfg: DictConfig):
    """
    Función asíncrona principal para construir el índice de LightRAG.
    """
    log.info("--- Iniciando la construcción del Índice con LightRAG ---")

    # --- 1. Cargar Configuración ---
    task_cfg = cfg.task
    strategy_cfg = cfg.strategy.lightrag
    model_cfg = cfg.models
    db_cfg = cfg.db.naive # Usamos la config de embedding naive

    # Definir la ruta absoluta al workspace de LightRAG donde se guardarán los índices
    workspace_path = os.path.join(hydra.utils.get_original_cwd(), strategy_cfg.index_path)
    log.info(f"Workspace de LightRAG (salida): {workspace_path}")

    # --- 2. Definir Funciones Wrapper para los Modelos ---
    # Estas funciones leen la config de Hydra y llaman a las funciones de LightRAG
    async def llm_wrapper(prompt, **kwargs):
        return await ollama_model_complete(prompt, model=model_cfg.llm, **kwargs)

    async def embedding_wrapper(texts: list[str]):
        return await ollama_embed(texts, model=db_cfg.embedding_model)

    # --- 3. Configurar e Instanciar LightRAG ---
    rag_app = None # Asegurarse de que rag_app existe en el bloque finally
    try:
        rag_app = LightRAG(
            working_dir=workspace_path,
            llm_model_func=llm_wrapper,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024, # Ajustar según el modelo, ej. 768 para nomic, 4096 para bge-m3
                func=embedding_wrapper
            )
        )
        
        # Inicialización obligatoria en dos pasos
        await rag_app.initialize_storages()
        # await initialize_pipeline_status() # Esta función parece no ser necesaria en versiones recientes

        log.info("Instancia de LightRAG inicializada correctamente.")

        # --- 4. Cargar Datos y Ejecutar Indexación ---
        documents_to_index = load_dataset_for_indexing(task_cfg)
        texts_to_index = [doc['text'] for doc in documents_to_index]

        log.info(f"Iniciando la indexación de {len(texts_to_index)} documentos...")
        rag_app.insert(texts_to_index)
        log.info("El proceso de indexación de LightRAG ha finalizado.")

    except Exception as e:
        log.error(f"Ocurrió un error durante el proceso de LightRAG: {e}", exc_info=True)
    finally:
        if rag_app:
            await rag_app.finalize_storages()
            log.info("Conexiones de almacenamiento de LightRAG finalizadas.")

    log.info(f"--- ✅ Índice de LightRAG para '{task_cfg.name}' creado en: {workspace_path} ---")

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Punto de entrada síncrono que ejecuta el bucle de eventos de asyncio."""
    asyncio.run(main_async(cfg))

if __name__ == "__main__":
    main()