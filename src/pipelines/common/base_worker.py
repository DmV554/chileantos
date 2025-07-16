import mlflow
import logging
from abc import ABC, abstractmethod
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class BaseExperimentWorker(ABC):
    """
    Clase base abstracta para todos los workers de experimentos.
    Maneja la configuración y el boilerplate de MLflow.
    """

    def __init__(self, cfg: DictConfig):
        """
        Inicializa el worker con la configuración de Hydra.
        """
        self.cfg = cfg
        self.task_cfg = cfg.task
        self.run_config = cfg.run_config
        self.task_name_key = self.task_cfg.name.split(' - ')[-1].lower()

    def run(self) -> None:
        """
        Orquesta el flujo completo de un experimento:
        1. Configura MLflow.
        2. Inicia un run.
        3. Loguea parámetros.
        4. Ejecuta la tarea principal.
        """
        mlflow.set_tracking_uri(self.run_config.mlflow_tracking_uri)
        mlflow.set_experiment(self.run_config.mlflow_experiment_name)
        
        run_name = self._get_run_name()

        log.info(f"--- Iniciando run: '{run_name}' ---")
        with mlflow.start_run(run_name=run_name) as run:
            # Set parent run ID if provided by the orchestrator
            if "parent_run_id" in self.run_config and self.run_config.parent_run_id != "placeholder":
                mlflow.set_tag("mlflow.parentRunId", self.run_config.parent_run_id)

            # Log parameters defined by the specific worker
            params = self._get_parameters_to_log()
            mlflow.log_params(params)
            log.info(f"Parámetros logueados: {params}")

            # Execute the main task of the worker
            log.info("Ejecutando la tarea principal del worker...")
            self._execute_task()
            log.info(f"--- ✅ Run '{run_name}' finalizado exitosamente. ---")

    @abstractmethod
    def _get_run_name(self) -> str:
        """
        Debe ser implementado por la subclase para devolver el nombre
        específico del run de MLflow.
        """
        pass

    @abstractmethod
    def _get_parameters_to_log(self) -> dict:
        """
        Debe ser implementado por la subclase para devolver un diccionario
        con los parámetros específicos a loguear en MLflow.
        """
        pass

    @abstractmethod
    def _execute_task(self) -> None:
        """
        Debe ser implementado por la subclase para ejecutar la lógica
        principal del experimento (inferencia, evaluación, etc.).
        """
        pass
