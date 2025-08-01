import ast
import logging
import re

log = logging.getLogger(__name__)


def get_model_short_name(model_name: str) -> str:
    """
    Extrae el nombre corto del modelo de Hugging Face.

    Args:
        model_name (str): Nombre completo del modelo (e.g., 'Qwen/Qwen3-Embedding-0.6B')

    Returns:
        str: Nombre corto del modelo (e.g., 'Qwen3-Embedding-0.6B')
    """
    return model_name.split("/")[-1]


def parse_model_output_with_reasoning(output: str) -> dict:
    """
    Extrae el reasoning y la lista de etiquetas desde la salida del modelo.

    Args:
        output (str): Texto completo de la respuesta del modelo.

    Returns:
        dict: Contiene 'reasoning' (str) y 'labels' (list[str]). Vacíos si hay error.
    """
    result = {
        "reasoning": "",
        "labels": []
    }

    # Extraer reasoning
    reasoning_match = re.search(r"\*\*REASONING:\*\*\s*(.*?)\s*\*\*LABELS:\*\*", output, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extraer labels
    labels_match = re.search(r"\*\*LABELS:\*\*\s*(\[.*?\])", output, re.DOTALL)
    if labels_match:
        labels_str = labels_match.group(1).strip()
        try:
            result["labels"] = ast.literal_eval(labels_str)
        except (SyntaxError, ValueError):
            result["labels"] = []

    return result


def parse_model_output(output: str):
    try:
        # Eliminar razonamiento entre <think>...</think> si existe
        output_clean = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

        # Buscar la última lista válida en el string (para evitar texto adicional)
        matches = re.findall(r"\[[^\[\]]+\]", output_clean)
        if matches:
            return ast.literal_eval(matches[-1])
    except (ValueError, SyntaxError):
        pass
    return []

