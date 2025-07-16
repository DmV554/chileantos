import ast
import logging
import re

log = logging.getLogger(__name__)

def parse_llm_output(output: str, possible_labels: list) -> list:
    """
    Parsea la salida de un LLM, que se espera que sea una lista de strings.
    Intenta primero con una evaluación literal de Python (ast.literal_eval).
    Si falla, busca las etiquetas conocidas dentro del texto como fallback.
    """
    try:
        # Intenta el parseo estricto primero
        predicted_labels = ast.literal_eval(output.strip())
        if isinstance(predicted_labels, list):
            # Filtra para devolver solo las etiquetas que son válidas
            return [str(label) for label in predicted_labels if str(label) in possible_labels]
    except (ValueError, SyntaxError, NameError):
        log.warning(f"No se pudo parsear la salida con ast: '{output}'. Usando fallback de regex.")
    
    # Fallback: buscar las etiquetas conocidas en el texto
    matches = []
    for label in possible_labels:
        # Busca la etiqueta como una palabra completa para evitar coincidencias parciales
        if re.search(rf'\b{re.escape(str(label))}\b', output):
            matches.append(str(label))
    return matches
