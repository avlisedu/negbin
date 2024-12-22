def validate_data(data):
    """
    Valida se o DataFrame segue o formato esperado.
    Retorna True se for válido, False caso contrário.
    """
    if data.shape[1] < 2:
        return False
    return True
