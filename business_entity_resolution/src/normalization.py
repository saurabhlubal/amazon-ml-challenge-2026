def normalize_record(record):
    """
    Normalize one business entity.

    Parameters
    ----------
    record : dict
        Raw entity record containing:
        entity_id, business_name, business_address, country

    Returns
    -------
    dict
        Multiple normalized representations.
    """
    raise NotImplementedError