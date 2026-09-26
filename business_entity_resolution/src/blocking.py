def generate_candidates(source1_record, indexes):
    """
    Generate candidate Source2/Source3 entity IDs for one Source1 record.

    Parameters
    ----------
    source1_record : dict
        Source1 entity.

    indexes : dict
        Pre-built blocking indexes.

    Returns
    -------
    set[str]
        Candidate entity IDs.
    """
    raise NotImplementedError