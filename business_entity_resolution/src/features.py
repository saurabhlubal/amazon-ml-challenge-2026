def build_features(source1_record, candidate_record):
    """
    Build comparison features for an S1/candidate pair.

    Parameters
    ----------
    source1_record : dict
        Source1 entity.

    candidate_record : dict
        Source2 or Source3 candidate entity.

    Returns
    -------
    dict
        Feature name -> numeric value.
    """
    raise NotImplementedError