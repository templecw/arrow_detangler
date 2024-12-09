# unified_label_mapping.py

generalized_categories = {
    'circle': 1,
    'rectangle': 2,
    'diamond': 3,
    'oval': 4,
    'arrow': 5,
    'text': 6,
}

unified_label_mapping = {
    # FA dataset
    'state': 1,            # circle
    'final state': 1,      # circle
    'text': 6,             # text
    'arrow': 5,            # arrow

    # FC_B dataset
    'connection': 1,       # circle
    'data': 2,             # rectangle
    'decision': 3,         # diamond
    'process': 2,          # rectangle
    'terminator': 4,       # oval
    # 'text' and 'arrow' are already mapped

    # hdBPMN dataset
    'task': 2,             # rectangle
    'subProcess': 2,       # rectangle
    'event': 1,            # circle
    'messageEvent': 1,     # circle
    'timerEvent': 1,       # circle
    'exclusiveGateway': 3, # diamond
    'parallelGateway': 3,  # diamond
    'eventBasedGateway': 3,# diamond
    'pool': 4,             # oval
    'lane': 2,             # rectangle
    'dataObject': 2,       # rectangle
    'dataStore': 2,        # rectangle
    'sequenceFlow': 5,     # arrow
    'messageFlow': 5,      # arrow
    'dataAssociation': 5,  # arrow
}
