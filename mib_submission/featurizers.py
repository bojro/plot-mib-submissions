"""
Re-exports of upstream's ``Featurizer`` classes.

We deliberately do *not* define new featurizer subclasses. Upstream's
``Featurizer.load_modules`` only recognises ``SubspaceFeaturizerModule`` and
``IdentityFeaturizerModule`` — anything else raises ``ValueError``. The
mock submission in ``MIB/MIB-causal-variable-track/mock_submission/`` defines
a ``SubspaceFeaturizerModuleCopy`` and would silently fail to deserialise
against the evaluator; we route around that by saving with upstream classes.

Every method we benchmark (OT / GW / FGW / UOT / OT+gradient / OT+DAS /
OT+PCA) is encoded as one of the two upstream classes plus an
``_indices`` JSON file — see ``method_to_featurizer.py``.
"""

from CausalAbstraction.neural.featurizers import (  # noqa: F401
    Featurizer,
    IdentityFeaturizerModule,
    IdentityInverseFeaturizerModule,
    SubspaceFeaturizer,
    SubspaceFeaturizerModule,
    SubspaceInverseFeaturizerModule,
)
