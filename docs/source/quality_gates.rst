Quality Gates
=============

This repository enforces three maintainability gates for first-party Python code:

* Cyclomatic complexity must be less than 4, which means the configured maximum
  is 3.
* Cross-repository code duplication must stay below 1 percent.
* Public functions must include functional documentation: a docstring, an
  ``Args:`` section when parameters are present, and a ``Returns:`` or
  ``Yields:`` section when the function returns a value.

Run the gates locally:

.. code-block:: bash

   python tools/quality_gates.py

The current codebase has a ratchet baseline at
``tools/quality_gates_baseline.json``. New findings fail CI. As functions are
refactored or documented, remove resolved entries by regenerating the baseline:

.. code-block:: bash

   python tools/quality_gates.py --update-baseline

``third_party/`` is excluded because it is vendored code. The enforcement surface
is ``hawk_eye/``, which contains the repository's library, training, inference,
and data generation functionality.
