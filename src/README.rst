************
Installation
************

Download dataset
-------
.. code-block:: bash

	aws s3 sync --no-sign-request s3://openneuro.org/ds003688 ds003688-download/
	

Normalize fMRI
-------
.. code-block:: bash

	python mylib/data_preprocess_and_load/preprocessing.py


Train
-------
.. code-block:: bash
	
	

