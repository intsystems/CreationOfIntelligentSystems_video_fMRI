Requirements
-------

Python 3.8.16

Download dataset
-------
.. code-block:: bash

	aws s3 sync --no-sign-request s3://openneuro.org/ds003688 ds003688-download/

Also download video and VTN weights (link to VTN checkpoint: https://researchpublic.blob.core.windows.net/vtn/VTN_VIT_B_KINETICS.pyth)
	

Normalize fMRI
-------
.. code-block:: bash

	python mylib/data_preprocess_and_load/preprocessing.py


Train
-------
.. code-block:: bash
	
	python mylib/main.py
	
	

