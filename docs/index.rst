The Hypermodern Python Project
==============================

.. toctree::
   :hidden:
   :maxdepth: 1

   license
   reference

The example project for the
`Hypermodern Python <https://medium.com/@cjolowicz/hypermodern-python-d44485d9d769>`_
article series.
The command-line interface runs General Hough Transform on any .png images.


Installation
------------

To install the Hypermodern Python project,
run this command in your terminal:

.. code-block:: console

   $ pip install semester6


Usage
-----

Hypermodern Python's usage looks like:

.. code-block:: console

   $ poetry run semester6 [OPTIONS]

.. option:: --reference_path

   Path to the image that contains shape which is to be found on images.

.. option:: --query_path

   Path to the image that contains reference shape.

.. option:: --threshold

   Controls how many points will be counted as centers. Lower value means more points.
   For line detection, consider 0.05 - 0.15. For other shapes, 0.85 - 0.95.

.. option:: --threshold

   Option for showing that the reference image contains line. Used for adjusting the plot.

.. option:: --version

   Display the version and exit.

.. option:: --help

   Display a short usage message and exit.
