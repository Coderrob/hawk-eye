Setup
==============================================================================

This projects works best with Linux, WSL, and Mac systems. The ``setup_env.sh`` script
in ``hawk_eye/setup`` will prepare Ubuntu and WSL systems. To run the script, execute
::

    hawk_eye/setup/setup_env.sh


inside of the ``hawk_eye`` repository. This file also takes an optional argument to a python
virtual environemnt
::

    hawk_eye/setup/setup_env.sh ~/path_to_venv


Upon sucessful termination of this script, you need to then recieve access to Google
Cloud Storage. See your lead about gaining permissions, then run
::

    hawk_eye/setup/install_google_cloud.sh

To ensure you are all setup and ready to code, you can run the tests with
::

    bazel test //...
