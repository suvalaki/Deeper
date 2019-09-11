conda create -n tf2b python==3.6 pip tensorflow-gpu
conda activate tf2
conda uninstall tensorflow-gpu
pip install tensorflow-gpu==2.0.0-beta1 tfp-nightly