pip uninstall torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

pip install torch==1.13.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch==1.13.1+cu117.html
