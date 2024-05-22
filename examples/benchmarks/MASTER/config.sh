# create a new conda environment named `MASTER`
conda create -n MASTER python=3.8
eval "$(conda shell.bash hook)"
conda activate MASTER

# install `qlib`
pip install pyqlib
pip install -r requirements.txt

# the following codes copy the `*.so` files in the installed `qlib` package to our customizable `qlib`
path=$(which conda)
parent_path=$(dirname $path)
parent_path=$(dirname $parent_path)
lib_path="$parent_path/envs/MASTER/lib/python3.8/site-packages/qlib/data/_libs"
echo $lib_path
find $lib_path -type f -name "*.so" -exec cp {} "../../../qlib/data/_libs" \;

# we directly use our customizable `qlib`
pip uninstall pyqlib