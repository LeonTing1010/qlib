if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./backtest" ]; then
    mkdir ./backtest
fi

# set the market, you can choose `csi300` or `csi500`
universe=csi300

# set the flag: whether only backtest. 
# If choose 'true', we will directly use the trained model and do backtesting; If choose 'false', the script will first train then backtest.
only_backtest=false

sed -i "s/csi.../$universe/g" workflow_config_master_Alpha158.yaml
if [ $universe == 'csi300' ]; then
    sed -i "s/SH....../SH000300/g" workflow_config_master_Alpha158.yaml
elif [ $universe == 'csi500' ]; then
    sed -i "s/SH....../SH000905/g" workflow_config_master_Alpha158.yaml
fi
if $only_backtest; then
    nohup python -u main.py --only_backtest > ./backtest/${universe}.log 2>&1 &
else
    nohup python -u main.py > ./logs/${universe}.log 2>&1 &
fi
echo $!