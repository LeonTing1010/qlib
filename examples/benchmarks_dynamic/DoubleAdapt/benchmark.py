# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.tests.data import GetData
from qlib.workflow import R
from qlib.model.trainer import TrainerR
from tqdm.auto import tqdm
from qlib import auto_init, init
import yaml
import fire
from qlib.utils import init_instance_by_config
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.workflow.task.utils import TimeAdjuster
from qlib.data.dataset import Dataset, DataHandlerLP, TSDataSampler
from qlib.utils.data import update_config
import qlib
import torch
import pandas as pd
import numpy as np
from typing import Optional
from pprint import pprint
import time
import os
import pickle
from pathlib import Path
import sys

DIRNAME = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(DIRNAME.parent))
sys.path.append(str(DIRNAME.parent.parent))


class Benchmark:
    def __init__(self, data_dir="cn_data", market="csi300", model_type="linear", alpha="360", rank_label=True,
                 lr=0.001, early_stop=8, reload=False, horizon=1,
                 init_data=True,
                 h_path: Optional[str] = None,
                 train_start: Optional[str] = None,
                 train_end: Optional[str] = None,
                 valid_start: Optional[str] = None,
                 valid_end: Optional[str] = None,
                 test_start: Optional[str] = None,
                 test_end: Optional[str] = None,
                 task_ext_conf: Optional[dict] = None,) -> None:
        self.data_dir = data_dir
        self.market = market
        if init_data:
            if data_dir == "cn_data":
                GetData().qlib_data(target_dir="~/.qlib/qlib_data/cn_data", exists_skip=True)
                auto_init()
            else:
                qlib.init(
                    provider_uri="~/.qlib/qlib_data/" + data_dir, region="us" if self.data_dir == "us_data" else "cn",
                )
        self.horizon = horizon
        # self.rolling_exp = rolling_exp
        self.model_type = model_type
        self.h_path = h_path
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.test_start = test_start
        self.test_end = test_end
        self.task_ext_conf = task_ext_conf
        self.alpha = alpha
        self.exp_name = f"{model_type}_{self.data_dir}_{self.market}_{self.alpha}_rank{rank_label}"
        self.rank_label = rank_label
        self.lr = lr
        self.early_stop = early_stop
        self.reload = reload
        self.tag = ""
        if self.data_dir == "us_data":
            self.benchmark = "^gspc"
        elif self.market == "csi500":
            self.benchmark = "SH000905"
        elif self.market == "csi100":
            self.benchmark = "SH000903"
        else:
            self.benchmark = "SH000300"
        R.set_uri((DIRNAME / 'mlruns').as_uri())

    def basic_task(self):
        """For fast training rolling"""
        if self.model_type == "gbdt":
            conf_path = (
                DIRNAME.parent
                / "benchmarks"
                / "LightGBM"
                / "workflow_config_lightgbm_Alpha{}.yaml".format(self.alpha)
            )
            # dump the processed data on to disk for later loading to speed up the processing
            filename = "lightgbm_alpha{}_handler_horizon{}.pkl".format(
                self.alpha, self.horizon
            )
        elif self.model_type.lower() == "linear":
            conf_path = (
                DIRNAME.parent
                / "benchmarks"
                / "Linear"
                / "workflow_config_linear_Alpha{}.yaml".format(self.alpha)
            )
            filename = "linear_alpha{}_handler_horizon{}.pkl".format(
                self.alpha, self.horizon
            )
        elif "MLP" in self.model_type:
            conf_path = (
                DIRNAME.parent
                / "benchmarks"
                / self.model_type
                / "workflow_config_mlp_Alpha{}.yaml".format(self.alpha)
            )
            filename = "MLP_alpha{}_handler_horizon{}.pkl".format(
                self.alpha, self.horizon
            )
        else:
            # NOTE: For Alpha158, all RNN models only use 20 features. Carefully take the pkl if your model use different features.
            conf_path = (
                DIRNAME.parent
                / "benchmarks"
                / self.model_type
                / "workflow_config_{}_Alpha{}.yaml".format(
                    self.model_type.lower(), self.alpha
                )
            )
            filename = "alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        filename = f"{self.data_dir}_{self.market}_rank{self.rank_label}_{filename}"
        h_path = DIRNAME / filename

        if self.h_path is not None:
            h_path = Path(self.h_path)

        with conf_path.open("r") as f:
            conf = yaml.safe_load(f)

        # modify dataset horizon
        conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
            "Ref($close, -{}) / Ref($close, -1) - 1".format(self.horizon + 1)
        ]

        if self.market != "csi300":
            conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = self.market
            if self.data_dir == "us_data":
                conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
                    "Ref($close, -{}) / $close - 1".format(self.horizon)
                ]

        for k, v in {'early_stop': self.early_stop, "lr": self.lr, "seed": None, }.items():
            if k in conf["task"]["model"]["kwargs"]:
                conf["task"]["model"]["kwargs"][k] = v
        if conf["task"]["model"]["class"] == "TransformerModel":
            conf["task"]["model"]["kwargs"]["dim_feedforward"] = 32
            conf["task"]["model"]["kwargs"]["reg"] = 0
        if conf["task"]["model"]["class"] == "DNNModelPytorch":
            conf["task"]["model"]["kwargs"]["scheduler"] = None

        task = conf["task"]

        if self.task_ext_conf is not None:
            task = update_config(task, self.task_ext_conf)

        h_conf = task["dataset"]["kwargs"]["handler"]
        if not (self.model_type == "gbdt" and self.alpha == 158):
            expect_label_processor = "CSRankNorm" if self.rank_label else "CSZScoreNorm"
            delete_label_processor = "CSZScoreNorm" if self.rank_label else "CSRankNorm"
            proc = h_conf["kwargs"]["learn_processors"][-1]
            if (
                isinstance(proc, str) and self.rank_label and proc == delete_label_processor
                or
                isinstance(proc, dict) and proc["class"] == delete_label_processor
            ):
                h_conf["kwargs"]["learn_processors"] = h_conf["kwargs"]["learn_processors"][:-1]
                print("Remove", delete_label_processor)
                h_conf["kwargs"]["learn_processors"].append(
                    {"class": expect_label_processor, "kwargs": {"fields_group": "label"}}
                )
        print(h_conf)

        if not h_path.exists():
            h = init_instance_by_config(h_conf)
            h.to_pickle(h_path, dump_all=True)
            print('Save handler file to', h_path)

        # if not self.rank_label:
        #     task['model']['kwargs']['loss'] = 'ic'
        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        task["record"] = ["qlib.workflow.record_temp.SignalRecord"]

        if self.train_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["train"] = pd.Timestamp(self.train_start), seg[1]

        if self.train_end is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["test"] = seg[0], pd.Timestamp(self.train_end)

        if self.valid_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["valid"]
            task["dataset"]["kwargs"]["segments"]["valid"] = pd.Timestamp(self.valid_start), seg[1]

        if self.valid_end is not None:
            seg = task["dataset"]["kwargs"]["segments"]["valid"]
            task["dataset"]["kwargs"]["segments"]["valid"] = seg[0], pd.Timestamp(self.valid_end)

        if self.test_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["test"]
            task["dataset"]["kwargs"]["segments"]["test"] = pd.Timestamp(self.test_start), seg[1]

        if self.test_end is not None:
            seg = task["dataset"]["kwargs"]["segments"]["test"]
            task["dataset"]["kwargs"]["segments"]["test"] = seg[0], pd.Timestamp(self.test_end)
        print(task)
        return task

    def get_fitted_model(self, suffix=""):
        task = self.basic_task()
        try:
            assert self.reload
            rec = list(R.list_recorders(experiment_name=self.exp_name + suffix).values())[0]
            with open(os.path.join(rec.artifact_uri[rec.artifact_uri.find('mlruns'):], 'params.pkl'), 'rb') as f:
                model = pickle.Unpickler(f).load()
            # model = rec.load_object("params.pkl")
            print(f"Load pretrained model from {self.exp_name + suffix}.")
        except Exception as e:
            print(e)
            model = init_instance_by_config(task["model"])
            dataset = init_instance_by_config(task["dataset"])
            # start exp
            with R.start(experiment_name=self.exp_name + suffix):
                model.fit(dataset)
                R.save_objects(**{"params.pkl": model})
            print(f"Save pretrained model to {self.exp_name + suffix}.")
        return model

    def run_all(self):
        task = self.basic_task()
        test_begin, test_end = task["dataset"]["kwargs"]["segments"]["test"]
        ta = TimeAdjuster(future=True, end_time=test_end)
        test_begin = ta.get(ta.align_idx(test_begin))
        test_end = ta.get(ta.align_idx(test_end))

        # model = init_instance_by_config(task["model"])
        dataset = init_instance_by_config(task["dataset"])
        # start exp
        model = self.get_fitted_model(f"_{self.seed}")
        with R.start(experiment_name=self.exp_name):
            recorder = R.get_recorder()
            pred = model.predict(dataset)
            if isinstance(pred, pd.Series):
                pred = pred.to_frame("score")
            pred = pred.loc[test_begin:test_end]

            raw_label = dataset.prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_R)
            if isinstance(raw_label, TSDataSampler):
                raw_label = pd.DataFrame({"label": raw_label.data_arr[:-1][:, 0]}, index=raw_label.data_index)
                # raw_label = raw_label.loc[test_begin:test_end]
            raw_label = raw_label.loc[pred.index]

            recorder.save_objects(**{"pred.pkl": pred, "label.pkl": raw_label})

            # Signal Analysis
            SigAnaRecord(recorder).generate()

            # backtest. If users want to use backtest based on their own prediction,
            # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
            backtest_config = {
                "strategy": {
                    "class": "TopkDropoutStrategy",
                    "module_path": "qlib.contrib.strategy",
                    "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
                },
                "backtest": {
                    "start_time": None,
                    "end_time": None,
                    "account": 100000000,
                    "benchmark": self.benchmark,
                    "exchange_kwargs": {
                        "limit_threshold": None if self.data_dir == "us_data" else 0.095,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5,
                    },
                },
            }
            PortAnaRecord(recorder=recorder, config=backtest_config).generate()

            label = init_instance_by_config(self.basic_task()["dataset"], accept_types=Dataset).\
                prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_L)
            label = label[label.index.isin(pred.index)]
            if isinstance(label, TSDataSampler):
                label = pd.DataFrame({'label': label.data_arr[:-1][:, 0]}, index=label.data_index)
            else:
                label.columns = ['label']

            label['pred'] = pred.loc[label.index]
            # rmse = np.sqrt(((label['pred'].to_numpy() - label['label'].to_numpy()) ** 2).mean())
            mse = ((label['pred'].to_numpy() - label['label'].to_numpy()) ** 2).mean()
            mae = np.abs(label['pred'].to_numpy() - label['label'].to_numpy()).mean()
            recorder.log_metrics(mse=mse, mae=mae)
            print(f"Your evaluation results can be found in the experiment named `{self.exp_name}`.")
        return recorder

    def run_exp(self):
        all_metrics = {
            k: []
            for k in [
                'mse', 'mae',
                "IC",
                "ICIR",
                "Rank IC",
                "Rank ICIR",
                "1day.excess_return_with_cost.annualized_return",
                "1day.excess_return_with_cost.information_ratio",
                # "1day.excess_return_with_cost.max_drawdown",
            ]
        }
        test_time = []
        for i in range(0, 5 if self.model_type.lower() != 'linear' else 1):
            np.random.seed(43 + i)
            torch.manual_seed(43 + i)
            torch.cuda.manual_seed(43 + i)
            start_time = time.time()
            self.seed = i + 43
            rec = self.run_all()
            test_time.append(time.time() - start_time)
            # exp = R.get_exp(experiment_name=self.exp_name)
            # rec = exp.list_recorders(rtype=exp.RT_L)[0]
            metrics = rec.list_metrics()
            for k in all_metrics.keys():
                all_metrics[k].append(metrics[k])
            pprint(all_metrics)

        with R.start(
            experiment_name=f"final_{self.data_dir}_{self.market}_{self.alpha}_{self.horizon}_{self.model_type}"
        ):
            R.save_objects(all_metrics=all_metrics)
            test_time = np.array(test_time)
            R.log_metrics(test_time=test_time)
            print(f"Time cost: {test_time.mean()}")
            res = {}
            for k in all_metrics.keys():
                v = np.array(all_metrics[k])
                res[k] = [v.mean(), v.std()]
                R.log_metrics(**{"final_" + k: res[k]})
            pprint(res)
        test_time = np.array(test_time)
        print(f"Time cost: {test_time.mean()}")


if __name__ == "__main__":
    # GetData().qlib_data(exists_skip=True)
    # auto_init()
    fire.Fire(Benchmark)
