# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import torch
import pandas as pd

from qlib.data.dataset import DatasetH, TSDatasetH, TSDataSampler
from typing import Union, List, Dict
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import check_transform_proc


device = "cuda" if torch.cuda.is_available() else "cpu"


###################################################################################
# lqa: for MASTER
class marketDataHandler(DataHandlerLP):
    """Market Data Handler for MASTER (see `examples/benchmarks/MASTER`)

    Args:
        instruments (str): instrument list
        start_time (str): start time
        end_time (str): end time
        freq (str): data frequency
        infer_processors (list): inference processors
        learn_processors (list): learning processors
        fit_start_time (str): fit start time
        fit_end_time (str): fit end time
        process_type (str): process type
        filter_pipe (list): filter pipe
        inst_processors (list): instrument processors
    """

    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    @staticmethod
    def get_feature_config():
        """
        Get market feature (63-dimensional), which are csi100 index, csi300 index, csi500 index. 
        The first list is the name to be shown for the feature, and the second list is the feature to fecth.
        """
        return (
            ['Mask($close/Ref($close,1)-1, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,5), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,5), "sh000300")', 'Mask(Mean($volume,5)/$volume, "sh000300")', 'Mask(Std($volume,5)/$volume, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,10), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,10), "sh000300")', 'Mask(Mean($volume,10)/$volume, "sh000300")', 'Mask(Std($volume,10)/$volume, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,20), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,20), "sh000300")', 'Mask(Mean($volume,20)/$volume, "sh000300")', 'Mask(Std($volume,20)/$volume, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,30), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,30), "sh000300")', 'Mask(Mean($volume,30)/$volume, "sh000300")', 'Mask(Std($volume,30)/$volume, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,60), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,60), "sh000300")', 'Mask(Mean($volume,60)/$volume, "sh000300")', 'Mask(Std($volume,60)/$volume, "sh000300")',
                'Mask($close/Ref($close,1)-1, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,5), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,5), "sh000903")', 'Mask(Mean($volume,5)/$volume, "sh000903")', 'Mask(Std($volume,5)/$volume, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,10), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,10), "sh000903")', 'Mask(Mean($volume,10)/$volume, "sh000903")', 'Mask(Std($volume,10)/$volume, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,20), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,20), "sh000903")', 'Mask(Mean($volume,20)/$volume, "sh000903")', 'Mask(Std($volume,20)/$volume, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,30), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,30), "sh000903")', 'Mask(Mean($volume,30)/$volume, "sh000903")', 'Mask(Std($volume,30)/$volume, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,60), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,60), "sh000903")', 'Mask(Mean($volume,60)/$volume, "sh000903")', 'Mask(Std($volume,60)/$volume, "sh000903")',
                'Mask($close/Ref($close,1)-1, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,5), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,5), "sh000905")', 'Mask(Mean($volume,5)/$volume, "sh000905")', 'Mask(Std($volume,5)/$volume, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,10), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,10), "sh000905")', 'Mask(Mean($volume,10)/$volume, "sh000905")', 'Mask(Std($volume,10)/$volume, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,20), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,20), "sh000905")', 'Mask(Mean($volume,20)/$volume, "sh000905")', 'Mask(Std($volume,20)/$volume, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,30), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,30), "sh000905")', 'Mask(Mean($volume,30)/$volume, "sh000905")', 'Mask(Std($volume,30)/$volume, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,60), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,60), "sh000905")', 'Mask(Mean($volume,60)/$volume, "sh000905")', 'Mask(Std($volume,60)/$volume, "sh000905")'],
            ['Mask($close/Ref($close,1)-1, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,5), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,5), "sh000300")', 'Mask(Mean($volume,5)/$volume, "sh000300")', 'Mask(Std($volume,5)/$volume, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,10), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,10), "sh000300")', 'Mask(Mean($volume,10)/$volume, "sh000300")', 'Mask(Std($volume,10)/$volume, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,20), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,20), "sh000300")', 'Mask(Mean($volume,20)/$volume, "sh000300")', 'Mask(Std($volume,20)/$volume, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,30), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,30), "sh000300")', 'Mask(Mean($volume,30)/$volume, "sh000300")', 'Mask(Std($volume,30)/$volume, "sh000300")', 'Mask(Mean($close/Ref($close,1)-1,60), "sh000300")', 'Mask(Std($close/Ref($close,1)-1,60), "sh000300")', 'Mask(Mean($volume,60)/$volume, "sh000300")', 'Mask(Std($volume,60)/$volume, "sh000300")',
                'Mask($close/Ref($close,1)-1, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,5), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,5), "sh000903")', 'Mask(Mean($volume,5)/$volume, "sh000903")', 'Mask(Std($volume,5)/$volume, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,10), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,10), "sh000903")', 'Mask(Mean($volume,10)/$volume, "sh000903")', 'Mask(Std($volume,10)/$volume, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,20), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,20), "sh000903")', 'Mask(Mean($volume,20)/$volume, "sh000903")', 'Mask(Std($volume,20)/$volume, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,30), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,30), "sh000903")', 'Mask(Mean($volume,30)/$volume, "sh000903")', 'Mask(Std($volume,30)/$volume, "sh000903")', 'Mask(Mean($close/Ref($close,1)-1,60), "sh000903")', 'Mask(Std($close/Ref($close,1)-1,60), "sh000903")', 'Mask(Mean($volume,60)/$volume, "sh000903")', 'Mask(Std($volume,60)/$volume, "sh000903")',
                'Mask($close/Ref($close,1)-1, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,5), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,5), "sh000905")', 'Mask(Mean($volume,5)/$volume, "sh000905")', 'Mask(Std($volume,5)/$volume, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,10), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,10), "sh000905")', 'Mask(Mean($volume,10)/$volume, "sh000905")', 'Mask(Std($volume,10)/$volume, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,20), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,20), "sh000905")', 'Mask(Mean($volume,20)/$volume, "sh000905")', 'Mask(Std($volume,20)/$volume, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,30), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,30), "sh000905")', 'Mask(Mean($volume,30)/$volume, "sh000905")', 'Mask(Std($volume,30)/$volume, "sh000905")', 'Mask(Mean($close/Ref($close,1)-1,60), "sh000905")', 'Mask(Std($close/Ref($close,1)-1,60), "sh000905")', 'Mask(Mean($volume,60)/$volume, "sh000905")', 'Mask(Std($volume,60)/$volume, "sh000905")']
        )


class MASTERTSDatasetH(TSDatasetH):
    """
    MASTER Time Series Dataset with Handler

    Args:
        market_data_handler_config (dict): market data handler config
    """

    def __init__(
        self,
        market_data_handler_config=Dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        marketdl = marketDataHandler(**market_data_handler_config)
        self.market_dataset = DatasetH(marketdl, segments=self.segments)

    def get_market_information(
        self,
        slc: slice,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        return self.market_dataset.prepare(slc)

    def _prepare_seg(self, slc: slice, **kwargs) -> TSDataSampler:
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        start, end = slc.start, slc.stop
        flt_col = kwargs.pop("flt_col", None)
        # TSDatasetH will retrieve more data for complete time-series

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        only_label = kwargs.pop("only_label", False)
        data = super(TSDatasetH, self)._prepare_seg(ext_slice, **kwargs)

        ############################## Add market information ###########################
        # If we only need label for testing, we do not need to add market information
        if not only_label:
            marketData = self.get_market_information(ext_slice)
            cols = pd.MultiIndex.from_tuples([("feature", feature) for feature in marketData.columns])
            marketData = pd.DataFrame(marketData.values, columns=cols, index=marketData.index)
            data = data.iloc[:, :-1].join(marketData).join(data.iloc[:, -1])
        #################################################################################
        flt_kwargs = copy.deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super()._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        tsds = TSDataSampler(
            data=data,
            start=start,
            end=end,
            step_len=self.step_len,
            dtype=dtype,
            flt_data=flt_data,
            fillna_type="ffill+bfill"
        )
        return tsds
