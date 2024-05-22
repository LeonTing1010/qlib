from qlib.data.dataset.handler import DataHandlerLP

from .handler import check_transform_proc

_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]


class DailyHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
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
                    "label": kwargs.pop("label", self.get_label_config()),
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

    def get_feature_config(self):
        """
        90集中度 close 主力资金流入 成交量 振幅 换手率 流通市值 竞价实际换手率 融资融券余额
        The first list is the name to be shown for the feature, and the second list is the feature to fecth.
        """
        return (
            ['$close', '$90集中度', '$主力资金流入', '$成交量', '$振幅', '$流通市值', '$竞价实际换手率', '$融资融券余额', '$换手率',
             '$close/Ref($close,1)-1',
             'Mean($close/Ref($close,1)-1,5)', 'Std($close/Ref($close,1)-1,5)',
             'Mean($close/Ref($close,1)-1,10)', 'Std($close/Ref($close,1)-1,10)',
             'Mean($close/Ref($close,1)-1,20)', 'Std($close/Ref($close,1)-1,20)',
             'Mean($close/Ref($close,1)-1,30)', 'Std($close/Ref($close,1)-1,30)',
             'Mean($close/Ref($close,1)-1,60)', 'Std($close/Ref($close,1)-1,60)',],
            ['$close', '$90集中度', '$主力资金流入', '$成交量', '$振幅', '$流通市值', '$竞价实际换手率', '$融资融券余额', '$换手率',
             '$close/Ref($close,1)-1',
             'Mean($close/Ref($close,1)-1,5)', 'Std($close/Ref($close,1)-1,5)',
             'Mean($close/Ref($close,1)-1,10)', 'Std($close/Ref($close,1)-1,10)',
             'Mean($close/Ref($close,1)-1,20)', 'Std($close/Ref($close,1)-1,20)',
             'Mean($close/Ref($close,1)-1,30)', 'Std($close/Ref($close,1)-1,30)',
             'Mean($close/Ref($close,1)-1,60)', 'Std($close/Ref($close,1)-1,60)',]
        )

    def get_label_config(self):
        return ["Ref($close, -5)/Ref($close, -1) - 1"], ["LABEL0"]

