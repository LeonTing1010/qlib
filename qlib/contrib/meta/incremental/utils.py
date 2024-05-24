import traceback
import warnings
from collections import defaultdict
import numpy as np
import torch
from typing import List

from torch import nn

from qlib.data.dataset import TSDataSampler


def get_data_and_idx(data: TSDataSampler, segment: List):
    d_idx = data.idx_df.loc(axis=0)[segment[0]: segment[1]].to_numpy().flatten()
    if len(d_idx) == 0:
        return None, None
    d_idx = d_idx[~np.isnan(d_idx.astype(np.float64))]
    d = data[d_idx]
    indices = data.data_index[d_idx.astype(int)]
    return d, indices


def get_data_from_seg(seg, data, test=False):
    try:
        d = (
            data.loc(axis=0)[seg[0]: seg[1]]
            if not test or seg[1] <= str(data.index[-1][0])
            else data.loc(axis=0)[seg[0]:]
        )
    except Exception as e:
        traceback.print_exc()
        new_seg = [seg[0], seg[1]]
        all_dates = data.index.levels[0]
        if seg[0] not in all_dates:
            new_seg[0] = all_dates[all_dates > seg[0]][0]
            if str(new_seg[0])[:10] > seg[1]:
                warnings.warn(f"Exceed test time{new_seg}")
                return None
        if seg[1] not in all_dates:
            new_seg[1] = all_dates[all_dates < seg[1]][-1]
            if str(new_seg[1])[:10] < seg[0]:
                warnings.warn(f"Exceed training time{new_seg}")
                return None
            d = (
                data.loc(axis=0)[new_seg[0]: new_seg[1]]
                if not test or new_seg[1] <= all_dates[-1]
                else data.loc(axis=0)[new_seg[0]:]
            )
        else:
            d = (
                data.loc(axis=0)[new_seg[0]: new_seg[1]]
                if not test or new_seg[1] <= str(all_dates[-1])
                else data.loc(axis=0)[new_seg[0]:]
            )
        warnings.warn(f"{seg} becomes {new_seg} after adjustment")
    return d


def override_state(groups, new_opt):
    saved_groups = new_opt.param_groups
    id_map = {old_id: p for old_id, p in zip(range(len(saved_groups[0]["params"])), groups[0]["params"])}
    state = defaultdict(dict)
    for k, v in new_opt.state[0].items():
        if k in id_map:
            param = id_map[k]
            for _k, _v in v.items():
                state[param][_k] = _v.detach() if isinstance(_v, torch.Tensor) else _v
        else:
            state[k] = v
    return state


def _mask_mlp158(meta_input):
    # X_test = meta_input["X_test"]
    y_test = meta_input["y_test"]
    # mask_x = torch.isnan(X_test).sum(-1) == 0
    # meta_input["X_test"] = X_test[mask_x]
    # y_test = y_test[mask_x]

    mask_y = ~torch.isnan(y_test)
    meta_input["mask_y"] = mask_y
    # meta_input["y_test"] = y_test[mask_y]
    # test_idx = meta_input["test_idx"]
    # meta_input["test_idx"] = test_idx[np.arange(len(test_idx))[mask_x][mask_y]]
    return meta_input


def preprocess(
    task_list, factor_num=6, is_mlp=False, alpha=360, step=20, H=1, not_sequence=False, to_tensor=True,
):
    skip_ids = []
    for i, task in enumerate(task_list):
        meta_input = task.get_meta_input()
        data_type = set()
        for k in meta_input.keys():
            if k.startswith("X") or k.startswith("y"):
                data_type.add(k[2:])
                if not isinstance(meta_input[k], np.ndarray):
                    meta_input[k] = meta_input[k].to_numpy()
                if to_tensor:
                    meta_input[k] = torch.tensor(meta_input[k], dtype=torch.float32)
        if task.processed_meta_input['y_test'].shape[0] == 0:
            skip_ids.append(i)
        if is_mlp and alpha == 158:
            _mask_mlp158(meta_input)

        if not_sequence:
            if alpha == 158:
                for dt in data_type:
                    k = "X_" + dt
                    meta_input[k] = meta_input[k].reshape(len(meta_input[k]), -1)
        elif alpha == 360:
            for dt in data_type:
                k = "X_" + dt
                meta_input[k] = meta_input[k].reshape(len(meta_input[k]), factor_num, -1)
                if isinstance(meta_input[k], torch.Tensor):
                    meta_input[k] = meta_input[k].permute(0, 2, 1)
                else:
                    meta_input[k] = meta_input[k].transpose(0, 2, 1)

        test_date = meta_input["test_idx"].codes[0] - meta_input["test_idx"].codes[0][0]
        meta_input["meta_end"] = (test_date <= (test_date[-1] - H + 1)).sum()

    if skip_ids:
        ''' Delete tasks with empty test data '''
        i = 0
        while i < len(skip_ids):
            task_w_train_wo_test = task_list[skip_ids[i]]
            # loop to skip continuous empty test data.
            while i + 1 < len(skip_ids) and skip_ids[i + 1] == skip_ids[i] + 1:
                i += 1
            # When test data comes, the current task has no training data and should look back
            not_skip = skip_ids[i] + 1
            if not_skip < len(task_list):
                for key in task_w_train_wo_test.keys():
                    if 'train' in key:
                        task_list[not_skip][key] = task_w_train_wo_test[key]
            i += 1

        task_list = [task_list[i] for i in range(len(task_list)) if i not in skip_ids]
    return task_list


def has_rnn(module: nn.Module):
    for module in module.modules():
        if isinstance(module, nn.RNNBase):
            return True
    return False
