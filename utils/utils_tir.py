import numpy as np

from typing import Dict, Union

import openpyxl

def get_stats_bg_tir(arr: np.ndarray, prefix: str=None, is_symbol_in_prefix: bool=True) -> Dict[str, Union[int, float]]:
    if prefix is None:
        prefix = ''
        prefix_hash = '#'
        prefix_perc = '%'
    else:
        if is_symbol_in_prefix:
            prefix_hash = f'#{prefix}\n'
            prefix_perc = f'%{prefix}\n'
        else:
            prefix_hash = f'{prefix}\n#'
            prefix_perc = f'{prefix}\n%'

    amount_all = arr.shape[0]

    amount_lt54 = np.sum(arr < 54)
    amount_54_69 = np.sum((arr >= 54) & (arr <= 69))
    amount_lt70 = np.sum(arr < 70)
    amount_70_180 = np.sum((arr >= 70) & (arr <= 180))
    amount_gt180 = np.sum(arr > 180)
    amount_181_250 = np.sum((arr >= 181) & (arr <= 250))
    amount_gt250 = np.sum(arr > 250)

    d_values = {
        f'#{prefix}': amount_all,
        
        f'{prefix_hash}<54': amount_lt54,
        f'{prefix_hash}54-69': amount_54_69,
        f'{prefix_hash}<70': amount_lt70,
        f'{prefix_hash}70-180': amount_70_180,
        f'{prefix_hash}181-250': amount_181_250,
        f'{prefix_hash}>180': amount_gt180,
        f'{prefix_hash}>250': amount_gt250,
    }

    d_other = {}
    if amount_all > 0:
        d_other[f'{prefix_perc}<54'] = amount_lt54 / amount_all
        d_other[f'{prefix_perc}54-69'] = amount_54_69 / amount_all
        d_other[f'{prefix_perc}<70'] = amount_lt70 / amount_all
        d_other[f'{prefix_perc}70-180'] = amount_70_180 / amount_all
        d_other[f'{prefix_perc}>180'] = amount_gt180 / amount_all
        d_other[f'{prefix_perc}181-250'] = amount_181_250 / amount_all
        d_other[f'{prefix_perc}>250'] = amount_gt250 / amount_all
    else:
        d_other[f'{prefix_perc}<54'] = None
        d_other[f'{prefix_perc}54-69'] = None
        d_other[f'{prefix_perc}<70'] = None
        d_other[f'{prefix_perc}70-180'] = None
        d_other[f'{prefix_perc}>180'] = None
        d_other[f'{prefix_perc}181-250'] = None
        d_other[f'{prefix_perc}>250'] = None

    d_values.update(d_other)

    return d_values


def get_stats_bg_tir_changes(arr: np.ndarray, prefix: Union[str, None]=None, is_symbol_in_prefix: bool=True) -> Dict[str, Union[int, float]]:
    if prefix is None:
        prefix = ''
        prefix_hash = '#'
        prefix_perc = '%'
    else:
        if is_symbol_in_prefix:
            prefix_hash = f'#{prefix}\n'
            prefix_perc = f'%{prefix}\n'
        else:
            prefix_hash = f'{prefix}\n#'
            prefix_perc = f'{prefix}\n%'

    state_lt_54 = '<54'
    state_54_69 = '54-69'
    state_70_180 = '70-180'
    state_181_250 = '181_250'
    state_gt_250 = '>250'

    l_state = [
        state_lt_54,
        state_54_69,
        state_70_180,
        state_181_250,
        state_gt_250,
    ]

    def get_state_from_value(v: int) -> str:
        state: Union[str, None] = None
        if v < 54:
            state = state_lt_54
        elif v >= 54 and v <= 69:
            state = state_54_69
        elif v >= 70 and v <= 180:
            state = state_70_180
        elif v >= 181 and v <= 250:
            state = state_181_250
        elif v > 250:
            state = state_gt_250
        else:
            assert False

        return state

    val_prev = arr[0]
    state_prev = get_state_from_value(val_prev)

    d_state_count_val = {state: 0 for state in l_state}
    d_state_count_range_change = {state: 0 for state in l_state}
    d_state_count_state_pair = {(state1, state2): 0 for state1 in l_state for state2 in l_state if state1 != state2}

    d_state_count_val[state_prev] += 1
    d_state_count_range_change[state_prev] += 1

    for val in arr[1:]:
        state = get_state_from_value(val)

        d_state_count_val[state] += 1
        d_state_count_state_pair[(state_prev, state)] += 1

        # now do every single state changes!
        if state_prev == state_lt_54 and state == state_54_69:
            d_state_count_range_change[state_54_69] += 1
        elif state_prev == state_lt_54 and state == state_70_180:
            d_state_count_range_change[state_54_69] += 1
            d_state_count_range_change[state_70_180] += 1
        elif state_prev == state_lt_54 and state == state_181_250:
            d_state_count_range_change[state_54_69] += 1
            d_state_count_range_change[state_70_180] += 1
            d_state_count_range_change[state_181_250] += 1
        elif state_prev == state_lt_54 and state == state_gt_250:
            d_state_count_range_change[state_54_69] += 1
            d_state_count_range_change[state_70_180] += 1
            d_state_count_range_change[state_181_250] += 1
            d_state_count_range_change[state_gt_250] += 1

        elif state_prev == state_54_69 and state == state_lt_54:
            d_state_count_range_change[state_lt_54] += 1
        elif state_prev == state_54_69 and state == state_70_180:
            d_state_count_range_change[state_70_180] += 1
        elif state_prev == state_54_69 and state == state_181_250:
            d_state_count_range_change[state_70_180] += 1
            d_state_count_range_change[state_181_250] += 1
        elif state_prev == state_54_69 and state == state_gt_250:
            d_state_count_range_change[state_70_180] += 1
            d_state_count_range_change[state_181_250] += 1
            d_state_count_range_change[state_gt_250] += 1

        elif state_prev == state_70_180 and state == state_lt_54:
            d_state_count_range_change[state_54_69] += 1
            d_state_count_range_change[state_lt_54] += 1
        elif state_prev == state_70_180 and state == state_54_69:
            d_state_count_range_change[state_54_69] += 1
        elif state_prev == state_70_180 and state == state_181_250:
            d_state_count_range_change[state_181_250] += 1
        elif state_prev == state_70_180 and state == state_gt_250:
            d_state_count_range_change[state_181_250] += 1
            d_state_count_range_change[state_gt_250] += 1

        elif state_prev == state_181_250 and state == state_lt_54:
            d_state_count_range_change[state_70_180] += 1
            d_state_count_range_change[state_54_69] += 1
            d_state_count_range_change[state_lt_54] += 1
        elif state_prev == state_181_250 and state == state_54_69:
            d_state_count_range_change[state_70_180] += 1
            d_state_count_range_change[state_54_69] += 1
        elif state_prev == state_181_250 and state == state_70_180:
            d_state_count_range_change[state_70_180] += 1
        elif state_prev == state_181_250 and state == state_gt_250:
            d_state_count_range_change[state_gt_250] += 1

        elif state_prev == state_gt_250 and state == state_lt_54:
            d_state_count_range_change[state_181_250] += 1
            d_state_count_range_change[state_70_180] += 1
            d_state_count_range_change[state_54_69] += 1
            d_state_count_range_change[state_lt_54] += 1
        elif state_prev == state_gt_250 and state == state_54_69:
            d_state_count_range_change[state_181_250] += 1
            d_state_count_range_change[state_70_180] += 1
            d_state_count_range_change[state_54_69] += 1
        elif state_prev == state_gt_250 and state == state_70_180:
            d_state_count_range_change[state_181_250] += 1
            d_state_count_range_change[state_70_180] += 1
        elif state_prev == state_gt_250 and state == state_181_250:
            d_state_count_range_change[state_181_250] += 1

        else:
            assert state_prev == state

        state_prev = state

    d_state_count_val
    d_state_count_range_change
    d_state_count_state_pair

    d_values = {
        f'#{prefix}': amount_all,
        
        f'{prefix_hash}<54': amount_lt54,
        f'{prefix_hash}54-69': amount_54_69,
        f'{prefix_hash}<70': amount_lt70,
        f'{prefix_hash}70-180': amount_70_180,
        f'{prefix_hash}181-250': amount_181_250,
        f'{prefix_hash}>180': amount_gt180,
        f'{prefix_hash}>250': amount_gt250,
    }

    d_other = {}
    if amount_all > 0:
        d_other[f'{prefix_perc}<54'] = amount_lt54 / amount_all
        d_other[f'{prefix_perc}54-69'] = amount_54_69 / amount_all
        d_other[f'{prefix_perc}<70'] = amount_lt70 / amount_all
        d_other[f'{prefix_perc}70-180'] = amount_70_180 / amount_all
        d_other[f'{prefix_perc}>180'] = amount_gt180 / amount_all
        d_other[f'{prefix_perc}181-250'] = amount_181_250 / amount_all
        d_other[f'{prefix_perc}>250'] = amount_gt250 / amount_all
    else:
        d_other[f'{prefix_perc}<54'] = None
        d_other[f'{prefix_perc}54-69'] = None
        d_other[f'{prefix_perc}<70'] = None
        d_other[f'{prefix_perc}70-180'] = None
        d_other[f'{prefix_perc}>180'] = None
        d_other[f'{prefix_perc}181-250'] = None
        d_other[f'{prefix_perc}>250'] = None

    d_values.update(d_other)




def get_simple_stats_calc(arr: np.ndarray, prefix: str=None, d_rename_suffix: Dict[str, str]={}) -> Dict[str, Union[int, float]]:
    if prefix is None:
        prefix = ''
    else:
        prefix = f'{prefix}\n'

    d_name_suffix = {
        'n': 'n',
        'mean': 'mean',
        'std': 'std',
        'cv': 'cv', # variabiltity
        'min': 'min',
        'q1': 'q1',
        'median': 'median',
        'q3': 'q3',
        'max': 'max',
    }

    for k in d_name_suffix:
        if k in d_rename_suffix:
            d_name_suffix[k] = d_rename_suffix[k]

    d_values = {}
    d_values[f'{prefix}{d_name_suffix["n"]}'] = arr.shape[0]

    if arr.shape[0] > 0:
        val_mean = np.mean(arr)
        val_std = np.std(arr)
        d_values[f'{prefix}{d_name_suffix["mean"]}'] = val_mean
        d_values[f'{prefix}{d_name_suffix["std"]}'] = val_std
        if val_mean > 0.:
            d_values[f'{prefix}{d_name_suffix["cv"]}'] = val_std / val_mean
        else:
            d_values[f'{prefix}{d_name_suffix["cv"]}'] = None

        d_values[f'{prefix}{d_name_suffix["min"]}'] = np.min(arr)
        d_values[f'{prefix}{d_name_suffix["q1"]}'] = np.quantile(arr, 0.25)
        d_values[f'{prefix}{d_name_suffix["median"]}'] = np.median(arr)
        d_values[f'{prefix}{d_name_suffix["q3"]}'] = np.quantile(arr, 0.75)
        d_values[f'{prefix}{d_name_suffix["max"]}'] = np.max(arr)
    else:
        d_values[f'{prefix}{d_name_suffix["mean"]}'] = None
        d_values[f'{prefix}{d_name_suffix["std"]}'] = None
        d_values[f'{prefix}{d_name_suffix["cv"]}'] = None
        d_values[f'{prefix}{d_name_suffix["min"]}'] = None
        d_values[f'{prefix}{d_name_suffix["q1"]}'] = None
        d_values[f'{prefix}{d_name_suffix["median"]}'] = None
        d_values[f'{prefix}{d_name_suffix["q3"]}'] = None
        d_values[f'{prefix}{d_name_suffix["max"]}'] = None
    
    return d_values
