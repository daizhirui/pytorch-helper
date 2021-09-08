from enum import Enum
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Collection
from typing import Dict
from typing import Final
from typing import Iterable

import numpy as np

from .io import load_from_pickle
from .io import save_as_pickle
from .. import __version__

__all__ = ['Meter']


class MeterItem:
    class RecordOp(Enum):
        APPEND = 0
        EXTEND = 1

    class ReduceOp(Enum):
        STORE = 0
        SUM = 1

    def __init__(self, record_op: RecordOp, reduce_op: ReduceOp):
        self.record_op: Final = record_op
        self.reduce_op: Final = reduce_op
        self.record: Final[Callable] = self._store_record \
            if reduce_op == self.ReduceOp.STORE else self._running_sum_record

        self.data = [] if reduce_op == self.ReduceOp.STORE else 0.
        self.value_cnt = 0
        self.weight_cnt = 0

    def state_dict(self):
        return dict(
            record_op=self.record_op,
            reduce_op=self.reduce_op,
            data=self.data,
            value_cnt=self.value_cnt,
            weight_cnt=self.weight_cnt,
            version=__version__  # reserved for future use
        )

    def load_state_dict(self, state_dict):
        for key in self.state_dict():
            if key == 'version':
                continue
            setattr(self, key, state_dict[key])

    def _pre_process(self, value, weight):
        if self.record_op == self.RecordOp.APPEND:
            value = [value]
        if not isinstance(weight, Collection):
            weight = [weight] * len(value)
        return value, weight

    def _store_record(self, value: Any, weight=1):
        value, weight = self._pre_process(value, weight)

        for w, v in zip(weight, value):
            self.data.append(np.array(v) * w)
            self.weight_cnt += w
        self.value_cnt = len(self.data)

    def _running_sum_record(self, value: Any, weight=1):
        value, weight = self._pre_process(value, weight)

        for w, v in zip(weight, value):
            self.data += np.array(v, dtype=float) * w
            self.value_cnt += 1
            self.weight_cnt += w

    def sum(self):
        """ get the sum of all the recorded values.

        :return: sum
        """
        if self.reduce_op == self.ReduceOp.STORE:
            return np.sum(np.stack(self.data), axis=0)
        else:
            return self.data

    def mean(self):
        """ get the mean of all the recorded values

        :return: mean
        """
        return self.sum() / self.weight_cnt

    def reset(self):
        self.data = [] if self.reduce_op == self.ReduceOp.STORE else 0.
        self.value_cnt = 0
        self.weight_cnt = 0


class Meter(object):
    RecordOp: ClassVar = MeterItem.RecordOp
    ReduceOp: ClassVar = MeterItem.ReduceOp

    def __init__(self):
        """ Meter is designed for tracking average and sum
        """
        self.meter_items: Dict[str, MeterItem] = dict()
        # self.data = dict()
        # self.cnt = dict()

    def __getitem__(self, tag: str) -> MeterItem:
        return self.meter_items[tag]
        # return self.data[tag]

    # def __setitem__(self, key: Hashable, value: Any):
    #     self.data[key] = value

    def __contains__(self, tag: str) -> bool:
        return tag in self.meter_items
        # return tag in self.data

    def _delete_tag(self, tag: str):
        if tag in self.meter_items:
            del self.meter_items[tag]
        # if tag in self.data:
        #     del self.data[tag]
        # if tag in self.cnt:
        #     del self.cnt[tag]

    def reset(self, tag: str = None):
        """ remove the data of tag

        :param tag: str of tag, if None, remove all the data
        """
        if tag is None:
            tags = self.meter_items.keys()
        else:
            tags = [tag]
        for tag in tags:
            self.meter_items[tag].reset()
        # if tag is None:
        #     self.data = dict()
        # else:
        #     self._delete_tag(tag)

    def reset_tags(self, tags: Iterable[str] = None):
        """ apply `self.reset` on each element in `tags`

        :param tags: Iterable of tags, if None, remove all the data
        """
        if tags is None:
            tags = self.meter_items.keys()
        for tag in tags:
            self.meter_items[tag].reset()
        # if tags is None:
        #     self.data = dict()
        # else:
        #     for t in tags:
        #         self._delete_tag(t)

    def record(
        self, tag: str, value: Any, weight=1, record_op: RecordOp = None,
        reduce_op: ReduceOp = None
    ):
        if tag not in self.meter_items:
            assert record_op is not None, \
                f'Need record_op to create a new {MeterItem.__name__}'
            assert reduce_op is not None, \
                f'Need reduce_op to create a new {MeterItem.__name__}'
            self.meter_items[tag] = MeterItem(record_op, reduce_op)
        meter = self.meter_items[tag]
        meter.record(value, weight)

    # def record(self, tag: Hashable, value: Any, op=Op.EXTEND):
    #     """ store `value` for `tag`
    #
    #     :param tag: str of tag
    #     :param value: data to record
    #     :param op: Op.EXTEND to treat `value` as a list of values or Op.APPEND
    #         to treat `value` as a single value
    #     """
    #     self.data.setdefault(tag, [])
    #     if op == self.Op.APPEND:
    #         self.data[tag].append(value)
    #     elif isinstance(value, Iterable):
    #         self.data[tag].extend(value)
    #     else:
    #         self.data[tag].append(value)
    #
    # def record_running_mean(
    #         self, tag: Hashable, value: Any, weight: Any, op=Op.APPEND
    # ):
    #     """ update the running mean of `tag`
    #
    #     :param tag: str of the running mean tag
    #     :param value: new data to update the running mean
    #     :param weight: the weight of value
    #     :param op: Op.APPEND or Op.EXTEND
    #     """
    #     cnt = self.cnt.get(tag, 0)
    #     v = self.data.get(tag, 0)
    #
    #     if isinstance(value, Iterable) and op != self.Op.APPEND:
    #         v = v * cnt
    #         for vv in value:
    #             v = v + vv * weight
    #             cnt += weight
    #         v = v / cnt
    #     else:
    #         v = cnt * v + value * weight
    #         cnt += weight
    #         v = v / cnt
    #     self.cnt[tag] = cnt
    #     self.data[tag] = v
    #
    # def record_running_sum(
    #         self, tag: Hashable, value: Any, weight: Any, op=Op.APPEND
    # ):
    #     """ update the running sum of `tag`
    #
    #     :param tag: str of the running sum tag
    #     :param value: new data to update the running sum
    #     :param weight: the weight of value
    #     :param op: Op.APPEND or Op.EXTEND
    #     """
    #     cnt = self.cnt.get(tag, 0)
    #     v = self.data.get(tag, 0)
    #
    #     if isinstance(value, Iterable) and op != self.Op.APPEND:
    #         for vv in value:
    #             v = v + vv * weight
    #             cnt += weight
    #     else:
    #         v = v + value * weight
    #         cnt += weight
    #     self.cnt[tag] = cnt
    #     self.data[tag] = v

    # def concat(self, tag: Hashable, axis: int) -> np.ndarray:
    #     """ concatenate the recorded numpy arrays of `tag` together
    #
    #     :param tag: str of tag
    #     :param axis: int of the axis to concatenate
    #     """
    #     self.data[tag] = np.concatenate(self.data[tag], axis=axis)
    #     return self.data[tag]
    #
    def mean(self, tag: str):
        return self.meter_items[tag].mean()

    def means(self, tags: Iterable[str] = None):
        if tags is None:
            tags = self.meter_items.keys()
        return {tag: self.meter_items[tag].mean() for tag in tags}

    # def mean(
    #         self, tag: Optional[Hashable] = None, axis: Optional[int] = None
    # ) -> Union[dict, np.ndarray]:
    #     """ calculate the mean of `tag` along axis `axis`
    #
    #     :param tag: str of the data tag
    #     :param axis: int of the axis to calculate the mean
    #     :return: dict or numpy.ndarray
    #     """
    #     if tag is None:
    #         return {k: v if k in self.cnt else np.mean(v, axis=axis)
    #                 for k, v in self.data.items()}
    #     else:
    #         return self.data[tag] if tag in self.cnt else \
    #             np.mean(self.data[tag], axis=axis)
    #
    # def sum(
    #         self,
    #         tag: Optional[Union[str, Collection[Hashable], Hashable]] = None,
    #         axis: Optional[int] = None
    # ) -> Union[dict, np.ndarray]:
    #     """ calculate the sum of `tag` along axis `axis`
    #
    #     :param tag: str of the data tag
    #     :param axis: int of the axis to calculate the sum
    #     :return: dict or numpy.ndarray
    #     """
    #     if tag is None:
    #         return {k: np.sum(v, axis=axis) for k, v in self.data.items()}
    #     elif isinstance(tag, Collection) and not isinstance(tag, str):
    #         return {t: np.sum(self.data[t], axis=axis) for t in tag}
    #     else:
    #         return np.sum(np.stack(self.data[tag]), axis=axis)

    def state_dict(self):
        return dict(
            meter_items={
                k: v.state_dict() for k, v in self.meter_items.items()
            },
            version=__version__
        )

    def load_state_dict(self, state_dict):
        for tag, state in state_dict['meter_items'].items():
            item = MeterItem(state['record_op'], state['reduce_op'])
            item.load_state_dict(state)
            self.meter_items[tag] = item

    def save(self, path: str):
        """ save `self.data` and `self.cnt` to `path` as a pickle file

        :param path: str of pickle file path
        """
        save_as_pickle(path, self.state_dict())
        # save_as_pickle(path, {'data': self.data, 'cnt': self.cnt})

    @staticmethod
    def load(path: str):
        """ load a Meter from `path`

        :param path: str of the pickle file to recover Meter from
        :return: a Meter instance
        """
        meter = Meter()
        meter.load_state_dict(load_from_pickle(path))
        # for key, value in load_from_pickle(path).items():
        #     setattr(meter, key, value)
        return meter

    # def __len__(self):
    #     """ get the number of values of each tag
    #
    #     :return: dict
    #     """
    #     return {key: len(value) for key, value in self.data.items()}
