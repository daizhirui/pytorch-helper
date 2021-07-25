from typing import Any
from typing import Collection
from typing import Hashable
from typing import Optional
from typing import Union

import numpy as np

from .io import load_result
from .io import save_result


class Meter(object):
    class Op:
        APPEND = 1
        EXTEND = 2

    def __init__(self):
        """ Meter is designed for tracking average and sum
        """
        self.data = dict()
        self.cnt = dict()

    def __getitem__(self, item: Hashable) -> Any:
        return self.data[item]

    def __setitem__(self, key: Hashable, value: Any):
        self.data[key] = value

    def __contains__(self, item: Hashable) -> bool:
        return item in self.data

    def _delete_tag(self, tag: Hashable):
        if tag in self.data:
            del self.data[tag]
        if tag in self.cnt:
            del self.cnt[tag]

    def reset(
            self,
            tag: Optional[Union[str, Collection[Hashable], Hashable]] = None
    ):
        """ remove the data of tag

        :param tag: str of tag
        """
        if tag is None:
            self.data = dict()
        elif isinstance(tag, str):
            self._delete_tag(tag)
        elif isinstance(tag, Collection):
            for t in tag:
                self._delete_tag(t)
        else:
            self._delete_tag(tag)

    def record(self, tag: Hashable, value: Any, op=Op.EXTEND):
        """ store `value` for `tag`

        :param tag: str of tag
        :param value: data to record
        :param op: Op.EXTEND to treat `value` as a list of values or Op.APPEND
            to treat `value` as a single value
        """
        self.data.setdefault(tag, [])
        if op == self.Op.APPEND:
            self.data[tag].append(value)
        elif isinstance(value, Collection):
            self.data[tag].extend(value)
        else:
            self.data[tag].append(value)

    def record_running_mean(
            self, tag: Hashable, value: Any, weight: Any, op=Op.APPEND
    ):
        """ update the running mean of `tag`

        :param tag: str of the running mean tag
        :param value: new data to update the running mean
        :param weight: the weight of value
        :param op: Op.APPEND or Op.EXTEND
        """
        cnt = self.cnt.get(tag, 0)
        v = self.data.get(tag, 0)

        if isinstance(value, Collection) and op != self.Op.APPEND:
            v = v * cnt
            for vv in value:
                v = v + vv * weight
                cnt += weight
            v = v / cnt
        else:
            v = cnt * v + value * weight
            cnt += weight
            v = v / cnt
        self.cnt[tag] = cnt
        self.data[tag] = v

    def record_running_sum(
            self, tag: Hashable, value: Any, weight: Any, op=Op.APPEND
    ):
        """ update the running sum of `tag`

        :param tag: str of the running sum tag
        :param value: new data to update the running sum
        :param weight: the weight of value
        :param op: Op.APPEND or Op.EXTEND
        """
        cnt = self.cnt.get(tag, 0)
        v = self.data.get(tag, 0)

        if isinstance(value, Collection) and op != self.Op.APPEND:
            v = v * cnt
            for vv in value:
                v = v + vv * weight
                cnt += weight
        else:
            v = cnt * v + value * weight
            cnt += weight
        self.cnt[tag] = cnt
        self.data[tag] = v

    def concat(
            self, tag: Union[str, Collection[Hashable], Hashable], axis: int
    ):
        """ concatenate the recorded values of `tag` together

        :param tag: str of tag
        :param axis: int of the axis to concatenate
        :return:
        """
        if not isinstance(tag, Collection) or isinstance(tag, str):
            tag = [tag]
        for t in tag:
            self.data[t] = np.concatenate(self.data[t], axis=axis)

    def mean(
            self,
            tag: Optional[Union[str, Collection[Hashable], Hashable]] = None,
            axis: Optional[int] = None
    ) -> Union[dict, np.ndarray]:
        """ calculate the mean of `tag` along axis `axis`

        :param tag: str of the data tag
        :param axis: int of the axis to calculate the mean
        :return: dict or numpy.ndarray
        """
        if tag is None:
            return {k: v if k in self.cnt else np.mean(v, axis=axis)
                    for k, v in self.data.items()}
        elif isinstance(tag, Collection) and not isinstance(tag, str):
            return {t: self.mean(t, axis=axis) for t in tag}
        else:
            return self.data[tag] if tag in self.cnt else \
                np.mean(self.data[tag], axis=axis)

    def sum(
            self,
            tag: Optional[Union[str, Collection[Hashable], Hashable]] = None,
            axis: Optional[int] = None
    ) -> Union[dict, np.ndarray]:
        """ calculate the sum of `tag` along axis `axis`

        :param tag: str of the data tag
        :param axis: int of the axis to calculate the sum
        :return: dict or numpy.ndarray
        """
        if tag is None:
            return {k: np.sum(v, axis=axis) for k, v in self.data.items()}
        elif isinstance(tag, Collection) and not isinstance(tag, str):
            return {t: np.sum(self.data[t], axis=axis) for t in tag}
        else:
            return np.sum(np.stack(self.data[tag]), axis=axis)

    def save(self, path: str):
        """ save `self.data` and `self.cnt` to `path` as a pickle file

        :param path: str of pickle file path
        """
        save_result({'data': self.data, 'cnt': self.cnt}, path)

    @staticmethod
    def load(path: str):
        """ load a Meter from `path`

        :param path: str of the pickle file to recover Meter from
        :return: a Meter instance
        """
        meter = Meter()
        for key, value in load_result(path).items():
            setattr(meter, key, value)
        return meter

    def __len__(self):
        """ get the number of values of each tag

        :return: dict
        """
        return {key: len(value) for key, value in self.data.items()}
