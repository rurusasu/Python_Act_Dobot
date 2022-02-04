import sys
from queue import Queue
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Union
from threading import Thread, Timer

sys.path.append("../../")

import cv2

from lib.DobotFunction.Camera import SnapshotCvt, Contours

value = 1
value_2 = 10


class VisualFeedback2(Timer):
    def __init__(
        self,
        interval: float,
        function: Callable[..., Any],
        args: Union[Iterable[Any], None] = [],
        kwargs: Union[Mapping[str, Any], None] = {},
    ) -> None:
        Timer.__init__(self, interval, self.run, args, kwargs)
        self.thread = None
        self.function = function
        # self.data_que = Queue()  # ワーカープロセスへ送るデータ
        self.ui_que = Queue()  # ワーカーから送られてくるデータ

    def run(self):
        self.thread = Timer(self.interval, self.run)
        self.thread.start()
        self.function(self.ui_que, *self.args, **self.kwargs)

    def cancel(self):
        if self.thread is not None:
            self.thread.cancel()
            self.thread.join()
            del self.thread

    def GetValue(self):
        return self.ui_que.get_nowait()


def VF(ui_que: Queue, err, err_2) -> Queue:
    global value, value_2
    value += err
    value_2 += err_2

    return_param = {"value": value, "value_2": value_2}
    ui_que.put(return_param)


if __name__ == "__main__":
    import time

    err = 1
    err_2 = 2
    vf = VisualFeedback2(1, VF, {err, err_2})
    vf.start()
    for i in range(5):
        print(value, value_2)
        time.sleep(1)
    time.sleep(3)
    data = vf.GetValue()
    print(data)
    time.sleep(3)
    data = vf.GetValue()
    print(data)
    print(value, value_2)
    vf.cancel()
    data = vf.GetValue()
    print(data)
    print(value, value_2)
