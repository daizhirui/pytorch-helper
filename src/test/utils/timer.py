import time
from unittest import TestCase


class TestTimerManager(TestCase):

    def test_timer_manager(self):
        from pytorch_helper.utils.timer import TimerManager

        manager = TimerManager()

        with manager.timing('a') as timer:
            time.sleep(1)
            timer.lap()
            print(timer.laps)
            time.sleep(2)
            timer.lap()
            print(timer.laps)
            timer.stop()
            print(timer.elapsed)
            time.sleep(1)
            timer.start()
            time.sleep(4)
            print(timer.elapsed)
            timer.lap()
            time.sleep(1)
            print(timer.elapsed)
            print(timer.laps)
        time.sleep(1)
        timer.stop()
        print(timer.elapsed)

        with manager.timing('a') as timer:
            time.sleep(1)
        print(timer.elapsed)
