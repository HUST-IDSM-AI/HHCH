import datetime
import numpy
import platform


class Logger:

    @staticmethod
    def divider(title):
        print(
            "\n============================================{}=============================================\n".format(
                title), flush=True)

    @staticmethod
    def info(message):
        time = datetime.datetime.now()
        timeStr = time.strftime("[%Y%m%d-%H:%M:%S]")
        print(timeStr + '=>[info]: {}'.format(message), flush=True)

    @staticmethod
    def getTimeStr(time):
        if platform.system().lower() != 'windows':
            timeStr = time.strftime("[%m%d-%H:%M:%S]")
        else:
            timeStr = time.strftime("[%m%d-%H-%M-%S]")
        return timeStr


if __name__ == '__main__':
    a = numpy.arange(1, 10, 1)
    Logger.info("test {}".format(a))
    time = datetime.datetime.now()
    print(Logger.getTimeStr(time))
