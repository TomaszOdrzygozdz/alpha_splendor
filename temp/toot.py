import atexit
import time
def pupu():
    print('ufer')

try:
    while True:
        time.sleep(2)

except KeyboardInterrupt:
    pupu()