import sys
from io import StringIO
import neptune

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


with Capturing() as pupu:
    neptune.init(project_qualified_name='tomaszodrzygozdz/Splendor')
    neptune.create_experiment('ddd', 'bbb', upload_stdout=False)

print(f'pupu = {pupu}')