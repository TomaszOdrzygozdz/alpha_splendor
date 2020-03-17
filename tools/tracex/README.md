# TraceX

TraceX - Trace eXplorer.

## Quickstart

### Install

`pip install flask Pillow`

### Dump traces

You need to add `TraceCallback` to your `Agent` to collect traces. Gin snippet:

```
import alpacka.tracing

# Parameters for StochasticMCTSAgent:
# ==============================================================================
StochasticMCTSAgent.callback_classes = (@alpacka.tracing.TraceCallback,)

# Parameters for TraceCallback:
# ==============================================================================
TraceCallback.output_dir = './traces/'
TraceCallback.sample_rate = 0.01
```

`output_dir` is where the traces are saved. `sample_rate` is the fraction of episodes that are traced.

### Run the server

`python explore.py path/to/trace/file`

Then open `localhost:5000` in your browser.
