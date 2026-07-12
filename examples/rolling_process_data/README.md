# Rolling Process Data

This workflow is an example for `Rolling Process Data`.

## Background

When rolling train the models, data also needs to be generated in the different rolling windows. When the rolling window moves, the training data will change, and the processor's learnable state (such as standard deviation, mean, etc.) will also change. 

In order to avoid regenerating data, this example uses the `DataHandler-based DataLoader` to load the raw features that are not related to the rolling window, and then used Processors to generate processed-features related to the rolling window.

The example-specific `rolling_handler.Alpha158` keeps the regular Alpha158-style configuration,
automatically creates or reuses a hash-named raw Handler cache under `datacache/handler_cache`,
and delegates rolling processing to `RollingDataHandler`. The rolling Processors are not stored
in the raw cache, so their learnable state is fitted again for each rolling training window.


## Run the Code

Run the example by running the following command:
```bash
    python workflow.py rolling_process
```
