# hawk_eye.inference.benchmark_inference

```{automodule} hawk_eye.inference.benchmark_inference
:members:
```
## Example

```
$ PYTHONPATH=. hawk_eye/inference/benchmark_inference.py \
  --timestamp 2020-09-05T15.51.57 \
  --model_type classifier \
  --batch_size 50 \
  --run_time 10
```
```
$ PYTHONPATH=. hawk_eye/inference/benchmark_inference.py \
  --timestamp 2020-10-10T14.02.09 \
  --model_type detector \
  --batch_size 10 \
  --run_time 15
```
