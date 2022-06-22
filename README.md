# new_clothes_enalber_on_x86
First we need to deploy the `openvino` environment. And then, we should weak up it.

Like this:

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

Then, we should deploy the python environment for the clothes detection environment.

Like this:

```shell
source activate 'your env environment name'
```

Then `cd` our file path,

Like this:

```shell
python3 clothes_detection.py -i 'your test video path' -m 'your model xml file' -at yolo -ds modanet --output_limit 0 -d CPU -r
```
