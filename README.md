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

The model path is here: https://drive.google.com/file/d/1BBQPvkp-jwQNM2P-VJLGX101PlUhjpKN/view?usp=sharing
Another link is : https://pan.baidu.com/s/1F3Ltc0XH1lMkdF_oDW55ZQ . And the password is : 1234.

**The python version is 3.6**

Then `cd` our file path,

Add then use this command to install all the dependence:
```shell
pip install -r requirement.txt
```

Like this:

```shell
python3 clothes_detection.py -i 'your test video path' -m 'your model xml file' -at yolo -ds modanet --output_limit 0 -d CPU -r
```

