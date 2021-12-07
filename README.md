# Convert Models To Inference Mode

### Test Specifications
* jetpack-4.2.2 [L4T 32.2.1]
* CUDA-10.0.326
* cuDNN-7.5.0.56-1+cuda10.0
* TensorRT-5.1.6.1-1+cuda10.0
* Tensorflow-1.12.2(source)
* Keras-2.2.4

* **usage :**
``` bash
python3 example.py --c ./inference_converter/configs/convertor_<convert_mode>_config.json
```
#### 1-Mode ***keras2chpt***

* Convert keras model to tensorflow check point files.

##### 1.1 Custom Keras Model
* You need to change **TODO** part of *inference.py* from utils directory related to *chpt_convert* method.

##### 1.2 Tips

* It's better to import everything with keras and don't mix up tensorflow.
like **keras.models.load_model** instead of **tf.models.load_model** and get session with pure keras backed not with tensorflow.
* keras lower versions had a problem when we want to use the model in the inference phase. we must set **learning_phase to zero**.
* For custom layers, we must define **custom objects** when we want to load the saved model.
* For custom layers class, we must define **get_config** method.

#### 2-Mode ***chpt2pb***

* Convert Tensorflow checkpoint files(.chpt) to inference frozen graph(.pb) file.

##### 2.1 Tips

* The outputs graph node names should be passed without **:0 or :1** and so on.

#### 3-Mode ***pb2trt***
* Convert Tensorflow inference frozen graph files(.pb) to tensorrt-tensorflow graph(.pb) file.

#### 4-Unit Test
``` bash
python3 -m unittest test_inference_convertor.py
```

Output:
``` bash
Ran 3 tests in 24.708s

OK
```
