import keras
import sys
import keras.backend as K
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

from tensorflow.python.platform import gfile
from keras.models import load_model
from models import PriorBox
from models import SSD


class Convertor:
    def __init__(self, tf_pb_path='', trt_pb_path='',
                 h5_path_in='path_to_load', h5_path_out='path_to _save',
                 tf_chpt_path='', node_outputs_name='', max_batch_size=1,
                 input_shape=(300, 300, 3), num_classes=21,
                 max_workspace_size_bytes=1e9, minimum_segment_size=3,
                 precision_mode="FP16"):
        self.h5_path_in = h5_path_in
        self.h5_path_out = h5_path_out
        self.tf_pb_path = tf_pb_path
        self.trt_pb_path = trt_pb_path
        self.node_outputs_name = node_outputs_name
        self.max_batch_size = max_batch_size
        self.max_workspace_size_bytes = max_workspace_size_bytes
        self.minimum_segment_size = minimum_segment_size
        self.precision_mode = precision_mode
        self.tf_chpt_path = tf_chpt_path
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_graph(self):
        model = tf.Graph()
        with model.as_default():
            model_graph_def = tf.GraphDef()
            try:
                with tf.gfile.GFile(self.tf_pb_path, "rb") as f:
                    serialized_graph = f.read()
                    model_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(model_graph_def, name="")
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                model_sess = tf.Session(graph=model, config=config)
            except:
                raise Exception('No Weight File Model!')
        return model_graph_def, model_sess

    def chpt_convert(self):
        K.set_learning_phase(0)

        ## TODO 
        ## This part is related to model initialization
        ## and then load the weights and create custom layers if there.
        model = SSD(self.input_shape, self.num_classes)
        model.load_weights(self.h5_path_in)
        model.save(self.h5_path_out)

        custom_objects = {'PriorBox': PriorBox}
        model = load_model(self.h5_path_out, custom_objects=custom_objects)
        ##############################################################

        saver = tf.train.Saver()
        sess = keras.backend.get_session()
        saver.save(sess, self.tf_chpt_path)

        print('[INFO] The Model Inputs : {}'.format(model.inputs))
        print('[INFO] The Model Outputs : {}'.format(model.outputs))
        print("Keras model is successfully converted to Tensorflow model.")

    def pb_convert(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            saver = tf.train.import_meta_graph(self.tf_chpt_path + ".meta")
            saver.restore(sess, self.tf_chpt_path)
            your_outputs = self.node_outputs_name.split(',')
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                output_node_names=your_outputs)
            with gfile.FastGFile(self.tf_pb_path, 'wb') as f:
                f.write(frozen_graph.SerializeToString())
            print("Frozen model is successfully stored!")

    def trt_convert(self):
        model_graph_def = self.create_graph()[0]
        self.node_outputs_name = self.node_outputs_name.split(',')
        trt_graph = trt.create_inference_graph(
            input_graph_def=model_graph_def,
            outputs=self.node_outputs_name,
            max_batch_size=self.max_batch_size,
            is_dynamic_op=False,
            max_workspace_size_bytes=int(self.max_workspace_size_bytes),
            minimum_segment_size=self.minimum_segment_size,
            precision_mode=self.precision_mode)

        with gfile.FastGFile(self.trt_pb_path, 'wb') as f:
            f.write(trt_graph.SerializeToString())
        print("TensorRT model is successfully stored!")

        all_nodes = len([1 for _ in model_graph_def.node])
        print("[INFO] Number of all_nodes in frozen graph:", all_nodes)

        trt_engine_nodes = len(
            [1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
        print("[INFO] Number of trt_engine_nodes in TensorRT graph:",
              trt_engine_nodes)
        all_nodes = len([1 for _ in trt_graph.node])
        print("[INFO] Number of all_nodes in TensorRT graph:", all_nodes)
