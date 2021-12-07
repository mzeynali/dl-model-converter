from inference_converter.inference import Convertor
from inference_converter.utils import parse_options


if __name__ == "__main__":
    params = parse_options()

    ##### keras to Tensorflow Checkpoint
    if params['convert_mode'] == 'keras2chpt':
        convert = Convertor(
            h5_path_in=params['h5_path_in'],
            h5_path_out=params['h5_path_out'],
            tf_chpt_path=params['tf_chpt_path'])
        convert.chpt_convert()

    ##### Tensorflow Checkpoint to Tensorflow Frozen Graph
    elif params['convert_mode'] == 'chpt2pb':
        convert = Convertor(
            tf_pb_path=params['tf_pb_path'],
            tf_chpt_path=params['tf_chpt_path'],
            node_outputs_name=params['node_outputs_name'])
        convert.pb_convert()

    ##### Tensorflow Frozen Graph to TensorRT-Tensorflow Engines
    elif params['convert_mode'] == 'pb2trt':
        convert = Convertor(
            tf_pb_path=params['tf_pb_path'],
            trt_pb_path=params['trt_pb_path'],
            node_outputs_name=params['node_outputs_name'],
            max_workspace_size_bytes=params['max_workspace_size_bytes'],
            minimum_segment_size=params['minimum_segment_size'],
            precision_mode=params['precision_mode'])
        convert.trt_convert()
