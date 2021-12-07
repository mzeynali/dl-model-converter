import unittest
import sys

sys.path.append('../')
from inference_converter.inference import Convertor


class TestConvertor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.convert = Convertor(
            h5_path_in='../inference_converter/weights/h5/ssd_mobile_v1_300.h5',
            h5_path_out='../inference_converter/weights/h5/model.h5',
            tf_chpt_path='../inference_converter/weights/ckpt/model',
            tf_pb_path='../inference_converter/weights/pb/model.pb',
            trt_pb_path="../inference_converter/weights/trt/model_trt.pb",
            node_outputs_name="predictions_1/concat",
            max_batch_size=12,
            max_workspace_size_bytes=1e9,
            minimum_segment_size=3,
            precision_mode="FP16")

    @classmethod
    def tearDownClass(cls):
        pass

    def test_chpt_convert(self):
        self.convert.chpt_convert()

    def test_pb_convert(self):
        self.convert.pb_convert()

    def test_trt_convert(self):
        self.convert.trt_convert()


if __name__ == "__main__":
    unittest.main()
