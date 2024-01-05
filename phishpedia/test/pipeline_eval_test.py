import os
import unittest
from unittest.mock import MagicMock, patch, open
from src.pipeline_eval import phishpedia_eval
from tqdm import tqdm
import time
from phishpedia.src.siamese import *
from phishpedia.src.detectron2_pedia.inference import *
import argparse
import errno

class TestPipelineEval(unittest.TestCase):
    @patch('pipeline_eval.os')
    @patch('pipeline_eval.tqdm')
    @patch('pipeline_eval.time')
    @patch('pipeline_eval.pred_rcnn')
    @patch('pipeline_eval.phishpedia_classifier_logo')
    @patch('pipeline_eval.brand_converter')
    @patch('builtins.open', new_callable=open)
    def test_phishpedia_eval(self, open, brand_converter, phishpedia_classifier_logo, pred_rcnn):
        data_dir = '/path/to/data'
        mode = 'phish'
        siamese_ts = 0.5
        write_txt = '/path/to/write.txt'

        os.listdir.return_value = ['folder1', 'folder2']
        os.path.join.return_value = '/path/to/file'
        tqdm.return_value = iter(['folder1', 'folder2'])
        time.time.return_value = 0
        pred_rcnn.return_value = (MagicMock(), None, None, None)
        phishpedia_classifier_logo.return_value = (None, None, None)
        brand_converter.return_value = 'brand'

        phishpedia_eval(data_dir, mode, siamese_ts, write_txt)

        self.assertEqual(os.listdir.call_count, 1)
        self.assertEqual(os.path.join.call_count, 14)
        self.assertEqual(tqdm.call_count, 1)
        self.assertEqual(time.time.call_count, 2)
        self.assertEqual(pred_rcnn.call_count, 2)
        self.assertEqual(phishpedia_classifier_logo.call_count, 2)
        self.assertEqual(brand_converter.call_count, 4)
        self.assertEqual(open.call_count, 6)

if __name__ == '__main__':
    unittest.main()