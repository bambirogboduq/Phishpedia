import unittest
import torch
from unittest.mock import MagicMock, patch
from src.detectron2_pedia.detectron2_1.adv import DAGAttacker

class TestDAGAttacker(unittest.TestCase):
    def setUp(self):
        self.dag_attacker = DAGAttacker()

    def test_attack_image(self):
        self.dag_attacker.model = MagicMock()
        self.dag_attacker.model.preprocess_image.return_value = torch.rand(1, 3, 224, 224)
        batched_inputs = [{'instances': MagicMock()}]
        result = self.dag_attacker.attack_image(batched_inputs)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 3, 224, 224))

    def test_create_instance_dicts(self):
        outputs = [{'instances': MagicMock()}]
        image_id = 0
        result = self.dag_attacker._create_instance_dicts(outputs, image_id)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], dict)
        self.assertIn('image_id', result[0])
        self.assertIn('category_id', result[0])
        self.assertIn('bbox', result[0])
        self.assertIn('score', result[0])

    def test_post_process_image(self):
        image = torch.rand(1, 3, 224, 224)
        result = self.dag_attacker._post_process_image(image)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, image.shape)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 255))

    def test_get_adv_labels(self):
        labels = torch.randint(0, 10, (5,))
        result = self.dag_attacker._get_adv_labels(labels)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, labels.shape)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result < 10))

    @patch('adv.Boxes')
    def test_get_targets(self, mock_boxes):
        batched_inputs = [{'instances': MagicMock()}]
        result = self.dag_attacker._get_targets(batched_inputs)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], mock_boxes)
        self.assertIsInstance(result[1], torch.Tensor)

    # Add more tests for other methods here

if __name__ == '__main__':
    unittest.main()