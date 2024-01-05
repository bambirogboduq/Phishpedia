import unittest
from unittest.mock import MagicMock, patch
from src.detectron2_pedia.detectron2_1.register_backbone import register_resnet_fpn_backbone

class TestRegisterBackbone(unittest.TestCase):
    @patch('register_backbone.Backbone')
    @patch('register_backbone.build_resnet_backbone')
    @patch('register_backbone.LastLevelMaxPool')
    def test_register_resnet_fpn_backbone(self, mock_last_level_max_pool, mock_build_resnet_backbone, mock_backbone):
        cfg = MagicMock()
        input_shape = MagicMock()

        # Mock the objects returned by the functions we're not testing
        mock_build_resnet_backbone.return_value = MagicMock()
        mock_last_level_max_pool.return_value = MagicMock()

        # Call the function we're testing
        result = register_resnet_fpn_backbone(cfg, input_shape)

        # Check that the functions we're not testing were called with the right arguments
        mock_build_resnet_backbone.assert_called_once_with(cfg, input_shape)
        mock_last_level_max_pool.assert_called_once()

        # Check that the function we're testing returned the right result
        self.assertEqual(result, mock_backbone.return_value)

if __name__ == '__main__':
    unittest.main()