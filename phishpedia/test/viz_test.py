import unittest
from unittest.mock import MagicMock, patch
from src.detectron2_pedia.detectron2_1.viz import viz_data, viz_preds

class TestViz(unittest.TestCase):
    @patch('viz.build_detection_train_loader')
    @patch('viz.MetadataCatalog.get')
    @patch('viz.Visualizer')
    @patch('viz.wandb.Image')
    def test_viz_data(self, mock_wandb_image, mock_visualizer, mock_metadata_get, mock_build_detection_train_loader):
        cfg = MagicMock()
        mock_build_detection_train_loader.return_value = iter([MagicMock() for _ in range(10)])
        mock_metadata_get.return_value = MagicMock()
        mock_visualizer.return_value = MagicMock()
        mock_wandb_image.return_value = MagicMock()

        result = viz_data(cfg)

        self.assertEqual(len(result), 8)
        mock_build_detection_train_loader.assert_called_once_with(cfg, mapper=MagicMock())
        mock_metadata_get.assert_called_once_with(cfg.DATASETS.TRAIN[0])
        self.assertEqual(mock_visualizer.call_count, 8)
        self.assertEqual(mock_wandb_image.call_count, 8)

    @patch('viz.Path')
    @patch('viz.PathManager.open')
    @patch('viz.json.load')
    @patch('viz.DatasetCatalog.get')
    @patch('viz.MetadataCatalog.get')
    @patch('viz.cv2.imread')
    @patch('viz.Visualizer')
    @patch('viz.wandb.Image')
    def test_viz_preds(self, mock_wandb_image, mock_visualizer, mock_cv2_imread, mock_metadata_get, mock_dataset_get, mock_json_load, mock_path_manager_open, mock_path):
        cfg = MagicMock()
        mock_path.return_value = MagicMock()
        mock_path_manager_open.return_value.__enter__.return_value = MagicMock()
        mock_json_load.return_value = [MagicMock() for _ in range(10)]
        mock_dataset_get.return_value = [MagicMock() for _ in range(10)]
        mock_metadata_get.return_value = MagicMock()
        mock_cv2_imread.return_value = MagicMock()
        mock_visualizer.return_value = MagicMock()
        mock_wandb_image.return_value = MagicMock()

        result = viz_preds(cfg)

        self.assertEqual(len(result), 8)
        mock_path.assert_called_once_with(cfg.OUTPUT_DIR)
        mock_path_manager_open.assert_called_once()
        mock_json_load.assert_called_once()
        mock_dataset_get.assert_called_once_with(cfg.DATASETS.TEST[0])
        mock_metadata_get.assert_called_once_with(cfg.DATASETS.TEST[0])
        self.assertEqual(mock_cv2_imread.call_count, 8)
        self.assertEqual(mock_visualizer.call_count, 16)
        self.assertEqual(mock_wandb_image.call_count, 8)

if __name__ == '__main__':
    unittest.main()