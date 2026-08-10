import numpy as np
from mmdet3d.datasets import KittiDataset
from mmdet3d.registry import DATASETS
from mmengine.dataset import Compose

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData

@DATASETS.register_module()
class KittiFusionDataset(KittiDataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline,
                 modality=dict(use_lidar=True, use_camera=True),
                 test_mode=False,
                 **kwargs):
        self.modality = modality
        print(f"✅ Creating KittiFusionDataset with modalities={self.modality}")
        super().__init__(data_root=data_root,
                         ann_file=ann_file,
                         pipeline=pipeline,
                         modality=modality,
                         test_mode=test_mode,                         
                         **kwargs)
        #self.pipeline = Compose(pipeline)
        print(f"✅ After init KittiFusionDataset with modalities={self.modality} user modality passed={modality}")
        print(f"✅ Using KittiFusionDataset with use_camera={self.modality.get('use_camera', 'N/A')}")
        print(f"✅ Using KittiFusionDataset with use_lidar={self.modality.get('use_lidar', 'N/A')}")
        #self.modality = modality
            

    def get_data_info(self, index):
        info = super().get_data_info(index)
        # print(f"✅fusion data set data_info {info.keys()}")
        # Ensure 'sample_idx' is included
        if 'sample_idx' not in info:
            info['sample_idx'] = index  # Use the index as a fallback if 'sample_idx' is not provided

        # Ensure 'img_path' is included
        if 'img_path' not in info and 'images' in info:
            info['img_path'] = info['images']['CAM2']['img_path']

        # Ensure 'lidar_path' is included
        if 'lidar_path' not in info:
            info['lidar_path'] = info['lidar_points']['lidar_path']

        # Handle training and testing modes
        if not self.test_mode:
            if 'ann_info' not in info:
                raise KeyError("Missing 'ann_info' key in info for training mode.")
        else:
            if 'eval_ann_info' in info:
                # print(f"✅fusion data set data_info {info['eval_ann_info'].keys()}")
                info['ann_info'] = info['eval_ann_info']  # Use eval_ann_info for validation/testing
        
        # # Populate 2D annotations if available
        # if 'instances' in info:
        #     gt_bboxes = []
        #     gt_labels = []
        #     for instance in info['instances']:
        #         if 'bbox' in instance and 'label' in instance:
        #             gt_bboxes.append(instance['bbox'])
        #             gt_labels.append(instance['label'])
        #     info['ann_info']['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32)
        #     info['ann_info']['gt_labels'] = np.array(gt_labels, dtype=np.int64)

        return info
    
    
    def show(self, results, out_dir, show=False, score_thr=0.0):
        """Visualize the results of the dataset.

        Args:
            results (list[dict]): The detection results.
            out_dir (str): The directory to save the visualization results.
            show (bool): Whether to display the results. Defaults to False.
            score_thr (float): The score threshold to filter results. Defaults to 0.0.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] Output directory: {out_dir}")

        visualizer = Det3DLocalVisualizer(save_dir=out_dir)
        for i, result in enumerate(results):
            data_info = self.get_data_info(i)
            # Prepare the data sample for visualization

            # Create a Det3DDataSample object
            data_sample = Det3DDataSample()

            # print(f"[DEBUG] data_info keys: {data_info.keys()}")
            # print(f"[DEBUG] data_info anno info keys: {data_info['ann_info'].keys()}")
            # print(f"[DEBUG] data_info eval anno info keys: {data_info['eval_ann_info'].keys()}")

            # Populate ground truth fields
            if 'ann_info' in data_info:
                gt_instances_3d = InstanceData()
                gt_instances_3d.bboxes_3d = data_info['ann_info'].get('gt_bboxes_3d', None)
                gt_instances_3d.labels_3d = data_info['ann_info'].get('gt_labels_3d', None)
                data_sample.gt_instances_3d = gt_instances_3d

                gt_instances = InstanceData()
                gt_instances.bboxes = data_info['ann_info'].get('gt_bboxes', None)
                gt_instances.labels = data_info['ann_info'].get('gt_labels', None)
                data_sample.gt_instances = gt_instances

            print(f"[DEBUG] result: {result}")
            # Create an InstanceData object for predictions
            pred_instances_3d = InstanceData()
            
            pred_instances_3d.bboxes_3d = result.get('dimensions', None)
            pred_instances_3d.scores_3d = result.get('score', None)
            pred_instances_3d.labels_3d = result.get('name', None)

            data_sample.pred_instances_3d = pred_instances_3d
            data_sample.set_metainfo(data_info)  # Copy metadata

            # Debugging
            # print(f"[DEBUG] gt_instances_3d: {data_sample.gt_instances_3d}")
            # print(f"[DEBUG] pred_instances_3d: {data_sample.pred_instances_3d}")
            # print(f"[DEBUG] metainfo: {data_sample.metainfo}")

            # Add the sample to the visualizer
            visualizer.add_datasample(
                name=f'sample_{i}',
                data_input=data_info,
                data_sample=data_sample,
                show=show,
                pred_score_thr=score_thr,
                out_file=f'{out_dir}/sample_{i}',
                o3d_save_path=out_dir,
            )
            # Explicitly save the visualization
        print(f"✅ Visualization completed. Results saved to: {out_dir}")

'''
    def get_data_info(self, index):
        print(f"🚀 [DEBUG] get_data_info() from: {self.__class__}")

        """Get image and point cloud paths + related info."""
        info = self.data_list[index]
        input_dict = dict()

        if self.modality.get('use_lidar', True):
            input_dict['pts_filename'] = info['lidar_points']['lidar_path']

        if self.modality.get('use_camera', True):
            cam_infos = info.get('images', {})
            input_dict['img_filename'] = [cam['img_path'] for cam in cam_infos.values()]
            input_dict['cam2img'] = [cam['cam2img'] for cam in cam_infos.values()]

        input_dict['ann_info'] = info.get('annos', None)
        
        if 'img' not in info:
            print(f'[Warning] Image not loaded at index {index}')
        else:
            print(f'[OK] Loaded image at index {index}')
            
        return input_dict

    def __getitem__(self, idx):
        """Returns processed data dict containing both lidar and camera."""
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_data(self, index):
        """Apply pipeline to training data."""
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        """Apply pipeline to testing data."""
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example

'''
    