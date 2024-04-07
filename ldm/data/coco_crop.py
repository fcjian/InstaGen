import random
import json
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.transforms import RandomGTCrop, PackDetInputs
from mmengine.registry import TRANSFORMS

# register module
TRANSFORMS.register_module(module=RandomGTCrop)
TRANSFORMS.register_module(module=PackDetInputs)

class CocoDatasetGTCrop(CocoDataset):
    def __init__(self, ann_file='data/coco/annotations/instances_train2017_seen.json', img_prefix='data/coco/train2017', \
                 crop_size=512, random_gt=3, num_imgs=-1, name_mapping_file=None, return_gt=False):
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomGTCrop', crop_size=(crop_size, crop_size), random_gt=random_gt),
            dict(type='Resize',
                 scale=(crop_size, crop_size),
                 keep_ratio=True
            ),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Normalize', mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True),
            dict(type='Pad', size_divisor=crop_size),
            dict(type='PackDetInputs')
        ]
        super(CocoDatasetGTCrop, self).__init__(ann_file=ann_file, pipeline=pipeline, data_prefix=dict(img=img_prefix))

        # for quick test
        if num_imgs > 0:
            self.data_address = self.data_address[:num_imgs]

        self.name_mapping = None
        if name_mapping_file is not None:
            with open(name_mapping_file, 'r') as f:
                self.name_mapping = json.load(f)
        print(f"name mapping:\n {self.name_mapping}")

        self.return_gt = return_gt

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

                Args:
                    idx (int): Index of data.

                Returns:
                    dict: Training/test data (with annotation if `test_mode` is set \
                        True).
        """

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another()
                continue

            sd_data = {}
            sd_data['image'] = data['inputs'].data.permute(1, 2, 0)

            cat_names = []
            gt_labels = data['data_samples'].gt_instances.labels
            for cat_id in gt_labels:
                cat_name = self.metainfo['classes'][cat_id].replace('_', ' ')
                if self.name_mapping is not None:
                    cat_name = self.name_mapping[cat_name]
                cat_names.append(cat_name)

            cat_names = list(set(cat_names))
            random.shuffle(cat_names)
            sd_data['txt'] = 'a photograph of ' + ' and '.join(cat_names)

            if self.return_gt:
                sd_data['gt_labels'] = data['data_samples'].gt_instances.labels
                sd_data['gt_bboxes'] = data['data_samples'].gt_instances.bboxes

            return sd_data
