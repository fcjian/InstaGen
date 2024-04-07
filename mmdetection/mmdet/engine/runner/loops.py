# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.model import is_model_wrapper
from mmengine.runner import ValLoop, EpochBasedTrainLoop

from mmdet.registry import LOOPS
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.runner.utils import calc_dynamic_intervals
import logging
import torch
from mmengine.structures import InstanceData
import math
from mmdet.structures.bbox import HorizontalBoxes
import os
import cv2
import matplotlib.pyplot as plt


@LOOPS.register_module()
class TeacherStudentValLoop(ValLoop):
    """Loop for validation of model teacher and student."""

    def run(self):
        """Launch validation for model teacher and student."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        model = self.runner.model
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'teacher')
        assert hasattr(model, 'student')

        predict_on = model.semi_test_cfg.get('predict_on', None)
        multi_metrics = dict()
        for _predict_on in ['teacher', 'student']:
            model.semi_test_cfg['predict_on'] = _predict_on
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            # compute metrics
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            multi_metrics.update(
                {'/'.join((_predict_on, k)): v
                 for k, v in metrics.items()})
        model.semi_test_cfg['predict_on'] = predict_on

        self.runner.call_hook('after_val_epoch', metrics=multi_metrics)
        self.runner.call_hook('after_val')


@LOOPS.register_module()
class EpochDatasetTrainLoop(EpochBasedTrainLoop):
    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
            train_dataloaders=[], 
            self_train=False,
            score_thresh=0.4,
            start_epoch=6,
            momentum=0.999,
            filter_empty_novel_gt=True,
            dataset_percs=None) -> None:
        self._runner = runner
        self.num_dataloders = 0
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloaders = []
            if len(train_dataloaders) > 0:
                for train_dataloader in train_dataloaders:
                    cur_dataloader = runner.build_dataloader(
                        train_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
                    self.dataloaders.append(cur_dataloader)
                    self.num_dataloders += 1
                self.dataloader = self.dataloaders[0]
            else: 
                self.dataloader = runner.build_dataloader(
                    dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader = dataloader

        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        for i in range(1, self.num_dataloders):
            self._max_iters = self._max_epochs * len(self.dataloaders[i])
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)
        
        self.self_train = self_train
        self.updated_teacher = False
        self.score_thresh = score_thresh
        self.start_epoch = start_epoch
        self.dataset_percs = dataset_percs
        self.momentum = momentum
        self.filter_empty_novel_gt = filter_empty_novel_gt
        print(f"############## self-train: {self.self_train} ################")
        print(f"############## score thresh: {score_thresh} ###############")
        print(f"############## start epoch: {start_epoch} ###############")
        print(f"############## momentum: {momentum} ###############")
        print(f"############## filter empty novel gt: {filter_empty_novel_gt} ###############")
        print(f"############## dataset percentages: {dataset_percs} ###############")

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        self.count_iters = [len(self.dataloaders[i]) for i in range(len(self.dataloaders))]
        self.loader_iters = None

        while self._epoch < self._max_epochs and not self.stop_training:
            if self.self_train and self._epoch >= self.start_epoch:
                if not self.updated_teacher:
                    print(f"########### copy weight from student to teacher at iter {self._iter} ###########")
                    self.runner.teacher_model.cuda()
                    self.momentum_update(self.runner.model, self.runner.teacher_model, 0)
                    self.updated_teacher = True

            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')

        self.runner.model.train()
        if self.num_dataloders > 1:
            if self._epoch >= self.start_epoch:
                print(f"################# using {self.num_dataloders} datasets! #################")

                num_iters = []
                loader_iters = []
                self._max_iters = 0
                for i, data_loader in enumerate(self.dataloaders): 
                    try:
                        print(f"#cat2label of {i}-th dataset: {len(data_loader.dataset.cat2label)}")
                    except:
                        print(f"#cat2label of {i}-th dataset: {len(data_loader.dataset.dataset.cat2label)}")

                    if self.dataset_percs is not None:
                        assert len(self.dataloaders) == len(self.dataset_percs)
                        num_iter = int(len(data_loader) * self.dataset_percs[i])
                    else:
                        num_iter = len(data_loader)
                    self._max_iters += self._max_epochs * num_iter
                    num_iters.append(num_iter)

                    if self.count_iters[i] >= len(data_loader):
                        print(f"################# epoch start: init {i}-th data iter #################")
                        loader_iters.append(iter(data_loader))
                        self.count_iters[i] = 0
                    else:
                        print(f"################# epoch start: continue {i}-th data iter #################")
                        loader_iters.append(self.loader_iters[i])
                self.loader_iters = loader_iters

                print(f"############### iter number of data loaders: {num_iters} ###############")
                sum_iters = sum(num_iters)
                iter_percs = [0. for _ in range(len(self.dataloaders))]
                for idx in range(sum_iters):
                    index = iter_percs.index(min(iter_percs))

                    if self.count_iters[index] >= len(self.dataloaders[index]):
                        print(f"################# epoch runing: init {index}-th data iter #################")
                        loader_iters[index] = iter(self.dataloaders[index])
                        self.count_iters[index] = 0
                    self.count_iters[index] += 1

                    iter_percs[index] += 1.0 / num_iters[index]
                    data_batch = next(loader_iters[index])
                    self.run_iter(idx, data_batch)
            else:
                loader_iters = []
                for idx, data_batch in enumerate(self.dataloaders[0]):
                    self.run_iter(idx, data_batch)
        else:
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        
        if self.self_train and self._epoch >= self.start_epoch:
            self.momentum_update(self.runner.model, self.runner.teacher_model, momentum=self.momentum)

            inputs = data_batch['inputs']
            data_samples = data_batch['data_samples']
            novel_image_flag = False
            for i in range(len(data_samples)):
                novel_image_flag = novel_image_flag | data_samples[i].novel_image_flag[0]
            if novel_image_flag:
                valid_image_flags = self.generate_psedo_label(self.runner.teacher_model, data_batch, data_samples, score_thresh=self.score_thresh)

                num_images = len(valid_image_flags)
                num_valid_images = num_images
                if self.filter_empty_novel_gt:
                    num_valid_images = sum(valid_image_flags)
                    if num_valid_images == 0:
                        valid_inputs = inputs[:1]
                        valid_data_samples = data_samples[:1]
                    else:
                        valid_inputs = [inputs[i] for i in range(len(inputs)) if valid_image_flags[i]]
                        valid_data_samples = [data_samples[i] for i in range(len(data_samples)) if valid_image_flags[i]]

                    data_batch['inputs'] = valid_inputs
                    data_batch['data_samples'] = valid_data_samples

                outputs = self.runner.model.train_step(
                    data_batch, optim_wrapper=self.runner.optim_wrapper)
                for key in outputs.keys():
                    outputs[key] = outputs[key] * math.sqrt(float(num_valid_images) / num_images)
            else:
                outputs = self.runner.model.train_step(
                    data_batch, optim_wrapper=self.runner.optim_wrapper)
        else:
            # Enable gradient accumulation mode and avoid unnecessary gradient
            # synchronization during gradient accumulation process.
            # outputs should be a dict of loss.
            outputs = self.runner.model.train_step(
                data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
        
    def momentum_update(self, student, teacher, momentum):  # student = self.runner.model; teacher = self.runner.teacher_model
        for (src_name, src_parm), (tgt_name, tgt_parm) in zip(
            student.named_parameters(), teacher.named_parameters()
        ):
            tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

    def generate_psedo_label(self, model, data_batch, data_samples, score_thresh=0.4):
        model.eval()
        with torch.no_grad(): 
            results = model.val_step(data_batch)

            device = data_samples[0].gt_instances.labels.device
            valid_image_flags = []
            for i in range(len(results)):
                if data_samples[i].novel_image_flag[0]:
                    bboxes = results[i].pred_instances.bboxes
                    scores = results[i].pred_instances.scores
                    labels = results[i].pred_instances.labels
                    pseudo_labels = labels[scores > score_thresh]
                    pseudo_bboxes = bboxes[scores > score_thresh]

                    label_mappings = data_samples[i].label_mappings
                    novel_cls_inds = data_samples[i].novel_cls_inds

                    flags = pseudo_labels > 100000
                    for grounding_label, dataset_label in label_mappings.items():
                        if dataset_label in novel_cls_inds:
                            flags = flags | (pseudo_labels == dataset_label)
                            pseudo_labels[pseudo_labels == dataset_label] = grounding_label
                    pseudo_labels = pseudo_labels[flags]
                    pseudo_bboxes = pseudo_bboxes[flags]

                    if len(pseudo_labels) > 0:
                        gt_instances = InstanceData(
                            labels = torch.cat([data_samples[i].gt_instances.labels, pseudo_labels.to(device)]), 
                            bboxes = HorizontalBoxes(torch.cat([data_samples[i].gt_instances.bboxes.tensor, pseudo_bboxes.to(device)]))
                        )
                        data_samples[i].gt_instances = gt_instances
                        valid_image_flags.append(True)
                    else:
                        valid_image_flags.append(False)
                else:
                    valid_image_flags.append(True)

        return valid_image_flags
    
