# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os.path as osp
import time
import types
import warnings
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Union

from mmengine.dist.utils import is_main_process
from mmengine.fileio import FileClient
from mmengine.logging import print_log
from mmengine.model import BaseModel, is_model_wrapper, revert_sync_batchnorm

try:
    import deepspeed
    from deepspeed.pipe import PipelineModule  # noqa: F401
except ImportError:
    deepspeed = None
    PipelineModule = None

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.device import get_device
from mmengine.evaluator import Evaluator
from mmengine.hooks import Hook
from mmengine.optim import (DeepSpeedOptimWrapper, OptimWrapperDict,
                            _ParamScheduler)
from mmengine.registry import RUNNERS, DefaultScope, count_registered_modules
from mmengine.utils import get_git_hash
from mmengine.visualization import Visualizer
from .checkpoint import _load_checkpoint
from .checkpoint import get_state_dict as _get_state_dict
from .checkpoint import save_checkpoint, weights_to_cpu
from .runner import Runner

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[DeepSpeedOptimWrapper, OptimWrapperDict]


@RUNNERS.register_module()
class DeepSpeedRunner(Runner):
    """A training helper for ``DeepSpeedEngine``.

    Main logic to initialize DeepSpeed:
        >>> self.model = self.build_model(model)
        >>> self.optim_wrapper = self.build_optim_wrapper(optim_wrapper)
        >>> self.model, optimizer = deepspeed.initialize(
        >>>    model=self.model,
        >>>    optimizer=self.optim_wrapper.optimizer,
        >>>    model_parameters=self.model.parameters(),
        >>>    config=ds_config)
        >>> self.optim_wrapper.optimizer = optimizer
        >>> self.inject_base_model_methods()

    Note:
        Optimizing steps are done in ``DeepSpeedEngine`` and
        ``PipelineEngine``.
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict],
        work_dir: str,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        train_cfg: Optional[Dict] = None,
        val_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        auto_scale_lr: Optional[Dict] = None,
        optim_wrapper: Optional[Union[DeepSpeedOptimWrapper, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        data_preprocessor: Union[nn.Module, Dict, None] = None,
        load_from: Optional[str] = None,
        resume: bool = False,
        launcher: str = 'none',
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_processor: Optional[Dict] = None,
        log_level: str = 'INFO',
        visualizer: Optional[Union[Visualizer, Dict]] = None,
        default_scope: str = 'mmengine',
        randomness: Dict = dict(seed=None),
        experiment_name: Optional[str] = None,
        cfg: Optional[ConfigType] = None,
    ):
        assert deepspeed is not None, 'DeepSpeed should be installed!'
        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optimizer should be either '
                'all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        self.optim_wrapper: Optional[Union[DeepSpeedOptimWrapper, dict]]
        self.optim_wrapper = optim_wrapper

        self.auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optimizer is None, '
                f'but got {param_scheduler}')

        # Parse `param_scheduler` to a list or a dict. If `optim_wrapper` is a
        # `dict` with single optimizer, parsed param_scheduler will be a
        # list of parameter schedulers. If `optim_wrapper` is
        # a `dict` with multiple optimizers, parsed `param_scheduler` will be
        # dict with multiple list of parameter schedulers.
        self._check_scheduler_cfg(param_scheduler)
        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        # self._timestamp will be set in the `setup_env` method. Besides,
        # it also will initialize multi-process and (or) distributed
        # environment.
        self.setup_env(env_cfg)
        # self._deterministic and self._seed will be set in the
        # `set_randomness`` method
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        self.default_scope = DefaultScope.get_instance(
            self._experiment_name, scope_name=default_scope)
        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)
        # Since `get_instance` could return any subclass of ManagerMixin. The
        # corresponding attribute needs a type hint.
        self.logger = self.build_logger(log_level=log_level)

        # Collect and log environment information.
        self._log_env(env_cfg)

        # collect information of all modules registered in the registries
        registries_info = count_registered_modules(
            self.work_dir if self.rank == 0 else None, verbose=False)
        self.logger.debug(registries_info)

        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(visualizer)

        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        # load DeepSpeed configuration file
        self.ds_config = self.cfg.get('ds_config', None)
        assert self.ds_config is not None, 'ds_config should be specified.'

        self.check_ds_config(self.ds_config)

        # build a model
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            model.setdefault('data_preprocessor', data_preprocessor)
        self.model = self.build_model(model)

        # initialize the model weights before wrapping it with deepspeed
        self._weights_initialized = False
        self._init_model_weights()

        # wrap model
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)

        # build optim wrapper
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        assert isinstance(self.optim_wrapper, DeepSpeedOptimWrapper), (
            'OptimWrapper type should be \'DeepSpeedOptimWrapper\' when using '
            'DeepSpeedRunner.')

        # initialize DeepSpeed Engine
        self.model, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optim_wrapper.optimizer,
            model_parameters=self.model.parameters(),
            config=self.ds_config)
        self.optim_wrapper.optimizer = optimizer

        # Set logging level to remove duplicate training log from DeepSpeed
        deepspeed_logger = logging.getLogger('DeepSpeed')
        deepspeed_logger.setLevel(logging.WARNING)

        self.inject_basemodel_methods()

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)

        # dump `cfg` to `work_dir`
        self.dump_config()

    def wrap_model(self, model_wrapper_cfg: Optional[Dict],
                   model: BaseModel) -> Union[BaseModel, PipelineModule]:
        """Wrap the model to :obj:``PipelineModule`` or other custom
        distributed data-parallel module wrappers.

        An example of ``model_wrapper_cfg``::

            model_wrapper_cfg = dict(
                type='PipelineModule',
            )

        Args:
            model_wrapper_cfg (dict, optional): Config to wrap model. If not
                specified, ``BaseModel`` will be used. Defaults to None.
            model (BaseModel): Model to be wrapped.

        Returns:
            BaseModel or PipelineModule: BaseModel or ``PipelineModule``.
        """
        if is_model_wrapper(model):
            if model_wrapper_cfg is not None:
                raise TypeError(
                    'model has been wrapped and "model_wrapper_cfg" should be '
                    f'None, but got {model_wrapper_cfg}')

            return model

        # Set `export CUDA_VISIBLE_DEVICES=-1` to enable CPU training.
        model = model.to(get_device())

        if not self.distributed:
            self.logger.info(
                'Distributed training is not used, all SyncBatchNorm (SyncBN) '
                'layers in the model will be automatically reverted to '
                'BatchNormXd layers if they are used.')
            model = revert_sync_batchnorm(model)
            return model

        if model_wrapper_cfg is None:
            # Model will be wrapped in `deepspeed.initialize`.
            pass
        elif model_wrapper_cfg.get('type') == 'PipelineModule':
            if 'zero_optimization' in self.ds_config and self.ds_config[
                    'zero_optimization']['stage'] > 1:
                raise RuntimeError(
                    'Pipeline Parallel is compatible with only ZeRO stage 1')

            # TODO: Model Sequentializing
            # sequential_model = convert_to_sequential_model(model)
            # model = PipelineModule(
            #     layers=[model], num_stages=int(os.environ['WORLD_SIZE']))
            raise NotImplementedError(
                'Pipeline Parallel is not implemented yet.')
        else:
            raise NotImplementedError(
                'DeepSpeed only supports ``PipelineModule`` for '
                '"model_wrapper" yet.')
        return model

    def _init_model_weights(self) -> None:
        """Initialize the model weights when it is not initialized."""
        if not self._weights_initialized:
            super()._init_model_weights()
            self._weights_initialized = True

    def check_ds_config(self, ds_config):
        """Check DeepSpeed configuration to prevent duplicated settings."""

        # batch size
        if ds_config.keys() & {
                'train_batch_size', 'train_micro_batch_size_per_gpu',
                'gradient_accumulation_steps'
        }:
            self.logger.warning(
                'Batch size in DeepSpeed configuration will be ignored')

        # optimizer
        if 'optimizer' in ds_config.keys():
            self.logger.warning(
                'Optimizer in DeepSpeed configuration will be ignored')

        # LR scheduler
        if 'scheduler' in ds_config.keys():
            self.logger.warning(
                'LR scheduler in DeepSpeed configuration will be ignored')

    def inject_basemodel_methods(self):
        """inject methods from ``BaseModel`` into ``DeepSpeedEngine`` to make
        ``DeepSpeedEngine`` support the ``train_step`` method appropriately.

        Without injecting, ``DeepSpeedOptimWrapper`` tries ``backward`` from
        ``BaseModel``, which should be in ``DeepSpeedEngine``.
        """

        def _train_step(self, data: Union[dict, tuple, list],
                        optim_wrapper) -> Dict[str, torch.Tensor]:
            with optim_wrapper.optim_context(self):
                data = self.data_preprocessor(data, True)
                losses = self._run_forward(data, mode='loss')  # type: ignore
            parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
            optim_wrapper.update_params(parsed_losses)
            return log_vars

        self.model.train_step = types.MethodType(_train_step, self.model)

    def resume(
        self,
        filename: str,
        resume_optimizer: bool = True,
        resume_param_scheduler: bool = True,
        map_location: Union[str, Callable] = 'default',
    ) -> None:
        """Resume model from checkpoint.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            resume_optimizer (bool): Whether to resume optimizer state.
                Defaults to True.
            resume_param_scheduler (bool): Whether to resume param scheduler
                state. Defaults to True.
            map_location (str or callable):A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'default'.
        """

        if self.model.zero_optimization_partition_weights():
            device = get_device()
            checkpoint = _load_checkpoint(filename, map_location=device)
        elif map_location == 'default':
            device = get_device()
            # Note that do not remove the 'module.' prefix in keys
            checkpoint = self.load_checkpoint(
                filename, map_location=device, revise_keys=[])
        else:
            # Note that do not remove the 'module.' prefix in keys
            checkpoint = self.load_checkpoint(
                filename, map_location=map_location, revise_keys=[])

        self.train_loop._epoch = checkpoint['meta']['epoch']
        self.train_loop._iter = checkpoint['meta']['iter']

        # check whether the number of GPU used for current experiment
        # is consistent with resuming from checkpoint
        if 'config' in checkpoint['meta']:
            config = mmengine.Config.fromstring(
                checkpoint['meta']['config'], file_format='.py')
            previous_gpu_ids = config.get('gpu_ids', None)
            if (previous_gpu_ids is not None and len(previous_gpu_ids) > 0
                    and len(previous_gpu_ids) != self._world_size):
                if self.model.zero_optimization():
                    raise RuntimeError(
                        'Cannot resuming training with ZeRO. Automatic '
                        'adjustment of ZeRO\'s optimizer state partitioning '
                        'with a new world size is not currently supported.'
                        'Make sure the number of GPU is consistent with the '
                        'previous training state resuming from the checkpoint.'
                    )
                else:
                    self.logger.info(
                        'Number of GPU used for current experiment is not '
                        'consistent with resuming from checkpoint')
                    if self.auto_scale_lr is None or \
                            not self.auto_scale_lr.get('enable', False):
                        raise RuntimeError(
                            'Cannot automatically rescale lr in resuming. '
                            'Please make sure the number of GPU is consistent '
                            'with the previous training state resuming from '
                            'the checkpoint or set `enable` in `auto_scale_lr`'
                            ' to False.')

        # resume random seed
        resumed_seed = checkpoint['meta'].get('seed', None)
        current_seed = self._randomness_cfg.get('seed')
        if resumed_seed is not None and resumed_seed != current_seed:
            if current_seed is not None:
                warnings.warn(f'The value of random seed in the '
                              f'checkpoint "{resumed_seed}" is '
                              f'different from the value in '
                              f'`randomness` config "{current_seed}"')
            self._randomness_cfg.update(seed=resumed_seed)
            self.set_randomness(**self._randomness_cfg)

        resumed_dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        dataset_meta = getattr(self.train_dataloader.dataset, 'metainfo', None)
        if resumed_dataset_meta != dataset_meta:
            warnings.warn(
                'The dataset metainfo from the resumed checkpoint is '
                'different from the current training dataset, please '
                'check the correctness of the checkpoint or the training '
                'dataset.')

        self.message_hub.load_state_dict(checkpoint['message_hub'])

        # resume optimizer
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
            if self.model.zero_optimization():
                self.optim_wrapper.load_state_dict(  # type: ignore
                    checkpoint['optimizer'],
                    load_from_fp32_weights=self.model.
                    zero_load_from_fp32_weights())
            else:
                self.optim_wrapper.load_state_dict(  # type: ignore
                    checkpoint['optimizer'])

        # resume param scheduler
        if resume_param_scheduler and self.param_schedulers is None:
            print_log(
                '`resume_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip resuming parameter schedulers',
                logger='current',
                level=logging.WARNING,
            )
            resume_param_scheduler = False
        if 'param_schedulers' in checkpoint and resume_param_scheduler:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore
            if isinstance(self.param_schedulers, dict):
                for name, schedulers in self.param_schedulers.items():
                    for scheduler, ckpt_scheduler in zip(
                            schedulers, checkpoint['param_schedulers'][name]):
                        scheduler.load_state_dict(ckpt_scheduler)
            else:
                for scheduler, ckpt_scheduler in zip(
                        self.param_schedulers,  # type: ignore
                        checkpoint['param_schedulers'],
                ):
                    scheduler.load_state_dict(ckpt_scheduler)

        self._has_loaded = True

        self.logger.info(f'resumed epoch: {self.epoch}, iter: {self.iter}')

    def save_checkpoint(
        self,
        out_dir: str,
        filename: str,
        file_client_args: Optional[dict] = None,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        meta: dict = None,
        by_epoch: bool = True,
        backend_args: Optional[dict] = None,
    ):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.
        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. See :class:`mmengine.fileio.FileClient` for
                details. Defaults to None. It will be deprecated in future.
                Please use `backend_args` instead.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            by_epoch (bool): Whether the scheduled momentum is updated by
                epochs. Defaults to True.
            backend_args (dict, optional): Arguments to instantiate the
                preifx of uri corresponding backend. Defaults to None.
                New in v0.2.0.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if by_epoch:
            # self.epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self.epoch + 1`
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch, iter=self.iter + 1)

        file_client = FileClient.infer_client(file_client_args, out_dir)
        filepath = file_client.join_path(out_dir, filename)

        meta.update(
            cfg=self.cfg.pretty_text,
            seed=self.seed,
            experiment_name=self.experiment_name,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine_version=mmengine.__version__ + get_git_hash(),
            deepspeed_config=self.ds_config,
            deepspeed_version=deepspeed.git_version_info.version)

        if hasattr(self.train_dataloader.dataset, 'metainfo'):
            meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)

        # TODO: Support DeepSpeed-MoE
        if self.model.has_moe_layers:
            raise NotImplementedError('DeepSpeed-MoE is not supported yet.')

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        if self.model.zero_optimization_partition_weights():
            # Prepare for checkpoint save by
            # ensuring all parameters are partitioned
            self.model.optimizer.checkpoint_event_prologue()

        checkpoint = {
            'meta': meta,
            'message_hub': self.message_hub.state_dict(),
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if not self.model.zero_optimization():
                checkpoint['optimizer'] = self.optim_wrapper.state_dict()
            else:
                self.consolidate_state_dict(self.optim_wrapper.state_dict())
                # Only the main process needs to load the optimizer's state.
                optim_state = self.get_zero_state_dict()
                checkpoint['optimizer'] = optim_state

        # model state is stored after pulling optimizer state to handle ZeRO 3.
        checkpoint['state_dict'] = weights_to_cpu(self.get_state_dict(model))

        if self.model.zero_optimization_partition_weights():
            self.model.optimizer.checkpoint_event_epilogue()

        # save param scheduler state dict
        if save_param_scheduler and self.param_schedulers is None:
            print_log(
                '`save_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip saving parameter schedulers',
                logger='current',
                level=logging.WARNING)
            save_param_scheduler = False
        if save_param_scheduler:
            if isinstance(self.param_schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in self.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = []
                    for scheduler in schedulers:
                        state_dict = scheduler.state_dict()
                        checkpoint['param_schedulers'][name].append(state_dict)
            else:
                checkpoint['param_schedulers'] = []
                for scheduler in self.param_schedulers:  # type: ignore
                    state_dict = scheduler.state_dict()  # type: ignore
                    checkpoint['param_schedulers'].append(state_dict)

        self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
        save_checkpoint(checkpoint, filepath)

    def consolidate_state_dict(self,
                               state_dict: Dict[str, Any],
                               to: int = 0) -> None:
        r"""
        Consolidate a list of ``state_dict`` s (one per rank) on the target
        rank.
        Arguments:
            to (int): the rank that receives the optimizer states (default: 0).
        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt.
        .. warning:: This needs to be called on all ranks.
        """
        from torch.distributed.optim.zero_redundancy_optimizer import (
            _broadcast_object, _recursive_copy_to_device)

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        # self._sync_param_groups(self.param_groups, self.optim.param_groups)
        # Pull the sharded state from all ranks and store them in rank order
        empty_messenger = torch.tensor([0],
                                       dtype=torch.uint8,
                                       device=get_device())

        # NOTE: We wastefully use `broadcast()` (e.g. instead of `gather()`)
        # due to compatibility issues with NCCL backend; a possible follow-up
        # is to move all sharded state management to RPC RRef
        self._all_state_dicts = []

        from deepspeed import comm as deepspeed_comm

        process_group = self.model.data_parallel_group
        local_rank = deepspeed_comm.get_rank(group=process_group)

        for rank in range(self._world_size):
            global_rank = dist.distributed_c10d._get_global_rank(
                process_group, rank)
            if local_rank == to:
                # Consolidate all local `state_dict`s on this rank, storing on
                # CPU to save GPU memory
                if rank == local_rank:
                    # Directly append own optimizer state
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(
                            state_dict,
                            non_blocking=True,
                            device=torch.device('cpu'),
                        ))
                else:
                    # Receive the optimizer state from the source rank
                    local_state_dict = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=process_group,
                        device=get_device(),
                    )
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(
                            local_state_dict,
                            non_blocking=True,
                            device=torch.device('cpu')))
            else:
                if rank == local_rank:
                    # Send the optimizer state to the target rank
                    _ = _broadcast_object(
                        state_dict,
                        src_rank=global_rank,
                        group=process_group,
                        device=get_device(),
                    )
                elif rank != to:
                    # Discard the received object; `broadcast()` is used for
                    # compatibility reasons
                    _ = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=process_group,
                        device=get_device(),
                    )

    def get_zero_state_dict(self) -> Dict[str, Any]:
        r"""
        Returns the last global optimizer state known to this rank.
        .. warning:
            If the state has not been consolidated to this rank, this raises a
            runtime error, and even if it has, the state may not be up-to-date,
            depending on when :meth:`consolidate_state_dict` was last called.
        """
        if not is_main_process():
            return dict()

        if len(self._all_state_dicts) == 0:
            raise RuntimeError(
                'Optimizer state has not been consolidated on this rank. '
                f'Please call `consolidate_state_dict(to={self.rank})` on '
                'all ranks beforehand if you meant to save the global state.')

        return self._all_state_dicts

    def get_state_dict(self, model):
        from deepspeed.checkpoint.constants import FP32_FLAT_GROUPS
        from deepspeed.utils.zero_to_fp32 import \
            _get_fp32_state_dict_from_zero3_checkpoint

        if not is_main_process():
            return dict()

        # ZeRO 3 case
        if self.model.zero_optimization_partition_weights():
            optim_state = self.get_zero_state_dict()
            fp32_flat_groups = [
                torch.cat(optim_state[i][FP32_FLAT_GROUPS])
                for i in range(len(optim_state))
            ]
            param_shapes = self.model._get_zero_param_shapes()[0]
            param_shapes = OrderedDict(
                {'module.' + k: v
                 for k, v in param_shapes.items()})

            model_state = _get_fp32_state_dict_from_zero3_checkpoint(
                world_size=self._world_size,
                param_shapes=[param_shapes],
                fp32_flat_groups=fp32_flat_groups,
                buffers={})
        else:
            model_state = _get_state_dict(model)

        return model_state
