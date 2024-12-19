'''
Author: Luyao Zhu
Email: luyao001@e.ntu.edu.sg
revised from transformer (V4.29.1) trainer
https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/trainer.py#L230

Updated by Saleh Afzoon
'''
import math
import sys
import os
import time
import shutil

from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm
from packaging import version
from pathlib import Path

import torch

from torch import nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import Dataset

from transformers import Seq2SeqTrainer
from transformers.modeling_utils import unwrap_model, PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from torch.utils.data import DataLoader, RandomSampler
from transformers.data.data_collator import DataCollator
from transformers.utils import is_datasets_available, is_sagemaker_mp_enabled, logging, \
    is_torch_tpu_available, is_apex_available
from transformers.trainer_pt_utils import IterableDatasetShard, get_model_param_count, nested_detach
from transformers.trainer_utils import seed_worker, has_length, HPSearchBackend, speed_metrics, \
    TrainOutput, EvalPrediction
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer_callback import TrainerState
from transformers.integrations import hp_params
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

skip_first_batches = None

logger = logging.get_logger(__name__)

TRAINER_STATE_NAME = "trainer_state.json"

# Name of the files used for checkpointing
if is_datasets_available():
    import datasets


class CustomTrainer(Seq2SeqTrainer):

    def __init__(self,
                 model: Union["PreTrainedModel", nn.Module] = None,
                 args: "TrainingArguments" = None,
                 data_collator: Optional["DataCollator"] = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 tokenizer: Optional["PreTrainedTokenizerBase"] = None,
                 model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
                 compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
                 callbacks: Optional[List["TrainerCallback"]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                 vae_sampler: Union["PreTrainedModel", nn.Module] = None,
                 k: int = None,
                 path: str = None, ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.vae = vae_sampler
        self.k = k
        self.model_type = "vae"
        self.path_model = path
        self.fsdp = None  # Added by Saleh

    # def compute_loss(self, model, inputs, return_outputs=False, eval=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.
    #     Subclass and override for custom behavior.
    #     """

    #     outputs_gen, logits_cnt = self.model(inputs, model_type='both')
    #     loss_cnt = self.model.compute_cnt_loss(logits_cnt, temperature=7., k=self.k, kl_type='sum')

    #     # loss: extraction
    #     loss_gen = self.model.compute_ce_loss(self.model.model, inputs[0], outputs_gen)
    #     if eval:
    #         loss = loss_gen
    #     else:
    #         loss = loss_gen + 0.5 * loss_cnt

    #     return loss

    def compute_loss(self, model, inputs, return_outputs=False, eval=False):
        """
        Custom loss computation. Handles both training and evaluation scenarios.
        """
        # Ensure `inputs` is a dictionary
        if isinstance(inputs, list):
            inputs = inputs[0]  # Extract the relevant dictionary if inputs is a list

        outputs_gen, logits_cnt = model(inputs, model_type='both')
        loss_cnt = model.compute_cnt_loss(logits_cnt, temperature=7., k=self.k, kl_type='sum')

        # loss: extraction
        loss_gen = model.compute_ce_loss(model.model, inputs, outputs_gen)

        if eval:
            loss = loss_gen
        else:
            loss = loss_gen + 0.5 * loss_cnt

        return (loss, outputs_gen) if return_outputs else loss


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        self._train_batch_size = batch_size

        self.train_dataset.vae_sampling(self.vae, k=self.k)
        self.train_dataset.preprocess_fun()
        self.eval_dataset.vae_sampling(self.vae, k=self.k)
        self.eval_dataset.preprocess_fun()

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs

        elif args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError("args.max_steps must be set to a positive value if dataloader does not have a length.")

        delay_optimizer_creation = (
            self.fsdp is not None or is_sagemaker_mp_enabled()
        )

        if args.deepspeed:
            self.model_wrapped = self.model
            self.deepspeed = True
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info("***** Running training *****")

        self.state.epoch = 0
        start_time = time.time()

        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Additional updates to `_inner_training_loop` as needed.

        logger.info("Training loop updated for new transformers.")

        return TrainOutput(self.state.global_step, 0, {})

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Overrides save_model to handle shared weights and avoid safetensors issues.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Handle shared weights by saving with torch.save
        if hasattr(self.model, 'tie_weights') and callable(self.model.tie_weights):
            self.model.tie_weights = lambda: None  # Temporarily disable weight tying
        
        # Use torch.save to save model state_dict
        model_to_save = unwrap_model(self.model)
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        logger.info(f"Model saved to {output_dir}")