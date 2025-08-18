# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from functools import partial
from pathlib import Path
from typing import Any, Generic, Iterable, List, Optional, Type, TypeVar, Union

import torch.distributed
import transformers
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from typing_extensions import Unpack

from megatron.bridge.models.conversion import model_bridge
from megatron.bridge.models.conversion.model_bridge import (
    HFWeightTuple,
    MegatronModelBridge,
    WeightConversionTask,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.models.model_provider import GetModelKwargs, ModelProviderMixin


MegatronModelT = TypeVar("MegatronModelT", bound=MegatronModule)
DataclassT = TypeVar("DataclassT")


class AutoBridge(Generic[MegatronModelT]):
    """
    Automatically select and instantiate the appropriate bridge for a model.

    This unified bridge class combines automatic model detection with full bridge
    functionality for converting models between HuggingFace and Megatron formats.
    It handles the conversion of causal language models (e.g., GPT, Llama, Phi)
    between HuggingFace's transformers library format and Megatron-Core's distributed
    training format. It manages weight mapping, tensor parallelism distribution, and
    configuration translation.

    The bridge supports both directions of conversion:
    - HuggingFace → Megatron: For training or inference with Megatron
    - Megatron → HuggingFace: For saving trained models in HF format

    Args:
        hf_pretrained: Either a PreTrainedCausalLM instance with loaded model,
            or a PretrainedConfig for configuration-only operations

    Example:
        >>> # Load and convert a model to Megatron format
        >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3-8B")
        >>> provider = bridge.to_megatron_provider()
        >>> megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)

        >>> # Export a Megatron model back to HuggingFace format
        >>> bridge.save_hf_pretrained(megatron_model, "./exported_model")

        >>> # Convert weights with custom settings
        >>> for name, weight in bridge.export_hf_weights(
        ...     megatron_model,
        ...     cpu=True
        ... ):
        ...     print(f"Exported {name}: {weight.shape}")

        >>> # Check if a model is supported before loading
        >>> if AutoBridge.can_handle("microsoft/phi-2"):
        ...     bridge = AutoBridge.from_hf_pretrained("microsoft/phi-2")

    Note:
        The bridge automatically detects the model architecture and applies
        the appropriate weight mappings. Custom architectures require implementing
        a MegatronModelBridge subclass.
    """

    def __init__(self, hf_pretrained: PreTrainedCausalLM | PretrainedConfig):
        if not isinstance(hf_pretrained, (PreTrainedCausalLM, PretrainedConfig)):
            raise ValueError("hf_pretrained must be a PreTrainedCausalLM or PretrainedConfig instance")
        self.hf_pretrained: PreTrainedCausalLM | PretrainedConfig = hf_pretrained

    @classmethod
    def list_supported_models(cls) -> list[str]:
        """
        List all model architectures currently supported by the bridge system.

        Returns:
            List of supported HuggingFace model architecture names
        """
        # Get all registered implementations from the dispatch system
        supported = []

        # Access the dispatch registry to find all registered types

        if hasattr(model_bridge.get_model_bridge, "_exact_types"):
            for arch_type in model_bridge.get_model_bridge._exact_types.keys():
                if hasattr(arch_type, "__name__"):
                    supported.append(arch_type.__name__)

        return sorted(supported)

    @classmethod
    def supports(cls, config: Any) -> bool:
        """
        Check if this bridge supports the given model configuration.

        A model is supported if it has at least one architecture ending with 'ForCausalLM'.

        Args:
            config: HuggingFace model config object

        Returns:
            True if this bridge can handle the model, False otherwise
        """
        architectures = getattr(config, "architectures", [])
        if not architectures:
            return False

        return any(arch.endswith("ForCausalLM") for arch in architectures)

    @classmethod
    def from_hf_config(cls, config: PretrainedConfig) -> "AutoBridge":
        """
        Create an AutoBridge from a HuggingFace configuration.

        This method creates a bridge instance from just a model configuration,
        without loading any weights. This is useful for:
        - Creating Megatron models with random initialization
        - Working with model architectures without downloading weights
        - Testing and development scenarios

        Args:
            config: HuggingFace PretrainedConfig instance containing model
                architecture information

        Returns:
            AutoBridge: Bridge instance configured for the architecture

        Raises:
            ValueError: If the configuration is not for a supported CausalLM model

        Example:
            >>> from transformers import AutoConfig
            >>>
            >>> # Load just the configuration
            >>> config = AutoConfig.from_pretrained("meta-llama/Llama-3-8B")
            >>>
            >>> # Create bridge from config (no weights)
            >>> bridge = AutoBridge.from_hf_config(config)
            >>>
            >>> # Create Megatron model with random initialization
            >>> provider = bridge.to_megatron_provider(load_weights=False)
            >>> model = provider.provide_distributed_model(wrap_with_ddp=False)

            >>> # Or use for architecture exploration
            >>> transformer_config = bridge.transformer_config
            >>> print(f"Hidden size: {transformer_config.hidden_size}")
            >>> print(f"Num layers: {transformer_config.num_layers}")

        See Also:
            from_hf_pretrained: Create bridge with loaded weights
            transformer_config: Access the Megatron TransformerConfig
        """
        cls._validate_config(config)
        return cls(config)

    @classmethod
    def from_hf_pretrained(cls, path: Union[str, Path], **kwargs) -> "AutoBridge":
        """
        Load an AutoBridge from a pretrained model, automatically detecting the model type.

        This method loads a model from HuggingFace Hub or a local directory and
        creates a bridge instance ready for conversion operations. The model
        architecture is validated to ensure compatibility.

        Args:
            path: HuggingFace model ID or path to model directory
                Examples: "meta-llama/Llama-3-8B", "./my_model"
            **kwargs: Additional arguments passed to HuggingFace from_hf_pretrained
                Common options include:
                - torch_dtype: Model precision (torch.float16, torch.bfloat16)
                - device_map: Device placement strategy ("auto", "cuda:0", etc.)
                - trust_remote_code: Allow custom model code execution
                - attn_implementation: Attention implementation ("flash_attention_2", etc.)

        Returns:
            AutoBridge: Bridge instance with loaded model

        Raises:
            ValueError: If the model architecture is not supported

        Example:
            >>> # Basic loading
            >>> bridge = AutoBridge.from_hf_pretrained("gpt2")

            >>> # Load with specific settings
            >>> bridge = AutoBridge.from_hf_pretrained(
            ...     "meta-llama/Llama-3-8B",
            ...     torch_dtype=torch.float16,
            ...     device_map="auto"
            ... )

            >>> # Works with local paths too
            >>> bridge = AutoBridge.from_hf_pretrained("/path/to/model")
        """
        # First load just the config to check architecture support
        try:
            config = AutoConfig.from_pretrained(path, trust_remote_code=kwargs.get("trust_remote_code", False))
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from {path}. "
                f"Ensure the path is valid and contains a config.json file. "
                f"Error: {e}"
            )

        cls._validate_config(config, str(path))

        try:
            return cls(PreTrainedCausalLM.from_pretrained(path, **kwargs))
        except Exception as e:
            raise ValueError(f"Failed to load model with AutoBridge: {e}") from e

    @classmethod
    def can_handle(cls, path: Union[str, Path], trust_remote_code: bool = False) -> bool:
        """
        Check if the bridge can handle the model at the given path.

        This method allows you to verify model compatibility before attempting
        to load it, which can be useful for validation or UI feedback.

        Args:
            path: Path to model directory or HuggingFace model ID
                Examples: "meta-llama/Llama-3-8B", "/models/my_model"
            trust_remote_code: Whether to trust remote code when loading config.
                Set to True for models that use custom modeling code.

        Returns:
            bool: True if the bridge supports the model, False otherwise

        Example:
            >>> # Check if a model is supported
            >>> if AutoBridge.can_handle("meta-llama/Llama-3-8B"):
            ...     print("Model is supported!")
            ... else:
            ...     print("Model requires a custom bridge implementation")
        """
        try:
            config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
            return cls.supports(config)
        except Exception:
            return False

    def load_hf_weights(self, model: list[MegatronModelT], hf_path: str | Path | None = None) -> None:
        """
        Load HuggingFace weights into a Megatron model.

        This method handles the conversion and distribution of weights from
        HuggingFace format to Megatron's distributed format, including proper
        tensor parallel and pipeline parallel distribution.

        Args:
            model: List of Megatron model instances (one per virtual pipeline stage)
            hf_path: Optional path to load weights from. If None, uses weights
                from the bridge's hf_pretrained instance

        Returns:
            The input model with loaded weights

        Raises:
            ValueError: If hf_path is None and bridge was created without weights

        Example:
            >>> # Load weights from bridge's pretrained model
            >>> bridge = AutoBridge.from_hf_pretrained("gpt2")
            >>> megatron_model = create_megatron_model()  # Your model creation
            >>> bridge.load_hf_weights(megatron_model)

            >>> # Load weights from a different checkpoint
            >>> bridge.load_hf_weights(megatron_model, "./finetuned_model")
        """
        if hf_path is None:
            if not isinstance(self.hf_pretrained, PreTrainedCausalLM):
                raise ValueError("hf_path is required when hf_pretrained is not a PreTrainedCausalLM instance")
            pre_trained = self.hf_pretrained
        else:
            pre_trained = PreTrainedCausalLM.from_pretrained(hf_path)
        self._model_bridge.load_weights_hf_to_megatron(model, pre_trained)

        return model

    def export_hf_weights(
        self,
        model: list[MegatronModelT],
        cpu: bool = False,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable["HFWeightTuple"]:
        """
        Export Megatron model weights to HuggingFace format.

        This method yields weight tensors in HuggingFace format, handling the
        gathering of distributed tensors and format conversion. It's useful for
        streaming weight export or custom processing. All ranks get full tensors.

        Args:
            model: Megatron model instance or list of instances
            cpu: Whether to move tensors to CPU before yielding
            show_progress: Display progress bar during export
            conversion_tasks (Optional[List[WeightConversionTask]]): Pre-built conversion tasks.
                If not provided, tasks will be built automatically from the models.
                *Please note that this is an advanced feature and should be used with caution.
                The tasks needs to be built with the `get_conversion_tasks` method first and
                carefully adjust based on your needs.*


        Yields:
            HFWeightTuple: Named tuples of (param_name, weight_tensor)

        Example:
            >>> # Export and process weights
            >>> for name, weight in bridge.export_hf_weights(model):
            ...     print(f"{name}: {weight.shape}")

            >>> # Export with specific settings
            >>> weights = list(bridge.export_hf_weights(
            ...     model,
            ...     cpu=True
            ... ))
        """
        dispatch_instance = (self._get_causal_lm_architecture(), self._get_model_instance(model))
        return model_bridge.stream_weights_megatron_to_hf(
            dispatch_instance,
            model,
            self.hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
        )

    def save_hf_pretrained(self, model: list[MegatronModelT], path: str | Path, show_progress: bool = True) -> None:
        """
        Save a Megatron model in HuggingFace format.

        This method exports the complete model including configuration, tokenizer,
        and weights to a directory that can be loaded with HuggingFace's
        from_pretrained methods.

        Args:
            model: Megatron model instance or list of instances
            path: Directory path to save the model
            show_progress: Display progress bar during weight export

        Example:
            >>> # Save model after training
            >>> bridge.save_hf_pretrained(megatron_model, "./my_finetuned_model")

            >>> # Load the saved model with HuggingFace
            >>> from transformers import AutoModelForCausalLM
            >>> hf_model = AutoModelForCausalLM.from_pretrained("./my_finetuned_model")

        Note:
            This method is collective - all ranks must call it. Only rank 0
            saves the configuration files, while weight saving is coordinated
            across all ranks.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Distributed training, only rank 0 saves artifacts
            if torch.distributed.get_rank() == 0:
                self.hf_pretrained.save_artifacts(path)
        else:
            # No distributed training, save artifacts
            self.hf_pretrained.save_artifacts(path)

        self.save_hf_weights(model, path, show_progress)

    def save_hf_weights(self, model: list[MegatronModelT], path: str | Path, show_progress: bool = True) -> None:
        """
        Save Megatron model weights in HuggingFace safetensors format.

        This method exports only the model weights (not configuration or tokenizer)
        to safetensors files compatible with HuggingFace. It uses streaming save
        to handle large models efficiently without requiring all weights in memory
        at once.

        The weights are gathered from distributed ranks and saved in the standard
        HuggingFace sharded format when the model is large.

        Args:
            model: Megatron model instance or list of instances
            path: Directory path where weight files will be saved
            show_progress: Display progress bar during export

        Raises:
            ValueError: If the state source doesn't support streaming save

        Example:
            >>> # Save just the weights
            >>> bridge.save_hf_weights(megatron_model, "./model_weights")

            >>> # Save without progress bar (useful in scripts)
            >>> bridge.save_hf_weights(megatron_model, "./weights", show_progress=False)

        Note:
            - This method is collective and must be called by all ranks
            - Uses safetensors format for efficient loading and security
            - Automatically handles model sharding for large models
            - The saved weights can be loaded with HuggingFace's from_pretrained
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        dispatch_instance = (self._get_causal_lm_architecture(), self._get_model_instance(model))
        generator = model_bridge.stream_weights_megatron_to_hf(
            dispatch_instance, model, self.hf_pretrained, cpu=True, show_progress=show_progress
        )

        # Check if the state source is SafeTensorsStateSource for streaming save.
        if (
            hasattr(self.hf_pretrained, "state")
            and hasattr(self.hf_pretrained.state, "source")
            and isinstance(self.hf_pretrained.state.source, SafeTensorsStateSource)
        ):
            self.hf_pretrained.state.source.save_generator(generator, path)
        else:
            raise ValueError("The state source is not a SafeTensorsStateSource, cannot save in streaming mode.")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def save_megatron_model(self, model: list[MegatronModule], path: str | Path) -> None:
        """
        Save a Megatron model in native Megatron checkpoint format without optimizer
        state.

        This method saves the model in Megatron's native checkpoint format, which
        can be loaded directly by Megatron for training or inference. The checkpoint
        includes the model configuration and weights, NO optimizer state or other
        artifacts.

        Args:
            model: Megatron model instance or list of instances
            path: Directory path where the checkpoint will be saved
            ckpt_format: Checkpoint format to use ("torch_dist" or other supported formats)

        Example:
            >>> # Save model checkpoint after conversion
            >>> bridge.save_megatron_model(megatron_model, "./megatron_checkpoint")

        Note:
            - This method is collective and must be called by all ranks
            - The saved checkpoint can be loaded with Megatron's checkpoint loading utilities
            - The checkpoint format follows Megatron's standard structure for compatibility
        """
        try:
            from megatron.bridge.training.model_load_save import save_megatron_model
        except ImportError:
            raise ImportError("megatron.bridge.training is not available.")
        save_megatron_model(model, path)

    def load_megatron_model(self, path: str | Path, **kwargs: Unpack[GetModelKwargs]) -> list[MegatronModelT]:
        """
        Load a Megatron model from a native Megatron checkpoint.

        This method loads a model from a Megatron checkpoint that was saved using
        the save_megatron_model method. It reads the checkpoint configuration,
        creates the appropriate model provider, and loads the weights.

        Args:
            path: Directory path where the Megatron checkpoint is stored
            **kwargs: Additional arguments passed to the model provider

        Returns:
            List of Megatron model instances loaded from the checkpoint

        Example:
            >>> # Load a previously saved Megatron model
            >>> bridge = AutoBridge.from_hf_config(config)
            >>> model = bridge.load_megatron_model("./megatron_checkpoint")

            >>> # Load and specify model configuration
            >>> model = bridge.load_megatron_model(
            ...     "./megatron_checkpoint",
            ...     wrap_with_ddp=False
            ... )

        Note:
            - This method is collective and must be called by all ranks
            - The checkpoint must have been saved with save_megatron_model
            - The model architecture must match the bridge configuration
        """
        try:
            from megatron.bridge.training.model_load_save import load_megatron_model
        except ImportError:
            raise ImportError("megatron.bridge.training is not available.")

        checkpoint_path = Path(path)

        # Check for iter_* folders
        iter_folders = [f for f in checkpoint_path.iterdir() if f.is_dir() and f.name.startswith("iter_")]

        if iter_folders:
            # Find the folder with the largest iteration number
            def get_iter_number(folder_name):
                try:
                    return int(folder_name.replace("iter_", ""))
                except ValueError:
                    return -1  # Invalid format, put at the end

            latest_iter = max(iter_folders, key=lambda f: get_iter_number(f.name))
            checkpoint_path = checkpoint_path / latest_iter.name
        # else: checkpoint_path remains as the input path (no iter folders found)

        # Load the state dict
        model = load_megatron_model(
            str(checkpoint_path),
            use_cpu_init=True,
        )
        return model if isinstance(model, list) else [model]

    @classmethod
    def import_ckpt(
        cls,
        hf_model_id: str | Path,
        megatron_path: str | Path,
        **kwargs,
    ) -> None:
        """
        Import a HuggingFace model and save it as a Megatron checkpoint.

        This is a convenience method that combines loading a HuggingFace model,
        converting it to Megatron format, and saving it as a native Megatron
        checkpoint. This is useful for preparing models for Megatron training
        or creating Megatron checkpoints from pretrained HuggingFace models.

        Args:
            hf_model_id: HuggingFace model ID or path to model directory
                Examples: "meta-llama/Llama-3-8B", "./my_model"
            megatron_path: Directory path where the Megatron checkpoint will be saved
            **kwargs: Additional arguments passed to from_hf_pretrained
                Common options include:
                - torch_dtype: Model precision (torch.float16, torch.bfloat16)
                - device_map: Device placement strategy ("auto", "cuda:0", etc.)
                - trust_remote_code: Allow custom model code execution
                - attn_implementation: Attention implementation ("flash_attention_2", etc.)

        Example:
            >>> # Basic import
            >>> AutoBridge.import_ckpt(
            ...     "meta-llama/Llama-3-8B",
            ...     "./megatron_checkpoints/llama3_8b"
            ... )

            >>> # Import with specific settings
            >>> AutoBridge.import_ckpt(
            ...     "meta-llama/Llama-3-8B",
            ...     "./megatron_checkpoints/llama3_8b",
            ...     torch_dtype=torch.float16,
            ...     device_map="auto"
            ... )
        """
        # Load the HuggingFace model
        bridge = cls.from_hf_pretrained(hf_model_id, **kwargs)

        # Convert to Megatron model
        megatron_model = bridge.to_megatron_model(wrap_with_ddp=False, use_cpu_initialization=True)

        # Save as Megatron checkpoint
        bridge.save_megatron_model(megatron_model, megatron_path)

    def export_ckpt(
        self,
        megatron_path: str | Path,
        hf_path: str | Path,
        show_progress: bool = True,
    ) -> None:
        """
        Export a Megatron checkpoint to HuggingFace format.

        This is a convenience method that loads a Megatron checkpoint and
        exports it to HuggingFace format. This is useful for sharing trained
        models or deploying them with HuggingFace inference tools.

        Args:
            megatron_path: Directory path where the Megatron checkpoint is stored
            hf_path: Directory path where the HuggingFace model will be saved
            show_progress: Display progress bar during weight export

        Example:
            >>> # Basic export
            >>> bridge = AutoBridge.from_hf_config(config)
            >>> bridge.export_ckpt(
            ...     "./megatron_checkpoints/my_model",
            ...     "./hf_exports/my_model"
            ... )

            >>> # Export with specific settings
            >>> bridge.export_ckpt(
            ...     "./megatron_checkpoints/my_model",
            ...     "./hf_exports/my_model",
            ...     show_progress=False
            ... )

            >>> # Load the exported model with HuggingFace
            >>> from transformers import AutoModelForCausalLM
            >>> hf_model = AutoModelForCausalLM.from_pretrained("./hf_exports/my_model")
        """
        try:
            from megatron.bridge.training.model_load_save import temporary_distributed_context
        except ImportError:
            raise ImportError("megatron.bridge.training is not available.")

        # Export ckpt performs on CPU
        with temporary_distributed_context(backend="gloo"):
            # Load the Megatron model
            megatron_model = self.load_megatron_model(megatron_path, wrap_with_ddp=False)

            # Save in HuggingFace format
            self.save_hf_pretrained(megatron_model, hf_path, show_progress=show_progress)

    def push_to_hub(self, path: str | Path) -> None: ...

    def to_megatron_model(
        self,
        load_weights: bool = True,
        hf_path: str | Path | None = None,
        **kwargs: Unpack[GetModelKwargs],
    ) -> list[MegatronModelT]:
        provider = self.to_megatron_provider(load_weights, hf_path)
        return provider.provide_distributed_model(**kwargs)

    def to_megatron_provider(self, load_weights: bool = True, hf_path: str | Path | None = None) -> GPTModelProvider:
        """
        Convert to a Megatron model provider.

        This method creates a GPTModelProvider configured to match the HuggingFace
        model's architecture. The provider can then be used to instantiate
        Megatron models for training or inference.

        Args:
            load_weights: Whether to configure the provider to load weights
                from HuggingFace format. If False, creates model with random
                initialization.
            hf_path: Optional path to load weights from. If None, uses weights
                from the bridge's hf_pretrained instance. Useful for loading
                weights from a different checkpoint.

        Returns:
            GPTModelProvider: A configured model provider ready to create
                Megatron models

        Example:
            >>> # Create provider and model with loaded weights
            >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3-8B")
            >>> provider = bridge.to_megatron_provider()
            >>> model = provider.get_model()

            >>> # Create provider without loading weights (for training from scratch)
            >>> provider = bridge.to_megatron_provider(load_weights=False)
            >>> model = provider.get_model()  # Random initialization

            >>> # Load weights from a different checkpoint
            >>> bridge = AutoBridge.from_hf_config(config)  # Config only
            >>> provider = bridge.to_megatron_provider(hf_path="./finetuned_model")
            >>> model = provider.get_model()  # Loads finetuned weights

        See Also:
            GPTModelProvider: The provider class for creating models
            load_weights: Method to load weights into existing models
        """

        provider: ModelProviderMixin = self._model_bridge.provider_bridge(self.hf_pretrained)

        if load_weights:
            # Skip weights initialization since we are going to load weights
            provider.perform_initialization = False
            if hf_path is None:
                provider.register_pre_wrap_hook(
                    partial(self._model_bridge.load_weights_hf_to_megatron, self.hf_pretrained)
                )
            else:
                # Load from specified path
                pre_trained = PreTrainedCausalLM.from_pretrained(hf_path)
                provider.register_pre_wrap_hook(partial(self._model_bridge.load_weights_hf_to_megatron, pre_trained))

        return provider

    def get_conversion_tasks(
        self,
        megatron_model: Union[MegatronModelT, List[MegatronModelT]],
        hf_path: str | Path | None = None,
    ) -> List["WeightConversionTask"]:
        """Get the conversion tasks for weight conversion between HuggingFace and Megatron formats.

        This method returns the planned conversion tasks that would be executed during
        weight conversion in either direction. Each task contains information about parameter
        mappings, source and target parameters, and the conversion logic required.

        The tasks can be used for both HF→Megatron and Megatron→HF conversions since they
        contain bidirectional mapping information.

        Args:
            megatron_model: Megatron model instance or list of instances (one per
                virtual pipeline stage) that participate in the conversion.
            hf_path: Optional path to load HF weights from. If None, uses weights
                from the bridge's hf_pretrained instance.

        Returns:
            List[WeightConversionTask]: List of conversion tasks that would be executed.
                Each task contains:
                - param_name: Megatron parameter name
                - mapping: The parameter mapping object handling the conversion
                - pp_rank: Pipeline parallel rank that owns the parameter
                - vp_stage: Virtual pipeline stage index
                - megatron_module: Reference to the Megatron module owning the parameter
                - param_weight: The actual parameter tensor

        Example:
            >>> bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
            >>> megatron_model = bridge.to_megatron_model(load_weights=False, wrap_with_ddp=False)
            >>> tasks = bridge.get_conversion_tasks(megatron_model)
            >>>
            >>> for task in tasks:
            ...     # For HF→Megatron direction
            ...     print(f"HF param {task.mapping.hf_param} -> Megatron param {task.param_name}")
            ...
            ...     # For Megatron→HF direction
            ...     hf_params = task.mapping.hf_param
            ...     if isinstance(hf_params, str):
            ...         print(f"Megatron param {task.param_name} -> HF param {hf_params}")
            ...     else:
            ...         print(f"Megatron param {task.param_name} -> HF params {list(hf_params.values())}")
            ...
            ...     print(f"  Mapping type: {type(task.mapping).__name__}")
            ...     print(f"  PP rank: {task.pp_rank}, VP stage: {task.vp_stage}")

        Note:
            This method is useful for:
            - Debugging weight conversion issues in both directions
            - Understanding parameter mappings between formats
            - Custom weight conversion implementations
            - Analyzing model structure differences
            - Verifying parameter alignment and shapes
        """
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        if hf_path is None:
            if not isinstance(self.hf_pretrained, PreTrainedCausalLM):
                raise ValueError("hf_path is required when hf_pretrained is not a PreTrainedCausalLM instance")
            pre_trained = self.hf_pretrained
        else:
            pre_trained = PreTrainedCausalLM.from_pretrained(hf_path)

        return self._model_bridge.build_conversion_tasks(pre_trained, megatron_model)

    @property
    def transformer_config(self) -> TransformerConfig:
        _model_provider = self.to_megatron_provider(load_weights=False)
        return self._create_config_from_provider(_model_provider, TransformerConfig)

    @property
    def mla_transformer_config(self) -> MLATransformerConfig:
        _model_provider = self.to_megatron_provider(load_weights=False)
        return self._create_config_from_provider(_model_provider, MLATransformerConfig)

    @property
    def _model_bridge(self) -> "MegatronModelBridge":
        return model_bridge.get_model_bridge(self._get_causal_lm_architecture())

    def _get_causal_lm_architecture(self):
        """
        Get the CausalLM architecture class from the HuggingFace model.

        Returns:
            The transformers class for the CausalLM architecture

        Raises:
            ValueError: If no CausalLM architecture is found or if the class cannot be imported
        """
        if isinstance(self.hf_pretrained, PreTrainedCausalLM):
            architectures = getattr(self.hf_pretrained.config, "architectures", [])
        else:
            architectures = getattr(self.hf_pretrained, "architectures", [])

        if not architectures:
            raise ValueError(
                "\n✗ No architectures found in model config\n\n"
                "The model configuration does not specify any architectures.\n"
                "This is required for determining the model type."
            )

        causal_lm_arch = None
        for architecture_name in architectures:
            # TODO: Can we improve this?
            if architecture_name.endswith("ForCausalLM"):
                causal_lm_arch = architecture_name
                break

        if not causal_lm_arch:
            raise ValueError(
                f"\n✗ No CausalLM architecture found\n\n"
                f"Model architectures: {architectures}\n\n"
                f"None of the architectures end with 'ForCausalLM'.\n"
                f"This bridge only supports causal language models.\n"
                f"For other model types, use a different bridge class."
            )

        try:
            return getattr(transformers, causal_lm_arch)
        except AttributeError:
            raise ValueError(
                f"\n✗ Architecture class '{causal_lm_arch}' not found in transformers\n\n"
                f"This could mean:\n"
                f"1. The model requires a newer version of transformers\n"
                f"2. The model uses a custom modeling file not in the standard library\n"
                f"3. There's a typo in the architecture name\n\n"
                f"Please verify your transformers installation and the model requirements."
            )

    @classmethod
    def _validate_config(cls, config: PretrainedConfig, path: str | None = None) -> None:
        # Check if this is a causal LM model
        if not cls.supports(config):
            architectures = getattr(config, "architectures", [])
            raise ValueError(
                f"\n✗ Model architecture not supported by AutoBridge\n\n"
                f"Model: {path}\n"
                f"Architectures: {architectures}\n\n"
                f"AutoBridge only supports models with architectures ending in 'ForCausalLM'.\n"
                f"Found architectures that don't match this pattern.\n\n"
                f"If this is a different model type (e.g., Vision, Sequence-to-Sequence),\n"
                f"you may need to use a different bridge class."
            )

        # Check if we have an implementation for this specific architecture
        architecture = None
        for arch_name in config.architectures:
            if arch_name.endswith("ForCausalLM"):
                architecture = arch_name
                break

        if architecture:
            # Try to get the transformers class to check dispatch registration
            try:
                arch_class = getattr(transformers, architecture)
                # Test if we have a registered implementation
                # Check if this architecture is registered in the dispatch system
                has_implementation = False
                if hasattr(model_bridge.get_model_bridge, "_exact_types"):
                    has_implementation = arch_class in model_bridge.get_model_bridge._exact_types

                if not has_implementation:
                    # Get list of supported models
                    supported_models = cls.list_supported_models()

                    raise ValueError(
                        f"\n✗ Model architecture '{architecture}' is not yet supported\n\n"
                        f"Model: {path}\n"
                        f"Architecture: {architecture}\n\n"
                        f"Currently supported architectures:\n"
                        + "\n".join(f"  • {model}" for model in supported_models)
                        + f"\n\nTo add support for {architecture}, you need to:\n"
                        f"1. Create a new bridge class that inherits from MegatronModelBridge\n"
                        f"2. Implement the required methods (provider_bridge, mapping_registry)\n"
                        f"3. Register it with @MegatronModelBridge.register_bridge decorator\n\n"
                        f"Example implementation:\n"
                        f"  from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge\n"
                        f"  from transformers import {architecture}\n"
                        f"  from megatron.core.models.gpt import GPTModel\n\n"
                        f"  @MegatronModelBridge.register_bridge(source={architecture}, target=GPTModel)\n"
                        f"  class Megatron{architecture.replace('ForCausalLM', '')}Bridge(MegatronModelBridge):\n"
                        f"      def provider_bridge(self, hf_pretrained):\n"
                        f"          # Return a ModelProvider instance\n"
                        f"          ...\n\n"
                        f"      def mapping_registry(self):\n"
                        f"          # Return a MegatronMappingRegistry with weight mappings\n"
                        f"          ...\n\n"
                        f"For reference implementations, see:\n"
                        f"  • src/megatron/bridge/models/llama/llama_bridge.py\n"
                        f"  • src/megatron/bridge/models/qwen/qwen_2_causal_bridge.py"
                    ) from None
            except AttributeError:
                raise ValueError(
                    f"\n✗ Could not find architecture class '{architecture}' in transformers\n\n"
                    f"This might be because:\n"
                    f"1. The transformers library version is too old\n"
                    f"2. The model requires a custom modeling file\n"
                    f"3. The architecture name is incorrect\n\n"
                    f"Please check your transformers installation and model requirements."
                )

    def _get_model_instance(self, model: list[MegatronModelT]) -> MegatronModelT:
        model_instance = model[0]
        while hasattr(model_instance, "module"):
            model_instance = model_instance.module
        return model_instance

    def _create_config_from_provider(self, source_obj: Any, target_dataclass: Type[DataclassT]) -> DataclassT:
        kwargs = {}
        for field in dataclasses.fields(target_dataclass):
            if hasattr(source_obj, field.name):
                kwargs[field.name] = getattr(source_obj, field.name)
        return target_dataclass(**kwargs)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__

        lines_for_build = []

        # Format hf_pretrained
        hf_repr_actual_lines = repr(self.hf_pretrained).splitlines()
        if hf_repr_actual_lines:
            # First line of hf_pretrained part
            lines_for_build.append(f"  (hf_pretrained): {hf_repr_actual_lines[0]}")
            # Subsequent lines of hf_pretrained part, indented
            for line in hf_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (hf_pretrained): ")  # Fallback for empty repr

        # Format model bridge
        mb_repr_actual_lines = repr(self._model_bridge).splitlines()
        if mb_repr_actual_lines:
            # First line of model bridge part
            lines_for_build.append(f"  (model_bridge): {mb_repr_actual_lines[0]}")
            # Subsequent lines of model bridge part, indented
            for line in mb_repr_actual_lines[1:]:
                lines_for_build.append(f"  {line}")
        else:
            lines_for_build.append("  (model_bridge): ")  # Fallback for empty repr

        return f"{class_name}(\n" + "\n".join(lines_for_build) + "\n)"
