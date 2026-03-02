from typing import Dict, Optional, Type, Union, ClassVar
import os

from src.core.models.base import (
    BaseModelBackend,
    BaseTokenCounter,
    
)

from src.core.types import(
    UnifiedModelType,
    ModelPlatformType,
    ModelType
)

from src.core.models.openai_model import OpenAIModel
from src.core.models.openrouter_model import OpenRouterModel
from src.core.models.deepseek_model import DeepSeekModel
from src.core.models.vllm_model import VLLMModel
from src.core.models.bailian_model import BailianModel

class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """
    _MODEL_PLATFORM_TO_CLASS_MAP: ClassVar[
        Dict[ModelPlatformType, Type[BaseModelBackend]]
    ] = {
        ModelPlatformType.VLLM: VLLMModel,
        ModelPlatformType.OPENAI: OpenAIModel,
        ModelPlatformType.BAILIAN: BailianModel,
        ModelPlatformType.OPENROUTER: OpenRouterModel,
        ModelPlatformType.DEEPSEEK: DeepSeekModel,
    }
    
    
    @staticmethod
    def create(
        model_platform: Union[ModelPlatformType, str],
        model_type: Union[ModelType, str, UnifiedModelType],
        model_config_dict: Optional[Dict] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> BaseModelBackend:
        r"""Creates an instance of `BaseModelBackend` of the specified type.

        Args:
            model_platform (Union[ModelPlatformType, str]): Platform from
                which the model originates. Can be a string or
                ModelPlatformType enum.
            model_type (Union[ModelType, str, UnifiedModelType]): Model for
                which a backend is created. Can be a string, ModelType enum, or
                UnifiedModelType.
            model_config_dict (Optional[Dict]): A dictionary that will be fed
                into the backend constructor. (default: :obj:`None`)
            token_counter (Optional[BaseTokenCounter], optional): Token
                counter to use for the model. If not provided,
                :obj:`OpenAITokenCounter(ModelType.GPT_4O_MINI)`
                will be used if the model platform didn't provide official
                token counter. (default: :obj:`None`)
            api_key (Optional[str], optional): The API key for authenticating
                with the model service. (default: :obj:`None`)
            url (Optional[str], optional): The url to the model service.
                (default: :obj:`None`)
            timeout (Optional[float], optional): The timeout value in seconds
                for API calls. (default: :obj:`None`)
            **kwargs: Additional model-specific parameters that will be passed
                to the model constructor. For example, Azure OpenAI models may
                require `api_version`, `azure_deployment_name`,
                `azure_ad_token_provider`, and `azure_ad_token`.

        Returns:
            BaseModelBackend: The initialized backend.

        Raises:
            ValueError: If there is no backend for the model.
        """
        # Convert string to ModelPlatformType enum if needed
        if isinstance(model_platform, str):
            try:
                model_platform = ModelPlatformType(model_platform)
            except ValueError:
                raise ValueError(f"Unknown model platform: {model_platform}")

        # Convert string to ModelType enum or UnifiedModelType if needed
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                # If not in ModelType, create a UnifiedModelType
                model_type = UnifiedModelType(model_type)

        model_class: Optional[Type[BaseModelBackend]] = None
        model_type = UnifiedModelType(model_type)

        model_class = ModelFactory._MODEL_PLATFORM_TO_CLASS_MAP.get(
            model_platform
        )

        if model_type == ModelType.STUB:
            model_class = StubModel

        if model_class is None:
            raise ValueError(f"Unknown model platform `{model_platform}`")

        return model_class(
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key,
            url=url,
            token_counter=token_counter,
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    def __parse_model_platform(
        cls, model_platform_str: str
    ) -> ModelPlatformType:
        r"""Parses a string and returns the corresponding ModelPlatformType
        enum.

        Args:
            model_platform_str (str): The platform name as a string. Can be in
                the form "ModelPlatformType.<NAME>" or simply "<NAME>".

        Returns:
            ModelPlatformType: The matching enum value.

        Raises:
            ValueError: If the platform name is not a valid member of
                ModelPlatformType.
        """

        try:
            if model_platform_str.startswith("ModelPlatformType."):
                platform_name = model_platform_str.split('.')[-1]
            else:
                platform_name = model_platform_str.upper()

            if platform_name not in ModelPlatformType.__members__:
                raise ValueError(
                    f"Invalid model platform: {platform_name}. "
                    f"Valid options: "
                    f"{', '.join(ModelPlatformType.__members__.keys())}"
                )

            return ModelPlatformType[platform_name]

        except KeyError:
            raise KeyError(f"Invalid model platform: {model_platform_str}")

    @classmethod
    def __load_yaml(cls, filepath: str) -> Dict:
        r"""Loads and parses a YAML file into a dictionary.

        Args:
            filepath (str): Path to the YAML configuration file.

        Returns:
            Dict: The parsed YAML content as a dictionary.
        """
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)

        return config

    @classmethod
    def __load_json(cls, filepath: str) -> Dict:
        r"""Loads and parses a JSON file into a dictionary.

        Args:
            filepath (str): Path to the JSON configuration file.

        Returns:
            Dict: The parsed JSON content as a dictionary.
        """
        with open(filepath, 'r') as file:
            config = json.load(file)

        return config

    @classmethod
    def create_from_yaml(cls, filepath: str) -> BaseModelBackend:
        r"""Creates and returns a model base backend instance
        from a YAML configuration file.

        Args:
            filepath (str): Path to the YAML file containing model
                configuration.

        Returns:
            BaseModelBackend: An instance of the model backend based on the
                configuration.
        """

        config = cls.__load_yaml(filepath)
        config["model_platform"] = cls.__parse_model_platform(
            config["model_platform"]
        )

        model = ModelFactory.create(**config)

        return model

    @classmethod
    def create_from_json(cls, filepath: str) -> BaseModelBackend:
        r"""Creates and returns a base model backend instance
        from a JSON configuration file.

        Args:
            filepath (str): Path to the JSON file containing model
                configuration.

        Returns:
            BaseModelBackend: An instance of the model backend based on the
                configuration.
        """

        config = cls.__load_json(filepath)
        config["model_platform"] = cls.__parse_model_platform(
            config["model_platform"]
        )

        model = ModelFactory.create(**config)

        return model
