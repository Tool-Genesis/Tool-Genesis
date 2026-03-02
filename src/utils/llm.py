import os
import re
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_args, **_kwargs):
        return False

import openai
try:
    import requests
except Exception:
    requests = None
import time
import random
import json
from typing import Dict, List, Union, Any
from retry import retry
import base64
from openai import OpenAI
try:
    from PIL import Image
except Exception:
    class _ImageModule:
        class Image:  # type: ignore[empty-body]
            pass

    Image = _ImageModule()
from io import BytesIO

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_API_TYPE = os.environ.get("OPENAI_API_TYPE", "openai")
API_VERSION = os.environ.get("OPENAI_API_VERSION", "")
# 你可以按你项目的习惯放到配置文件里
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_HUNYUAN_BASE_URL = os.getenv(
    "DEEPSEEK_HUNYUAN_BASE_URL",
    DEEPSEEK_BASE_URL,
)
DEEPSEEK_WS_ID = os.getenv("DEEPSEEK_WS_ID") or os.getenv("DEEPSEEK_WSID", "")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

BAILIAN_API_KEY = os.environ.get("BAILIAN_API_KEY", "")
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

LLM_CLIENT_CONFIG = {
    "bailian": {
        "api_key": BAILIAN_API_KEY,
        "base_url": BAILIAN_BASE_URL,
    },
    "openrouter": {
        "api_key": OPENROUTER_API_KEY,
        "base_url": OPENROUTER_BASE_URL,
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL or "https://api.openai.com/v1",
    },
}

client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global client
    if client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for embedding calls.")
        kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**kwargs)
    return client

def get_llm_client(platform: str) -> OpenAI:
    normalized_platform = platform.lower()
    config = LLM_CLIENT_CONFIG.get(normalized_platform)
    if not config:
        raise ValueError(f"Unsupported LLM platform: {platform}")
    api_key = config.get("api_key")
    base_url = config.get("base_url")
    if not api_key or not base_url:
        raise ValueError(f"Missing configuration for LLM platform: {platform}")
    if API_VERSION != "":
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_query={"api-version": API_VERSION},
        )
    return OpenAI(api_key=api_key, base_url=base_url)


@retry(tries=3, backoff=3, max_delay=60)
def call_llm(
    text,
    system_prompt=(
        "You are a helpful and knowledgeable AI assistant. Provide accurate, "
        "concise, and relevant responses based on the user's instructions. "
        "Maintain a polite and neutral tone, and do not add any information "
        "beyond what is asked."
    ),
    model="gpt-4.1-mini",
    max_tokens=120,
    temperature=0.3,
    enable_thinking: bool = False,
    stream: bool = False,
    platform: str = "openai",
):
    """
    Call OpenAI-compatible LLM API via different platforms.

    Args:
        text (str): The input text to process
        system_prompt (str): The system prompt to use
        model (str): Model name for the selected platform
        max_tokens (int): Maximum tokens in the response
        temperature (float): Temperature for response generation
        enable_thinking (bool): Whether to enable thinking (for supported platforms)
        stream (bool): Whether to stream the response
        platform (str): LLM platform identifier, e.g. "bailian" or "openrouter"

    Returns:
        str: The model's response text
    """
    llm_client = get_llm_client(platform)
    try:
        request_kwargs = {}
        if platform.lower() == "bailian":
            request_kwargs["extra_body"] = {
                "enable_thinking": False if not stream else bool(enable_thinking)
            }
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **request_kwargs,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API Call Error: {e}")
        return None


def encode_image_to_base64(image_data: Image.Image) -> str:
    """
    Encodes a PIL Image object to a Base64 string.
    """
    try:
        buffered = BytesIO()
        # You need to specify the format when saving to BytesIO.
        # JPEG is generally a good choice for vision models due to compression.
        # You might want to infer format from original source or choose based on quality needs.
        image_data.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Error encoding image to Base64: {e}")

@retry(tries=3, backoff=3, max_delay=60)
def image_llm(
    user_prompt: str,
    image: Image.Image,
    system_prompt: str = "You are a helpful AI assistant with vision. Provide concise, accurate answers based on the image(s) and the user’s question.",
    model: str = "gpt-4o",  # Changed to gpt-4o as gpt-4o-vision-preview is often just 'gpt-4o' now or specific versions.
    max_tokens: int = 120,
    temperature: float = 0.3
) -> str:
    """
    Sends a user prompt and image(s) to OpenAI's vision-enabled model for analysis.

    Args:
        user_prompt (str): The text prompt from the user.
        image_paths (List[str]): A list of local file paths to the images.
        system_prompt (str): The system prompt to guide the AI's behavior.
        model (str): The OpenAI model to use (e.g., "gpt-4o").
        max_tokens (int): The maximum number of tokens for the AI's response.
        temperature (float): Controls the randomness of the output.

    Returns:
        str: The AI's response.

    Raises:
        FileNotFoundError: If any of the image files are not found.
        IOError: If there's an error reading or encoding an image.
        openai.OpenAIError: For API-related errors from OpenAI.
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    messages_content: List[Dict[str, Any]] = []

    # Add system prompt
    messages_content.append({"type": "text", "text": system_prompt})

    # Add user prompt
    user_message_content: List[Dict[str, str]] = []
    user_message_content.append({"type": "text", "text": user_prompt})

    # Add images by encoding them to Base64
    base64_image = encode_image_to_base64(image)
    user_message_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        },
    })

    messages = [
        {"role": "user", "content": user_message_content},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        print(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

@retry(openai.APIConnectionError, tries=3, backoff=3, max_delay=60)
def call_embedding(
    inputs: Union[str, List[str]],
    model: str = "text-embedding-3-large",
) -> List[List[float]]:
    if isinstance(inputs, str):
        payload = [inputs]
    else:
        payload = inputs

    try:
        response = _get_openai_client().embeddings.create(
            model=model,
            input=payload,
        )
        return [item.embedding for item in response.data]

    except Exception:
        if requests is None:
            raise RuntimeError("Missing dependency 'requests'. Install requirements.txt.")
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {"model": model, "input": payload}
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            headers=headers,
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]

@retry(openai.APIConnectionError, tries=3, backoff=3, max_delay=60)
def call_llm_dp(
    text: str,
    system_prompt: str = "Summarize the following in 2 short sentences:",
    model: str = "DeepSeekV3-0324-SGL-nj",
    max_tokens: int = 16384,
    temperature: float = 0.3,
) -> str:
    """
    Call DeepSeek / HunYuan chat-completion endpoint (OpenAI-compatible).
    Args:
        text (str): The input text to process
        system_prompt (str): The system prompt to use
        model (str): The model to use
        max_tokens (int): Maximum tokens in the response
        temperature (float): Temperature for response generation
       
    Returns:
        The assistant's reply extracted from the JSON response.
    """
    if not DEEPSEEK_API_KEY or not DEEPSEEK_WS_ID:
        raise ValueError(
            "Missing DeepSeek credentials. Set DEEPSEEK_API_KEY and DEEPSEEK_WS_ID "
            "(or legacy DEEPSEEK_WSID)."
        )
    url = f"{DEEPSEEK_HUNYUAN_BASE_URL.rstrip('/')}/openapi/chat/completions"

    # Mandatory headers for authentication + workspace ID
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Wsid": DEEPSEEK_WS_ID,
    }

    # Unique query ID helps the backend with tracking / debugging
    query_id = f"{int(time.time() * 1000)}{random.randint(0, 9999)}"

    payload: Dict = {
        "query_id": query_id,
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text},
        ],
        "model_type": "hunyuan",
        "temperature": temperature,
        "top_p": 1,
        "top_k": 40,
        "output_seq_len": max_tokens,
        "max_input_seq_len": max_tokens,    
        "repetition_penalty": 1,
        "debug_level": 0,
        "stream": False,
        "random_seed": 5610,
        "debug_flag": 0,
        "compatible_with_openai": True,
        "stop": ["</answer>", "</function_call>"],
    }

    if requests is None:
        raise RuntimeError("Missing dependency 'requests'. Install requirements.txt.")
    raw_resp = requests.post(url, headers=headers, json=payload, timeout=90)
    raw_resp.raise_for_status()
    resp_content_dict = raw_resp.json()
    try:
        return resp_content_dict["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected DeepSeek response payload: {resp_content_dict}") from exc

def extract_code(raw: str) -> str:
    """
    Extract the contents of all fenced code blocks (```…```) in the input.
    Supports any language tag (e.g., ```python, ```json, ```pddl, or no tag).
    If multiple code blocks are present, they are concatenated with two newlines.
    If no fences are found, returns the original text plus a trailing newline.
    """
    if not isinstance(raw, str):
        raise TypeError("raw must be a string")

    text = raw.strip()

    # Match ```<optional lang>\n<body>```
    # lang: anything up to end of first line (excluding backticks)
    fence_pat = re.compile(
        r"```[ \t]*(?P<lang>[^\n`]*)\n(?P<body>.*?)(?:\n)?```",
        flags=re.DOTALL,
    )

    blocks = []
    for m in fence_pat.finditer(text):
        lang = (m.group("lang") or "").strip().lower()
        code = (m.group("body") or "").strip()

        # If it looks heavily escape-encoded (lots of literal "\n"), try decoding.
        if code.count("\\n") >= max(5, code.count("\n") * 2):
            try:
                code = bytes(code, "utf-8").decode("unicode_escape")
            except Exception:
                pass

        # Remove stray triple-quotes that sometimes get glued after the language tag:
        # e.g. ```python\"\"\" ... \"\"\"```
        code = re.sub(r'^\s*([\'"]{3})', "", code)
        code = re.sub(r'([\'"]{3})\s*$', "", code)

        # Python-specific "import glue" fixes (only when it looks like python)
        if lang in ("python", "py", "") and ("import " in code or "from " in code):
            code = re.sub(r"([A-Za-z0-9_])from\s+", r"\1\nfrom ", code)
            code = re.sub(r"(typing\s+import\s+.*?)(from\s+)", r"\1\n\2", code)

        # Normalize newlines + trim trailing spaces
        code = code.replace("\r\n", "\n").replace("\r", "\n")
        code = "\n".join(line.rstrip() for line in code.split("\n")).strip()

        blocks.append(code)

    if not blocks:
        return text + "\n"

    return "\n\n".join(blocks).rstrip() + "\n"
