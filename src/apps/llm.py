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
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_HUNYUAN_BASE_URL = os.getenv(
    "DEEPSEEK_HUNYUAN_BASE_URL",
    DEEPSEEK_BASE_URL,
)
DEEPSEEK_WS_ID = os.getenv("DEEPSEEK_WS_ID") or os.getenv("DEEPSEEK_WSID", "")

openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_BASE_URL

@retry(tries=3, backoff=3, max_delay=60)
def call_llm_1(
    text, 
    system_prompt="You are a helpful and knowledgeable AI assistant. Provide accurate, concise, and relevant responses based on the user’s instructions. Maintain a polite and neutral tone, and do not add any information beyond what is asked.", 
    model="gpt-4o-mini", 
    max_tokens=120, 
    temperature=0.3
):
    """
    Call OpenAI API with the given text and parameters.
    
    Args:
        text (str): The input text to process
        system_prompt (str): The system prompt to use
        model (str): The model to use
        max_tokens (int): Maximum tokens in the response
        temperature (float): Temperature for response generation
        
    Returns:
        str: The model's response text
    """
    if requests is None:
        raise RuntimeError("Missing dependency 'requests'. Install requirements.txt.")
    base_url = OPENAI_BASE_URL or "https://api.openai.com/v1"
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    res = requests.post(url, headers=headers, json=payload)
    return res.json()["choices"][0]["message"]["content"]

@retry(tries=3, backoff=3, max_delay=60)
def call_llm(
    text, 
    system_prompt="You are a helpful and knowledgeable AI assistant. Provide accurate, concise, and relevant responses based on the user's instructions. Maintain a polite and neutral tone, and do not add any information beyond what is asked.", 
    model="gpt-4o-mini",
    max_tokens=4200,
    temperature=0.1,
    api_key=None
):
    """
    Call custom OpenAI-compatible API at https://one.ooo.cool/v1 with the given text and parameters.
    
    Args:
        text (str): The input text to process
        system_prompt (str): The system prompt to use
        model (str): The model to use
        max_tokens (int): Maximum tokens in the response
        temperature (float): Temperature for response generation
        api_key (str, optional): Custom API key. If None, uses OPENAI_API_KEY env var
        
    Returns:
        str: The model's response text
    """
    # 使用可配置的自定义BASE_URL
    base_url = os.getenv("ONE_OOO_BASE_URL", "https://one.ooo.cool/v1")
    url = f"{base_url}/chat/completions"
    
    # 使用自定义api_key或环境变量
    used_api_key = api_key if api_key is not None else OPENAI_API_KEY
    
    headers = {
        "Authorization": f"Bearer {used_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if requests is None:
        raise RuntimeError("Missing dependency 'requests'. Install requirements.txt.")
    try:
        res = requests.post(url, headers=headers, json=payload)
        
        # 检查HTTP状态码
        if res.status_code != 200:
            print(f"API请求失败，状态码: {res.status_code}")
            print(f"响应内容: {res.text}")
            raise Exception(f"API请求失败，状态码: {res.status_code}, 响应: {res.text}")
        
        # 解析JSON响应
        try:
            response_data = res.json()
        except Exception as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {res.text}")
            raise Exception(f"JSON解析失败: {e}, 原始响应: {res.text}")
        
        # 检查响应格式
        if "choices" not in response_data:
            print(f"API响应格式错误，缺少'choices'字段")
            print(f"完整响应: {response_data}")
            raise Exception(f"API响应格式错误，缺少'choices'字段。完整响应: {response_data}")
        
        if not response_data["choices"] or len(response_data["choices"]) == 0:
            print(f"API响应中choices为空")
            print(f"完整响应: {response_data}")
            raise Exception(f"API响应中choices为空。完整响应: {response_data}")
        
        if "message" not in response_data["choices"][0]:
            print(f"API响应格式错误，choices[0]中缺少'message'字段")
            print(f"完整响应: {response_data}")
            raise Exception(f"API响应格式错误，choices[0]中缺少'message'字段。完整响应: {response_data}")
        
        return response_data["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        raise Exception(f"网络请求错误: {e}")
    except Exception as e:
        print(f"调用自定义API时发生错误: {e}")
        raise

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

@retry(openai.APIConnectionError,tries=3, backoff=3, max_delay=60)
def call_embedding(
    inputs: Union[str, List[str]],
    model: str = "code-embedding-2",
) -> List[List[float]]:
    """
    Call OpenAI Embeddings API with retry.

    Args:
        inputs (str or List[str]): A single text or list of texts to embed.
        model (str): Name of the embedding model (default: 'code-embedding-2').

    Returns:
        List[List[float]]: A list of embedding vectors—one per input string.
    """
    # Normalize to a list
    if isinstance(inputs, str):
        payload = [inputs]
    else:
        payload = inputs

    # Call the API
    response = openai.embeddings.create(
        model=model,
        input=payload
    )
    
    # Pull out embeddings
    embeddings = [item.embedding for item in response.data]
    return embeddings

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

def extract_code(markdown: str) -> str:
    """
    Extract the contents of all fenced code blocks (```…```) in the input.
    If multiple code blocks are present, they are concatenated with two newlines.
    If no fences are found, returns the original text plus a trailing newline.
    """
    # Matches ```<optional-lang>\n ... ``` (non-greedy)
    pattern = r"```[^\n]*\n(.*?)(?=\n```)"  
    blocks = re.findall(pattern, markdown, re.DOTALL)
    if blocks:
        # strip leading/trailing whitespace on each block and join them
        return "\n\n".join(block.strip() for block in blocks) + "\n"
    # fallback: return whole text
    return markdown.strip() + "\n"
