INPUT_IMAGE_SIZE = [960, 960]       # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
MAX_SEQ_LENGTH = 1024               # The max token length. Note, this value include the 10 tokens for system prompt and (HEIGHT_FACTOR * WIDTH_FACTOR) tokens for image prompt. Hence, only (MAX_SEQ_LENGTH - (HEIGHT_FACTOR * WIDTH_FACTOR) - 10) tokens for query + response.
PROMPT_HEAD_LENGTH = 5              # <s><|im_start|>user\n<img>
