from modelscope import snapshot_download

#下载GLM-4-Voice模型权重到'weights/ZhipuAI/'
snapshot_download('ZhipuAI/glm-4-voice-tokenizer',cache_dir='./weights')
snapshot_download('ZhipuAI/glm-4-voice-decoder',cache_dir='./weights')
snapshot_download('ZhipuAI/glm-4-voice-9b',cache_dir='./weights')
