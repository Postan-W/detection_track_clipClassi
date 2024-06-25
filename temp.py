from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
os.environ['MODELSCOPE_CACHE'] = "./main_code/models/"
pipeline = pipeline(task=Tasks.multi_modal_embedding,
    model='damo/multi-modal_clip-vit-large-patch14_336_zh', model_revision='v1.0.1')