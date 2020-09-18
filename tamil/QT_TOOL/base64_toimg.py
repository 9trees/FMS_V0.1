import io
import base64
import PIL
import json
from PIL import Image
from glob import glob
from pathlib import Path

annotation_path ='/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/CHILI_LEARN/images/annotation'

jsons = glob(annotation_path+'/*.json')

for i in jsons:
    img_name = Path(i).name.split('.')[0]
    with open(i) as f:
        data = json.load(f)

    img_data = data['imageData']
    im = Image.open(io.BytesIO(base64.b64decode(img_data)))
    im.save(annotation_path+'/'+img_name+".png")
