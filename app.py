from flask import Flask, request, render_template, send_file, url_for
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import io
import os
from huggingface_hub import login

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 設定 Hugging Face 訪問令牌(請自行替換)
HUGGINGFACE_TOKEN = "your access token"
login(HUGGINGFACE_TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 載入兩種管線: text-to-image (txt2img) 與 image-to-image (img2img)
# 請依照需要的模型名稱。如果沒有inpainting需求，可使用base txt2img模型，如 stable-diffusion-2-base
txt2img_pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base", #stabilityai/stable-diffusion-2-base
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_image_url = None
    generated_image_path = None

    if request.method == 'POST':
        mode = request.form.get('mode')  # txt2img or img2img
        prompt = request.form.get('prompt')

        # 驗證輸入
        if not prompt:
            return "請輸入提示詞", 400

        if mode == 'img2img':
            # 使用者需要上傳一張初始圖
            if 'init_image' not in request.files or request.files['init_image'].filename == '':
                return "請上傳一張初始圖片", 400
            
            init_file = request.files['init_image']
            init_path = os.path.join(app.config['UPLOAD_FOLDER'], init_file.filename)
            init_file.save(init_path)
            
            init_image = Image.open(init_path).convert("RGB").resize((512,512))
            with torch.autocast(device):
                result = img2img_pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]
        else:
            # txt2img模式
            with torch.autocast(device):
                result = txt2img_pipe(prompt=prompt, height=512, width=512, guidance_scale=7.5, num_inference_steps=50).images[0]

        # 儲存生成圖片
        result_filename = f"generated_{len(os.listdir(UPLOAD_FOLDER))}.png"
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        result.save(result_path)

        generated_image_path = result_path
        generated_image_url = url_for('static', filename='uploads/' + result_filename)

    return render_template('index.html', generated_image_url=generated_image_url, generated_image_path=generated_image_path)

@app.route('/modify', methods=['POST'])
def modify():
    new_prompt = request.form.get('new_prompt')
    image_path = request.form.get('image_path')

    if not new_prompt or not image_path:
        return "請提供新的提示詞以及原圖路徑", 400

    if not os.path.exists(image_path):
        return "找不到原圖檔案", 404

    init_image = Image.open(image_path).convert("RGB").resize((512,512))
    with torch.autocast(device):
        # 再次使用 img2img 的管線，以新提示詞修改現有圖像
        modified = img2img_pipe(prompt=new_prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]

    modified_filename = f"modified_{len(os.listdir(UPLOAD_FOLDER))}.png"
    modified_path = os.path.join(UPLOAD_FOLDER, modified_filename)
    modified.save(modified_path)

    modified_url = url_for('static', filename='uploads/' + modified_filename)

    # 最終回到首頁模板顯示新圖，並可再次修改
    return render_template('index.html', generated_image_url=modified_url, generated_image_path=modified_path)




if __name__ == '__main__':
    app.run(debug=True)