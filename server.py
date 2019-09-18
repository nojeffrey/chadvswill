#!/usr/bin/python3
import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

#This is being monitored by supervisord, 'systemctl restart supervisord' to restart this app

classes = ['chad', 'will']
path = Path(__file__).parent

app = Starlette()
app.debug = False
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='static'))


async def setup_learner():
    data = ImageDataBunch.single_from_classes(
        path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data, models.resnet34)
    learn.load('will_chad_stage-1')
    #learn = load_learner(path)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))

    prediction, x, prob = learn.predict(img)
    if str(prediction) == 'will':
        prob = prob[1].item()
        prob = f'{prob:.2f}'
        if prob == '1.00': prob = '.99'
    else:
        prob = prob[0].item()
        prob = f'{prob:.2f}'
        if prob == '1.00': prob = '.99'


    final = str(prediction) + ', Probability: ' + str(prob)
    return JSONResponse({'result': str(final)})
    #prediction = learn.predict(img)[0]
    #return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    #if 'serve' in sys.argv:
    uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
