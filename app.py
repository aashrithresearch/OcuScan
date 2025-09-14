
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastai.vision.all import load_learner, PILImage
import io, os
import base64
from matplotlib import pyplot as plt

# Load FastAI learner
learner = load_learner('/content/export.pkl')
app = FastAPI(title='Fundus Classifier + WolframAlpha')

# Knowledge mapping for only normal and glaucoma
LOCAL_KNOWLEDGE = {
    "normal": "Normal fundus: no signs of disease. Verify image quality: brightness, focus, and centering.",
    "glaucoma": "Glaucoma: characterized by increased optic cup-to-disc ratio and loss of peripheral vision."
}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Load image
    contents = await file.read()
    img = PILImage.create(io.BytesIO(contents))
    pred_class, pred_idx, probs = learner.predict(img)

    # Generate knowledge text
    pred_str = str(pred_class).lower()
    if pred_str in LOCAL_KNOWLEDGE:
        knowledge = LOCAL_KNOWLEDGE[pred_str]
    else:
        knowledge = "No additional Wolfram knowledge available for this class."

    # Generate probability bar chart for normal & glaucoma only
    chart_classes = ['normal', 'glaucoma']
    chart_probs = []
    for c in chart_classes:
        try:
            idx = learner.dls.vocab.index(c)
            chart_probs.append(float(probs[idx]))
        except:
            chart_probs.append(0.0)

    # Create matplotlib figure
    plt.figure(figsize=(4,3))
    plt.bar(chart_classes, chart_probs, color=['green','red'])
    plt.ylim(0,1)
    plt.ylabel('Probability')
    plt.title('Prediction Probability (Normal vs Glaucoma)')

    # Convert plot to PNG bytes
    import io as sysio
    buf = sysio.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return JSONResponse({
        'pred': str(pred_class),
        'probs': [float(p) for p in probs],
        'classes': learner.dls.vocab,
        'knowledge': knowledge,
        'chart_base64': chart_base64
    })
