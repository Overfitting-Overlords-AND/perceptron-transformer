from os import walk
import fastapi
import processor
# from contextlib import asynccontextmanager

app = fastapi.FastAPI()


@app.get("/")
def on_root():
    return {"message": "Hello App"}


@app.post("/tell_me_stories")
async def on_tell_me_stories(request: fastapi.Request):
    text = (await request.json())["text"]
    # print("Input text:", text)
    tp = processor.TextProcessor(text)
    # print("Processed text:", tp.start())
    return {"story": tp.start()}
