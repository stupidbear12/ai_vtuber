from fastapi import FastAPI
import llm.tf_setup

app = FastAPI()


@app.get("/")

def read_root():
       return{"message" : "AI VTuber Server Running!"}

@app.get("/ping")
def ping():
     return {"pong": True}

