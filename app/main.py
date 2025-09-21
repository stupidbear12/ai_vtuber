from fastapi import FastAPI

app = FastAPI()


@app.get("/")

def read_root():
       return{"message" : "AI VTuber Server Running!"}

@app.get("/ping")
def ping():
     return {"pong": True}

