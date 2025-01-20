# asr-conformer-inference

To run as an api service, add the following to docker file
- this will start up the service on port 8080
```
    ports:
      - 8080:8000 # out: in
    command: ["uvicorn", "src.main:api", "--host", "0.0.0.0", "--port", "8000"] # container port
```

### For site
To run inference on an audio directory with or without LM

1. Spin up the docker container
```
docker compose -f build/docker-compose.yaml up -d
docker exec -it conformer-inference bash
```

2. Edit `src/config.py` to update
- language
- asr & LM model path
- audio directory
- output manifest path

3. Run main script for inference
```
python3 src/predict.py
```