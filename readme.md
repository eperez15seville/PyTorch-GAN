# Init docker image

```
docker run \
    --gpus all \
    -v /home/eperezp1990/.temp:/home/eperez \
    --name pytorch-2 -dit eperezp1990/pytorch:2.2.0-cuda11.4
```

# Execute experiment

```
export PYTHONPATH=.
nohup python3 acgan/acgan_skin.py &
```

# Copy results

```
scp -r Tesla:/home/eperezp1990/.temp/gans/acgan/images/0799.png images/
scp -r Tesla:/home/eperezp1990/.temp/gans/acgan/images/0799/00000.png images/
```