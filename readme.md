# Init docker image

```
docker run \
    --gpus all \
    --shm-size 1g \
    -v /home/eperezp1990/.temp:/home/eperez \
    --name pytorch-2 \
    -dit \
    eperezp1990/pytorch:2.2.0-cuda11.4-p310-min
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

# Experimental study

No categories:

```
aae/
began/
bgan/
dcgan/
```

Categories:

```
acgan/
cgan/
```

Issues:

```
bicyclegan/
ccgan/
cogan/
context_encoder/ dado un pedazo de imagen, la completa.
cyclegan/ depende de un validation dataset
```