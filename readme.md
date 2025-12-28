# Init docker image

```
docker run \
    --gpus all \
    -v /home/eperezp1990/.temp:/home/eperez \
    --name pytorch-2 -dit eperezp1990/pytorch:2.2.0-cuda11.4
```