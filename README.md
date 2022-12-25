
## Training
```
python main.py
```

## Evaluation
```
python main.py --eval --resume path/to/checkpoint
```

## Testing
```
python test.py --resume path/to/checkpoint
```

## Checkpoint link
- Sparse DETR
https://drive.google.com/file/d/1EkhefZQBE4OIndyHeEjgCG4uw0mc5fms/view?usp=sharing

## Datasets Configuration
```
/ycbv_BOP
    /annotations
        /train.json
        /test.json
    /models
    /models_eval
    /models_fine
    /test
    /train_pbr
    /ycbv

/YCB_Video_Dataset
    /image_sets
    /models
```