It's essence code from NanoGPT simplified to our Linear Algebra project.

# Downloading the data
```bash
python data/shakespeare_char/prepare.py
```

```bash
python train.py config/train_shakespeare_char.py

python train.py config/train_shakespeare_char.py --attention_type=linformer
```

```bash
python sample.py --out_dir=out-shakespeare-char
```
