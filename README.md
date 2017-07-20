# fcn

#### fcn-train.py: FCN training program.
Parameters:
- db: picpac image db, with json annotation
- mixin: picpac image db, without any annotation. Images are used as negative example.
- net: network type, pick up one from net.py
- val: validation picpac db.

#### fcn-cls-train.py: train a network with both FCN and classification branches. 
Annotation in FCN training data accelerates classifier learning process. And classification output will give more relable score.
Recommand to use picpac config "max_size=300" or "max_size=400". High-resolution images are not good training data in FCN, and could make training process extremely slow.

#### validation
`python fcn-val.py --model_snapshot_directory/100000 --out output_dir --db test_db`
