"""Dataset plumbing: manifest, splits, taxonomy, feature cache.

Modules that belong here:
  taxonomy.py       folder labels -> 3-class reporting
  manifest.py       build/read data/dataset/manifest.csv
  splits.py         grouped, leave-one-venue-out, temporal;
                    materialised to configs/splits/, and the
                    locked FINAL_TESTSET is enforced here
  feature_cache.py  embed each image once per backbone
  dedup.py          perceptual-hash near-duplicate removal
"""
