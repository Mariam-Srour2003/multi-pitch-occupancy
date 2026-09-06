"""Dataset plumbing: manifest, splits, taxonomy, feature cache.

Modules that belong here:
  taxonomy.py       4-class labels -> 3-class reporting          (WP0-T3)  [done]
  manifest.py       build/read data/dataset/manifest.csv         (WP0-T2)
  splits.py         grouped, leave-one-venue-out, temporal;
                    materialised to results/splits/, and the
                    locked FINAL_TESTSET is enforced here        (WP0-T4)
  feature_cache.py  embed each image once per backbone           (WP0-T5)
  dedup.py          perceptual-hash near-duplicate removal       (WP2-T4)
"""
